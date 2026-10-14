"""Evaluate phase: run a resolved :class:`~sparkleframe.python.ast.expressions.Expression`
against data.

This is the final phase of build → analyze → evaluate (see ``docs/design/python-engine-ast.md``).
The evaluator consumes an expression already resolved by :mod:`sparkleframe.python.ast.analyzer`
(so every node has a known ``data_type`` and every column reference a row position) and produces
the resulting column values.

Evaluation is **row-wise**: :func:`eval_row` computes one row's value. That mirrors Spark's
interpreted evaluation and gives its short-circuit semantics for free — ``when`` only evaluates
the branch it takes, ``coalesce`` stops at the first non-null, ``and`` / ``or`` skip their right
side — so an ANSI error in a branch Spark never reaches is never raised here either.

An :class:`EvalContext` carries what is shared across rows of one query: the query's
``current_timestamp``, the row position, window-function results and aggregate results the
DataFrame layer precomputes, lambda variables, and random-generator state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    CaseWhen,
    Cast,
    Expression,
    FunctionCall,
    LambdaVariable,
    Literal,
    UnaryExpression,
    WindowExpression,
)
from sparkleframe.python.ast.function_registry import lookup
from sparkleframe.python.column_helpers import (
    _INTEGRAL_RANGE,
    arithmetic,
    cast_value,
    compare_values,
    logical_and,
    logical_not,
    logical_or,
)
from sparkleframe.python.functions_helpers import spark_pow
from sparkleframe.python.types import ByteType, IntegerType, LongType, ShortType


class EvalContext:
    """Per-query evaluation state shared by every row."""

    def __init__(self, now: Optional[datetime] = None) -> None:
        self.now = now or datetime.now()
        self.row_index = 0
        self.window_values: Dict[int, List[Any]] = {}
        self.agg_values: Dict[int, Any] = {}
        self.lambda_values: Dict[int, Any] = {}
        self.random_state: Dict[int, Any] = {}

    def eval(self, expr: Expression, row: Any) -> Any:
        return eval_row(expr, row, self)


def evaluate(expr: Expression, rows: List[Any], ctx: Optional[EvalContext] = None) -> List[Any]:
    """Run a resolved ``expr`` against ``rows`` and return one value per row.

    Args:
        expr: An expression already resolved by :func:`sparkleframe.python.ast.analyzer.analyze`.
        rows: The frame's rows: tuples (positions bound by the analyzer) or dicts keyed by name.
        ctx: Shared per-query state; a fresh one is created when omitted.
    """
    ctx = ctx or EvalContext()
    out = []
    for index, row in enumerate(rows):
        ctx.row_index = index
        out.append(eval_row(expr, row, ctx))
    return out


def eval_row(expr: Expression, row: Any, ctx: EvalContext) -> Any:
    handler = _HANDLERS.get(type(expr))
    if handler is None:
        detail = f"{type(expr).__name__} {getattr(expr, 'op', '')}".strip()
        not_implemented_yet(f"the evaluate phase for {detail}")
    return handler(expr, row, ctx)


def _attribute(expr: AttributeReference, row: Any, ctx: EvalContext) -> Any:
    return row[expr.name] if isinstance(row, dict) else row[expr.index]


def _literal(expr: Literal, row: Any, ctx: EvalContext) -> Any:
    return expr.value


def _alias(expr: Alias, row: Any, ctx: EvalContext) -> Any:
    return eval_row(expr.child, row, ctx)


def _cast(expr: Cast, row: Any, ctx: EvalContext) -> Any:
    return cast_value(eval_row(expr.child, row, ctx), expr.target_type, expr.strict, expr.child.data_type)


def _unary(expr: UnaryExpression, row: Any, ctx: EvalContext) -> Any:
    value = eval_row(expr.child, row, ctx)
    op = expr.op
    if op == "not":
        return logical_not(value)
    if op == "isnull":
        return value is None
    if op == "isnotnull":
        return value is not None
    if op == "pos" or value is None:
        return value
    if op == "neg":
        dtype = expr.data_type
        integral = (ByteType, ShortType, IntegerType, LongType)
        if isinstance(dtype, integral) and -value > _INTEGRAL_RANGE[type(dtype)][1]:
            raise ArithmeticError(f"[ARITHMETIC_OVERFLOW] {dtype.simpleString()} overflow.")
        return -value
    not_implemented_yet(f"the evaluate phase for UnaryExpression {op}")


def _binary(expr: BinaryExpression, row: Any, ctx: EvalContext) -> Any:
    op = expr.op
    if op == "and":
        left = eval_row(expr.left, row, ctx)
        if left is False:
            return False
        return logical_and(left, eval_row(expr.right, row, ctx))
    if op == "or":
        left = eval_row(expr.left, row, ctx)
        if left is True:
            return True
        return logical_or(left, eval_row(expr.right, row, ctx))
    left = eval_row(expr.left, row, ctx)
    right = eval_row(expr.right, row, ctx)
    if op in ("+", "-", "*", "/"):
        return arithmetic(op, left, right, expr.data_type)
    if op == "**":
        return spark_pow(left, right)
    if op == "<=>":
        if left is None or right is None:
            return left is None and right is None
        return compare_values("==", left, right, expr.left.data_type)
    # The analyzer coerced both operands to one type; ordering needs it (NaN, arrays, structs).
    return compare_values(op, left, right, expr.left.data_type)


def _case_when(expr: CaseWhen, row: Any, ctx: EvalContext) -> Any:
    for condition, value in expr.branches:
        if eval_row(condition, row, ctx) is True:
            return eval_row(value, row, ctx)
    return eval_row(expr.otherwise, row, ctx) if expr.otherwise is not None else None


def _function(expr: FunctionCall, row: Any, ctx: EvalContext) -> Any:
    key = id(expr)
    if key in ctx.agg_values:
        return ctx.agg_values[key]
    spec = lookup(expr.name)
    if spec.kind != "scalar":
        raise RuntimeError(f"{spec.kind} function {expr.name}() evaluated outside its DataFrame operation")
    if spec.lazy:
        return spec.evaluate(expr, row, ctx)
    values = [eval_row(child, row, ctx) for child in expr.children]
    if spec.null_propagating and any(v is None for v in values):
        return None
    return spec.evaluate(values, expr, ctx)


def _window(expr: WindowExpression, row: Any, ctx: EvalContext) -> Any:
    return ctx.window_values[id(expr)][ctx.row_index]


def _lambda_variable(expr: LambdaVariable, row: Any, ctx: EvalContext) -> Any:
    return ctx.lambda_values[expr.slot]


_HANDLERS: Dict[type, Callable[[Any, Any, EvalContext], Any]] = {
    AttributeReference: _attribute,
    Literal: _literal,
    Alias: _alias,
    Cast: _cast,
    UnaryExpression: _unary,
    BinaryExpression: _binary,
    CaseWhen: _case_when,
    FunctionCall: _function,
    WindowExpression: _window,
    LambdaVariable: _lambda_variable,
}
