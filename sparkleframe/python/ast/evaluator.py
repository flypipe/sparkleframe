"""Evaluate phase: run a resolved :class:`~sparkleframe.python.ast.expressions.Expression`
against data.

This is the final phase of build → analyze → evaluate (see ``docs/design/python-engine-ast.md``).
The evaluator consumes an expression already resolved by :mod:`sparkleframe.python.ast.analyzer`
(so every node has a known ``data_type``) and produces the resulting column values.

:func:`evaluate` works **column-wise**: given the frame's rows (as dicts) it returns one value
per row for the expression. It dispatches on node kind plus ``op`` and computes Python values
directly.

This is the seam where per-feature execution lands. Implementing a feature here (plus
:mod:`sparkleframe.python.ast.analyzer`) is what lets its id be deleted from
``PYTHON_NOT_IMPLEMENTED`` in ``sparkleframe/tests/parity/gaps.py``.
"""

from __future__ import annotations

from typing import Any, List

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    Cast,
    Expression,
    Literal,
    UnaryExpression,
)
from sparkleframe.python.column_helpers import cast_value, compare_values, logical_and, logical_not, logical_or
from sparkleframe.python.functions_helpers import spark_pow


def evaluate(expr: Expression, rows: List[dict]) -> List[Any]:
    """Run a resolved ``expr`` against ``rows`` and return one value per row.

    Args:
        expr: An expression already resolved by :func:`sparkleframe.python.ast.analyzer.analyze`.
        rows: The frame's rows as dicts keyed by column name.
    """
    if isinstance(expr, AttributeReference):
        return [row[expr.name] for row in rows]

    if isinstance(expr, Literal):
        return [expr.value for _ in rows]

    if isinstance(expr, Alias):
        return evaluate(expr.child, rows)

    if isinstance(expr, Cast):
        return [cast_value(v, expr.target_type, expr.strict) for v in evaluate(expr.child, rows)]

    if isinstance(expr, UnaryExpression) and expr.op == "not":
        return [logical_not(v) for v in evaluate(expr.child, rows)]

    if isinstance(expr, BinaryExpression):
        pairs = zip(evaluate(expr.left, rows), evaluate(expr.right, rows))
        if expr.op in ("+", "-", "*"):
            return [_arithmetic(expr.op, lv, rv) for lv, rv in pairs]
        if expr.op == "**":
            return [spark_pow(lv, rv) for lv, rv in pairs]
        if expr.op in ("==", "!=", "<", "<=", ">", ">="):
            # The analyzer coerced both operands to one type; ordering needs it (NaN, arrays, structs).
            return [compare_values(expr.op, lv, rv, expr.left.data_type) for lv, rv in pairs]
        if expr.op == "and":
            return [logical_and(lv, rv) for lv, rv in pairs]
        if expr.op == "or":
            return [logical_or(lv, rv) for lv, rv in pairs]

    detail = f"{type(expr).__name__} {getattr(expr, 'op', '')}".strip()
    not_implemented_yet(f"the evaluate phase for {detail}")


def _arithmetic(op: str, left: Any, right: Any) -> Any:
    """Apply ``+ - *`` element-wise; any null operand yields null (Spark semantics)."""
    if left is None or right is None:
        return None
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    return left * right
