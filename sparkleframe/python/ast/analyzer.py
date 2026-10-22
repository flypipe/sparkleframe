"""Analyze phase: resolve an unresolved :class:`~sparkleframe.python.ast.expressions.Expression`
against a schema.

This is the middle phase of build → analyze → evaluate (see ``docs/design/python-engine-ast.md``).
Because resolution happens *here* — where the schema is in hand — the "dtype unknown at build
time" state that limits the Polars engine cannot occur.

:func:`analyze` walks the unresolved tree and returns a **new** tree of the same node classes
with each node's ``data_type`` slot filled and coercion :class:`Cast` nodes inserted where Spark
would. The original build tree is left untouched; "resolved" means every returned node has
``data_type is not None``. Column references are bound to a position in the frame's row tuples,
so frames with duplicate column names (after a join) resolve exactly like Spark: a ``df["x"]``
reference picks the column that came from ``df``, and an unqualified ambiguous name raises.

Per-function rules live in :mod:`sparkleframe.python.ast.function_registry`; implementing a feature
there (plus its evaluate routine) is what lets its id be deleted from ``PYTHON_NOT_IMPLEMENTED`` in
``sparkleframe/tests/parity/gaps.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast import coercion
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    CaseWhen,
    Cast,
    Expression,
    FunctionCall,
    LambdaFunction,
    LambdaVariable,
    Literal,
    SortOrder,
    Star,
    UnaryExpression,
    WindowExpression,
)
from sparkleframe.python.ast.function_registry import lookup
from sparkleframe.python.types import (
    BooleanType,
    ByteType,
    DecimalType,
    DoubleType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    ShortType,
    StringType,
    parse_ddl_type,
)


class AnalysisException(Exception):
    """Raised for Spark analysis errors (unresolved / ambiguous columns)."""


def analyze(expr: Expression, schema: Any, plan_sets: Optional[Sequence[frozenset]] = None) -> Expression:
    """Resolve ``expr``'s output types against ``schema`` and return the resolved tree.

    Args:
        expr: The unresolved expression produced by the build phase.
        schema: The engine ``StructType`` of the frame the expression is applied to.
        plan_sets: Per column, the ids of the frames it can be referenced through (``df["x"]``).
    """
    return _Analyzer(schema, plan_sets).resolve(expr)


class _Analyzer:
    def __init__(self, schema: Any, plan_sets: Optional[Sequence[frozenset]]) -> None:
        self.schema = schema
        self.plan_sets = plan_sets
        self.lambda_depth = 0

    def resolve(self, expr: Expression) -> Expression:
        handler = _HANDLERS.get(type(expr))
        if handler is None:
            detail = f"{type(expr).__name__} {getattr(expr, 'op', '')}".strip()
            not_implemented_yet(f"the analyze phase for {detail}")
        return handler(self, expr)

    # --- columns -------------------------------------------------------------
    def _attribute(self, expr: AttributeReference) -> Expression:
        if expr.index is not None and expr.data_type is not None:
            return expr  # already bound by the DataFrame layer (star expansion, pass-through columns)
        if self.schema is None:
            raise AnalysisException(f"Cannot resolve column {expr.name!r}: no schema in hand")
        index = self._column_index(expr.name, expr.plan_id)
        if index is not None:
            return self._bound(expr.name, expr.plan_id, index)
        parts = _split_name(expr.name)
        if len(parts) > 1:
            base_index = self._column_index(parts[0], expr.plan_id)
            if base_index is not None:
                return self._nested(self._bound(parts[0], expr.plan_id, base_index), parts[1:])
        raise AnalysisException(
            f"[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column, variable, or function parameter with name "
            f"`{expr.name}` cannot be resolved. Did you mean one of the following? "
            f"[{', '.join(f'`{f.name}`' for f in self.schema.fields)}]."
        )

    def _bound(self, name: str, plan_id: Optional[int], index: int) -> AttributeReference:
        resolved = AttributeReference(self.schema.fields[index].name, plan_id)
        resolved.index = index
        resolved.data_type = self.schema.fields[index].dataType
        return resolved

    def _column_index(self, name: str, plan_id: Optional[int]) -> Optional[int]:
        fields = self.schema.fields
        unquoted = name[1:-1] if len(name) > 1 and name.startswith("`") and name.endswith("`") else name
        candidates = [i for i, f in enumerate(fields) if f.name == unquoted]
        if not candidates:
            candidates = [i for i, f in enumerate(fields) if f.name.lower() == unquoted.lower()]
        if plan_id is not None and self.plan_sets is not None:
            narrowed = [i for i in candidates if plan_id in self.plan_sets[i]]
            candidates = narrowed or candidates
        if len(candidates) > 1:
            raise AnalysisException(
                f"[AMBIGUOUS_REFERENCE] Reference `{name}` is ambiguous, could be: "
                f"[{', '.join(f'`{fields[i].name}`' for i in candidates)}]."
            )
        return candidates[0] if candidates else None

    def _nested(self, base: Expression, parts: List[str]) -> Expression:
        resolved = base
        for part in parts:
            if isinstance(resolved.data_type, MapType):
                resolved = self.resolve(FunctionCall("get_item", [_Resolved(resolved), Literal(part)]))
            else:
                resolved = self.resolve(FunctionCall("get_field", [_Resolved(resolved)], {"name": part}))
        return resolved

    # --- simple nodes --------------------------------------------------------
    def _literal(self, expr: Literal) -> Expression:
        return Literal(expr.value, data_type=expr.data_type or coercion.infer_literal_type(expr.value))

    def _alias(self, expr: Alias) -> Expression:
        child = self.resolve(expr.child)
        resolved = Alias(child, expr.name)
        resolved.data_type = child.data_type
        return resolved

    def _cast(self, expr: Cast) -> Expression:
        child = self.resolve(expr.child)
        target = parse_ddl_type(expr.target_type) if isinstance(expr.target_type, str) else expr.target_type
        mismatch = coercion.ansi_cast_error(child.data_type, target)
        if mismatch is not None:
            source_name, target_name = child.data_type.simpleString().upper(), target.simpleString().upper()
            raise AnalysisException(
                f'[DATATYPE_MISMATCH.{mismatch}] Cannot resolve "CAST({expr.child!r} AS {target_name})" due to data '
                f'type mismatch: cannot cast "{source_name}" to "{target_name}" with ANSI mode on.'
            )
        resolved = Cast(child, target, strict=expr.strict)
        resolved.data_type = target
        return resolved

    def _sort_order(self, expr: SortOrder) -> Expression:
        child = self.resolve(expr.child)
        if not coercion.is_orderable(child.data_type):
            raise AnalysisException(
                f'[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] Cannot resolve "{expr.child!r}" due to data type '
                f"mismatch: The `sortorder` does not support ordering on type "
                f'"{child.data_type.simpleString().upper()}".'
            )
        resolved = SortOrder(child, expr.ascending, expr.nulls_first)
        resolved.data_type = child.data_type
        return resolved

    def _lambda_variable(self, expr: LambdaVariable) -> Expression:
        return expr

    def _passthrough(self, expr: "_Resolved") -> Expression:
        return expr.resolved

    # --- operators -----------------------------------------------------------
    def _binary(self, expr: BinaryExpression) -> Expression:
        left = self.resolve(expr.left)
        right = self.resolve(expr.right)
        op = expr.op
        if op in ("+", "-", "*", "/"):
            left_dt, right_dt = _narrow_literal(left, right), _narrow_literal(right, left)
            result_dt, left_cast, right_cast = coercion.coerce_arithmetic(op, left_dt, right_dt)
            if left_cast is None and left_dt is not left.data_type:
                left_cast = left_dt
            if right_cast is None and right_dt is not right.data_type:
                right_cast = right_dt
        elif op == "**":
            result_dt, left_cast, right_cast = coercion.coerce_pow(left.data_type, right.data_type)
        elif op in ("==", "!=", "<", "<=", ">", ">=", "<=>"):
            result_dt, left_cast, right_cast = coercion.coerce_comparison(left.data_type, right.data_type)
        elif op in ("and", "or"):
            result_dt, left_cast, right_cast = coercion.coerce_logical(op, left.data_type, right.data_type)
        else:
            not_implemented_yet(f"the analyze phase for BinaryExpression {op}")
        resolved = BinaryExpression(op, _with_cast(left, left_cast), _with_cast(right, right_cast))
        resolved.data_type = result_dt
        return resolved

    def _unary(self, expr: UnaryExpression) -> Expression:
        child = self.resolve(expr.child)
        op = expr.op
        if op == "not":
            result_dt, child_cast = coercion.coerce_not(child.data_type)
            child = _with_cast(child, child_cast)
        elif op in ("isnull", "isnotnull"):
            result_dt = BooleanType()
        elif op in ("neg", "pos"):
            if isinstance(child.data_type, (StringType, NullType)):
                child = _with_cast(child, DoubleType())
            elif not (coercion.is_numeric(child.data_type) or isinstance(child.data_type, DecimalType)):
                raise TypeError(
                    f'[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve "(- {expr.child!r})": the parameter '
                    f'requires a numeric type, however it has the type "{child.data_type.simpleString().upper()}".'
                )
            result_dt = child.data_type
        else:
            not_implemented_yet(f"the analyze phase for UnaryExpression {op}")
        resolved = UnaryExpression(op, child)
        resolved.data_type = result_dt
        return resolved

    def _case_when(self, expr: CaseWhen) -> Expression:
        conditions = []
        for condition, _ in expr.branches:
            resolved_condition = self.resolve(condition)
            if isinstance(resolved_condition.data_type, NullType):
                resolved_condition = _with_cast(resolved_condition, BooleanType())
            elif not isinstance(resolved_condition.data_type, BooleanType):
                raise TypeError(
                    f'[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve "CASE WHEN": the WHEN expressions '
                    f'require the "BOOLEAN" type, however it has the type '
                    f'"{resolved_condition.data_type.simpleString().upper()}".'
                )
            conditions.append(resolved_condition)
        values = [self.resolve(value) for _, value in expr.branches]
        otherwise = self.resolve(expr.otherwise) if expr.otherwise is not None else None
        types = [v.data_type for v in values] + ([otherwise.data_type] if otherwise is not None else [])
        common = coercion.find_wider_common_type(types)
        if common is None:
            raise TypeError(
                f'[DATATYPE_MISMATCH.DATA_DIFF_TYPES] Cannot resolve "CASE WHEN" due to data type mismatch: Input to '
                f"`casewhen` should all be the same type, but it's "
                f"[{', '.join(t.simpleString().upper() for t in types)}]."
            )
        values = [_with_cast(v, None if v.data_type == common else common) for v in values]
        if otherwise is not None:
            otherwise = _with_cast(otherwise, None if otherwise.data_type == common else common)
        resolved = CaseWhen(list(zip(conditions, values)), otherwise)
        resolved.data_type = common
        return resolved

    # --- functions -----------------------------------------------------------
    def _function(self, expr: FunctionCall) -> Expression:
        spec = lookup(expr.name)
        args = [a if isinstance(a, (LambdaFunction, Star)) else self.resolve(a) for a in expr.args]
        resolved = FunctionCall(expr.name, args, expr.options)
        resolved.data_type = spec.resolve(resolved, self)
        return resolved

    def analyze_lambda(self, func: LambdaFunction, params: List[LambdaVariable]) -> Expression:
        """Build and resolve a lambda body with ``params`` typed and bound to fresh scope slots."""
        from sparkleframe.python.column import Column, _to_expression

        for offset, param in enumerate(params):
            param.slot = self.lambda_depth + offset
        self.lambda_depth += len(params)
        try:
            body = func.func(*[Column(p) for p in params])
            return self.resolve(_to_expression(body))
        finally:
            self.lambda_depth -= len(params)

    def _window(self, expr: WindowExpression) -> Expression:
        function = self.resolve(expr.function)
        if not isinstance(function, FunctionCall) or lookup(function.name).kind != "window":
            not_implemented_yet(f"window aggregate {function!r}")
        partition_by = [self.resolve(p) for p in expr.partition_by]
        order_by = [self.resolve(o) for o in expr.order_by]
        if not order_by:
            raise AnalysisException(
                f"[WINDOW_FUNCTION_WITHOUT_ORDER_CLAUSE] Window function {function.name}() requires an ORDER BY "
                f"clause in the window specification."
            )
        resolved = WindowExpression(function, partition_by, order_by)
        resolved.data_type = function.data_type
        return resolved

    def _star(self, expr: Star) -> Expression:
        raise AnalysisException("[INVALID_USAGE_OF_STAR_OR_REGEX] Invalid usage of '*' in expression.")


class _Resolved(Expression):
    """Wraps an already-resolved node so it can be fed back through :meth:`_Analyzer.resolve`."""

    def __init__(self, resolved: Expression) -> None:
        super().__init__()
        self.resolved = resolved


_HANDLERS: Dict[type, Callable[[_Analyzer, Any], Expression]] = {
    AttributeReference: _Analyzer._attribute,
    Literal: _Analyzer._literal,
    Alias: _Analyzer._alias,
    Cast: _Analyzer._cast,
    BinaryExpression: _Analyzer._binary,
    UnaryExpression: _Analyzer._unary,
    CaseWhen: _Analyzer._case_when,
    FunctionCall: _Analyzer._function,
    SortOrder: _Analyzer._sort_order,
    WindowExpression: _Analyzer._window,
    LambdaVariable: _Analyzer._lambda_variable,
    Star: _Analyzer._star,
    _Resolved: _Analyzer._passthrough,
}


def _split_name(name: str) -> List[str]:
    """Split ``a.b.c`` on dots outside back-ticks (``a.`b.c``` is two parts)."""
    parts: List[str] = []
    current: List[str] = []
    quoted = False
    for ch in name:
        if ch == "`":
            quoted = not quoted
        elif ch == "." and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _narrow_literal(operand: Expression, other: Expression) -> Any:
    """Spark's ``DecimalPrecision`` rule: an integral literal next to a decimal uses its own digits.

    ``price / 100`` is ``decimal / decimal(3,0)``, not ``decimal / decimal(10,0)``.
    """
    if (
        isinstance(operand, Literal)
        and isinstance(operand.data_type, (ByteType, ShortType, IntegerType, LongType))
        and isinstance(other.data_type, DecimalType)
        and operand.value is not None
    ):
        return coercion.literal_decimal_type(operand.value)
    return operand.data_type


def _with_cast(resolved: Expression, target_dt: Any) -> Expression:
    """Wrap ``resolved`` in a strict :class:`Cast` to ``target_dt`` (or return it unchanged)."""
    if target_dt is None:
        return resolved
    cast = Cast(resolved, target_dt, strict=True)
    cast.data_type = target_dt
    return cast
