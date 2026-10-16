"""``Column`` for the pure-Python engine — the public face of the **build** phase.

A :class:`Column` wraps exactly one :class:`~sparkleframe.python.ast.expressions.Expression`.
Operators and transforms build AST nodes and make **zero type decisions** (the schema is not
known yet). Producing an actual result happens later, when a :class:`DataFrame` operation runs
the expression through ``analyzer.analyze`` then ``evaluator.evaluate``.
"""

from __future__ import annotations

from typing import Any, Union

from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    CaseWhen,
    Cast,
    Expression,
    FunctionCall,
    Literal,
    SortOrder,
    UnaryExpression,
    WindowExpression,
)


class Column:
    def __init__(self, expr: Expression) -> None:
        if not isinstance(expr, Expression):
            raise TypeError(f"Column expects an Expression, got {type(expr).__name__}")
        self._expr = expr

    # --- Arithmetic ---------------------------------------------------------
    def __add__(self, other: Any) -> "Column":
        return Column(BinaryExpression("+", self._expr, _to_expression(other)))

    def __sub__(self, other: Any) -> "Column":
        return Column(BinaryExpression("-", self._expr, _to_expression(other)))

    def __mul__(self, other: Any) -> "Column":
        return Column(BinaryExpression("*", self._expr, _to_expression(other)))

    def __truediv__(self, other: Any) -> "Column":
        return Column(BinaryExpression("/", self._expr, _to_expression(other)))

    def __pow__(self, other: Any) -> "Column":
        return Column(BinaryExpression("**", self._expr, _to_expression(other)))

    def __radd__(self, other: Any) -> "Column":
        return Column(BinaryExpression("+", _to_expression(other), self._expr))

    def __rsub__(self, other: Any) -> "Column":
        return Column(BinaryExpression("-", _to_expression(other), self._expr))

    def __rmul__(self, other: Any) -> "Column":
        return Column(BinaryExpression("*", _to_expression(other), self._expr))

    def __rtruediv__(self, other: Any) -> "Column":
        return Column(BinaryExpression("/", _to_expression(other), self._expr))

    def __rpow__(self, other: Any) -> "Column":
        return Column(BinaryExpression("**", _to_expression(other), self._expr))

    def __neg__(self) -> "Column":
        return Column(UnaryExpression("neg", self._expr))

    def __pos__(self) -> "Column":
        return Column(UnaryExpression("pos", self._expr))

    # --- Comparison ---------------------------------------------------------
    def __eq__(self, other: Any) -> "Column":  # type: ignore[override]
        return Column(BinaryExpression("==", self._expr, _to_expression(other)))

    def __ne__(self, other: Any) -> "Column":  # type: ignore[override]
        return Column(BinaryExpression("!=", self._expr, _to_expression(other)))

    def __lt__(self, other: Any) -> "Column":
        return Column(BinaryExpression("<", self._expr, _to_expression(other)))

    def __le__(self, other: Any) -> "Column":
        return Column(BinaryExpression("<=", self._expr, _to_expression(other)))

    def __gt__(self, other: Any) -> "Column":
        return Column(BinaryExpression(">", self._expr, _to_expression(other)))

    def __ge__(self, other: Any) -> "Column":
        return Column(BinaryExpression(">=", self._expr, _to_expression(other)))

    def eqNullSafe(self, other: Any) -> "Column":
        """``<=>``: null-safe equality (two nulls are equal; null vs value is false, never null)."""
        return Column(BinaryExpression("<=>", self._expr, _to_expression(other)))

    def between(self, lowerBound: Any, upperBound: Any) -> "Column":
        return (self >= lowerBound) & (self <= upperBound)

    # Column overrides __eq__, so it is unhashable by default — mirror PySpark.
    __hash__ = None  # type: ignore[assignment]

    # --- Logical ------------------------------------------------------------
    def __and__(self, other: Any) -> "Column":
        return Column(BinaryExpression("and", self._expr, _to_expression(other)))

    def __rand__(self, other: Any) -> "Column":
        return Column(BinaryExpression("and", _to_expression(other), self._expr))

    def __or__(self, other: Any) -> "Column":
        return Column(BinaryExpression("or", self._expr, _to_expression(other)))

    def __ror__(self, other: Any) -> "Column":
        return Column(BinaryExpression("or", _to_expression(other), self._expr))

    def __invert__(self) -> "Column":
        return Column(UnaryExpression("not", self._expr))

    def __bool__(self) -> bool:
        raise ValueError(
            "[CANNOT_CONVERT_COLUMN_INTO_BOOL] Cannot convert column into bool: please use '&' for 'and', '|' for "
            "'or', '~' for 'not' when building DataFrame boolean expressions."
        )

    def __iter__(self) -> None:
        raise TypeError("[NOT_ITERABLE] Column is not iterable.")

    def __contains__(self, item: Any) -> bool:
        raise ValueError(
            "[CANNOT_APPLY_IN_FOR_COLUMN] Cannot apply 'in' operator against a column: please use 'contains' in a "
            "string column or 'array_contains' function for an array column."
        )

    # --- Transforms ---------------------------------------------------------
    def alias(self, *alias: str) -> "Column":
        node = Alias(self._expr, alias[0])
        node.names = tuple(alias)  # ``explode(map).alias("k", "v")`` names every generated column
        return Column(node)

    name = alias

    def cast(self, data_type: Any) -> "Column":
        """ANSI-strict cast: raises on malformed input at evaluate time (Spark 4 default)."""
        return Column(Cast(self._expr, data_type, strict=True))

    def try_cast(self, data_type: Any) -> "Column":
        """Lenient cast (Spark 4 ``try_cast``): returns null instead of raising."""
        return Column(Cast(self._expr, data_type, strict=False))

    def isNull(self) -> "Column":
        return Column(UnaryExpression("isnull", self._expr))

    def isNotNull(self) -> "Column":
        return Column(UnaryExpression("isnotnull", self._expr))

    def isin(self, *values: Any) -> "Column":
        if len(values) == 1 and isinstance(values[0], (list, tuple, set)):
            values = tuple(values[0])
        return Column(FunctionCall("in", [self._expr, *(_to_expression(v) for v in values)]))

    def rlike(self, pattern: str) -> "Column":
        return Column(FunctionCall("rlike", [self._expr], {"pattern": pattern}))

    def contains(self, other: Any) -> "Column":
        return Column(FunctionCall("contains", [self._expr, _to_expression(other)]))

    def startswith(self, other: Any) -> "Column":
        return Column(FunctionCall("startswith", [self._expr, _to_expression(other)]))

    def endswith(self, other: Any) -> "Column":
        return Column(FunctionCall("endswith", [self._expr, _to_expression(other)]))

    def substr(self, startPos: Any, length: Any) -> "Column":
        return Column(FunctionCall("substring", [self._expr, _to_expression(startPos), _to_expression(length)]))

    def getItem(self, key: Any) -> "Column":
        return Column(FunctionCall("get_item", [self._expr, _to_expression(key)]))

    def getField(self, name: str) -> "Column":
        return Column(FunctionCall("get_field", [self._expr], {"name": name}))

    def __getitem__(self, key: Any) -> "Column":
        if isinstance(key, slice):
            return self.substr(key.start, key.stop)
        return self.getItem(key)

    def __getattr__(self, item: str) -> "Column":
        if item.startswith("__"):
            raise AttributeError(item)
        return self.getField(item)

    # --- Sort orders --------------------------------------------------------
    def asc(self) -> "Column":
        return Column(SortOrder(self._expr, ascending=True, nulls_first=True))

    def asc_nulls_first(self) -> "Column":
        return Column(SortOrder(self._expr, ascending=True, nulls_first=True))

    def asc_nulls_last(self) -> "Column":
        return Column(SortOrder(self._expr, ascending=True, nulls_first=False))

    def desc(self) -> "Column":
        return Column(SortOrder(self._expr, ascending=False, nulls_first=False))

    def desc_nulls_first(self) -> "Column":
        return Column(SortOrder(self._expr, ascending=False, nulls_first=True))

    def desc_nulls_last(self) -> "Column":
        return Column(SortOrder(self._expr, ascending=False, nulls_first=False))

    # --- Conditional / window -----------------------------------------------
    def when(self, condition: Any, value: Any) -> "Column":
        case = self._open_case_when("when")
        return Column(CaseWhen([*case.branches, (_to_expression(condition), _to_expression(value))]))

    def otherwise(self, value: Any) -> "Column":
        case = self._open_case_when("otherwise")
        return Column(CaseWhen(case.branches, _to_expression(value)))

    def _open_case_when(self, method: str) -> CaseWhen:
        if not isinstance(self._expr, CaseWhen) or self._expr.otherwise is not None:
            raise TypeError(f"{method}() can only be applied on a Column previously generated by when() function.")
        return self._expr

    def over(self, window: Any) -> "Column":
        return Column(WindowExpression(self._expr, window._partition_exprs(), window._order_exprs()))

    def __repr__(self) -> str:
        return f"Column({self._expr!r})"


def _to_expression(value: Any) -> Expression:
    """Coerce an operand into an :class:`Expression` for the build phase.

    A :class:`Column` contributes its wrapped expression; a raw :class:`Expression` is used
    as-is; anything else is a constant literal. (Column *names* become references only via
    :func:`sparkleframe.python.functions.col`, never here.)
    """
    if isinstance(value, Column):
        return value._expr
    if isinstance(value, Expression):
        return value
    return Literal(value)


def _column_ref(value: Union[str, Column, Any]) -> Expression:
    """Coerce a ``str | Column`` 'column argument' into an :class:`Expression`.

    Unlike :func:`_to_expression`, a bare ``str`` here means a *column name* (the convention
    for ``functions`` like ``abs(col_name)``), so it becomes an :class:`AttributeReference`.
    """
    if isinstance(value, str):
        return AttributeReference(value)
    return _to_expression(value)
