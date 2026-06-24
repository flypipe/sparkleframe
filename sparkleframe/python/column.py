"""``Column`` for the pure-Python engine — the public face of the **build** phase.

A :class:`Column` wraps exactly one :class:`~sparkleframe.python.ast.expressions.Expression`.
Operators and transforms build AST nodes and make **zero type decisions** (the schema is not
known yet). Producing an actual result happens later, when a :class:`DataFrame` action runs
the expression through ``analyzer.analyze`` then ``evaluator.evaluate``.

Methods that map cleanly onto the generic node set (arithmetic, comparison, logical, cast,
alias, null checks) are wired. The remaining transforms are declared with their PySpark
signatures and raise — build the matching node here when implementing them.
"""

from __future__ import annotations

from typing import Any, Union

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

    # --- Transforms (build wired) ------------------------------------------
    def alias(self, name: str) -> "Column":
        return Column(Alias(self._expr, name))

    def cast(self, data_type: Any) -> "Column":
        """ANSI-strict cast: raises on malformed input at evaluate time (Spark 4 default)."""
        return Column(Cast(self._expr, data_type, strict=True))

    def try_cast(self, data_type: Union[Any, str]) -> "Column":
        """Lenient cast (Spark 4 ``try_cast``): returns null instead of raising."""
        return Column(Cast(self._expr, data_type, strict=False))

    def isNull(self) -> "Column":
        return Column(UnaryExpression("isnull", self._expr))

    def isNotNull(self) -> "Column":
        return Column(UnaryExpression("isnotnull", self._expr))

    # --- Transforms (slots — build the matching node when implementing) -----
    def isin(self, *values: Any) -> "Column":
        not_implemented_yet("Column.isin")

    def rlike(self, pattern: str) -> "Column":
        not_implemented_yet("Column.rlike")

    def contains(self, substring: str) -> "Column":
        not_implemented_yet("Column.contains")

    def getItem(self, key: Union[str, int]) -> "Column":
        not_implemented_yet("Column.getItem")

    def __getitem__(self, key: Union[str, int]) -> "Column":
        not_implemented_yet("Column.__getitem__")

    def asc(self) -> "Column":
        not_implemented_yet("Column.asc")

    def desc(self) -> "Column":
        not_implemented_yet("Column.desc")

    def asc_nulls_last(self) -> "Column":
        not_implemented_yet("Column.asc_nulls_last")

    def desc_nulls_last(self) -> "Column":
        not_implemented_yet("Column.desc_nulls_last")

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


def _column_ref(value: Any) -> Expression:
    """Coerce a ``str | Column`` 'column argument' into an :class:`Expression`.

    Unlike :func:`_to_expression`, a bare ``str`` here means a *column name* (the convention
    for ``functions`` like ``abs(col_name)``), so it becomes an :class:`AttributeReference`.
    """
    if isinstance(value, str):
        return AttributeReference(value)
    return _to_expression(value)
