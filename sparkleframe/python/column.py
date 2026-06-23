"""``Column`` for the pure-Python engine — a declared surface (walking skeleton).

A :class:`Column` will be the public face of the build phase: its operators and transforms will
construct unresolved expression-AST nodes that a later analyze → evaluate pass resolves and runs
(see ``docs/design/python-engine-ast.md``). None of that exists yet — this class is a husk whose
every operator and transform raises. The build phase, and the AST it produces, land with the
engine itself.
"""

from __future__ import annotations

from typing import Any

from sparkleframe.python._errors import not_implemented_yet


class Column:
    def __init__(self) -> None:
        """No state yet; the wrapped expression arrives with the build phase."""

    # --- Arithmetic ---------------------------------------------------------
    def __add__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__add__")

    def __sub__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__sub__")

    def __mul__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__mul__")

    def __truediv__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__truediv__")

    def __pow__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__pow__")

    def __radd__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__radd__")

    def __rsub__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__rsub__")

    def __rmul__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__rmul__")

    def __rtruediv__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__rtruediv__")

    def __rpow__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__rpow__")

    def __neg__(self) -> "Column":
        not_implemented_yet("Column.__neg__")

    def __pos__(self) -> "Column":
        not_implemented_yet("Column.__pos__")

    # --- Comparison ---------------------------------------------------------
    def __eq__(self, other: Any) -> "Column":  # type: ignore[override]
        not_implemented_yet("Column.__eq__")

    def __ne__(self, other: Any) -> "Column":  # type: ignore[override]
        not_implemented_yet("Column.__ne__")

    def __lt__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__lt__")

    def __le__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__le__")

    def __gt__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__gt__")

    def __ge__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__ge__")

    # Column overrides __eq__, so it is unhashable by default — mirror PySpark.
    __hash__ = None  # type: ignore[assignment]

    # --- Logical ------------------------------------------------------------
    def __and__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__and__")

    def __rand__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__rand__")

    def __or__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__or__")

    def __ror__(self, other: Any) -> "Column":
        not_implemented_yet("Column.__ror__")

    def __invert__(self) -> "Column":
        not_implemented_yet("Column.__invert__")

    # --- Transforms ---------------------------------------------------------
    def alias(self, name: str) -> "Column":
        not_implemented_yet("Column.alias")

    def cast(self, data_type: Any) -> "Column":
        not_implemented_yet("Column.cast")

    def try_cast(self, data_type: Any) -> "Column":
        not_implemented_yet("Column.try_cast")

    def isNull(self) -> "Column":
        not_implemented_yet("Column.isNull")

    def isNotNull(self) -> "Column":
        not_implemented_yet("Column.isNotNull")

    def isin(self, *values: Any) -> "Column":
        not_implemented_yet("Column.isin")

    def rlike(self, pattern: str) -> "Column":
        not_implemented_yet("Column.rlike")

    def contains(self, substring: str) -> "Column":
        not_implemented_yet("Column.contains")

    def getItem(self, key: Any) -> "Column":
        not_implemented_yet("Column.getItem")

    def __getitem__(self, key: Any) -> "Column":
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
        return "Column()"
