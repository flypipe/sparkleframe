"""``Window`` / ``WindowSpec`` for the pure-Python engine.

A spec only records the partition and order expressions; ``Column.over(spec)`` turns a ranking
function into a :class:`~sparkleframe.python.ast.expressions.WindowExpression`, which the
DataFrame layer evaluates over the partitioned, ordered rows. Frame bounds are recorded for API
compatibility; ranking functions ignore them, as Spark does.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from sparkleframe.python.ast.expressions import Expression, SortOrder
from sparkleframe.python.column import Column, _column_ref


class WindowSpec:
    def __init__(
        self,
        partition_cols: Optional[List[Any]] = None,
        order_cols: Optional[List[Any]] = None,
        frame: Optional[Tuple[str, int, int]] = None,
    ) -> None:
        self._partition_cols: List[Any] = list(partition_cols or [])
        self._order_cols: List[Any] = list(order_cols or [])
        self._frame = frame

    def partitionBy(self, *cols: Any) -> "WindowSpec":
        return WindowSpec(_flatten(cols), self._order_cols, self._frame)

    def orderBy(self, *cols: Any) -> "WindowSpec":
        return WindowSpec(self._partition_cols, _flatten(cols), self._frame)

    def rowsBetween(self, start: int, end: int) -> "WindowSpec":
        return WindowSpec(self._partition_cols, self._order_cols, ("rows", start, end))

    def rangeBetween(self, start: int, end: int) -> "WindowSpec":
        return WindowSpec(self._partition_cols, self._order_cols, ("range", start, end))

    def _partition_exprs(self) -> List[Expression]:
        return [_column_ref(c) for c in self._partition_cols]

    def _order_exprs(self) -> List[SortOrder]:
        orders = []
        for col in self._order_cols:
            expr = _column_ref(col)
            orders.append(expr if isinstance(expr, SortOrder) else SortOrder(expr, ascending=True, nulls_first=True))
        return orders


def _flatten(cols: tuple) -> List[Any]:
    if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
        return list(cols[0])
    return list(cols)


class Window:
    unboundedPreceding = -(2**63)
    unboundedFollowing = 2**63 - 1
    currentRow = 0

    @staticmethod
    def partitionBy(*cols: Any) -> WindowSpec:
        return WindowSpec().partitionBy(*cols)

    @staticmethod
    def orderBy(*cols: Any) -> WindowSpec:
        return WindowSpec().orderBy(*cols)

    @staticmethod
    def rowsBetween(start: int, end: int) -> WindowSpec:
        return WindowSpec().rowsBetween(start, end)


__all__ = ["Column", "Window", "WindowSpec"]
