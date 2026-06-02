"""Window / WindowSpec for pythondf — mirrors a subset of PySpark's window API.

This module is intentionally pure-stdlib: it owns only the *specification* of a
window (partition keys, order keys, frame bounds). The actual windowed
computation lives in :meth:`sparkleframe.pythondf.column.Column.over`, which
reads the spec and produces a per-row result column.
"""
from __future__ import annotations

import sys
from typing import Callable, List, Optional, Tuple, Union

from sparkleframe.pythondf.column import Column

# Tuple shape used internally for order keys.
# ``(eval_fn, descending, nulls_last, name)`` — ``name`` is best-effort for debug.
OrderSpec = Tuple[Callable[[dict], list], bool, bool, Optional[str]]


def _name_for_partition(col: Union[str, Column]) -> str:
    if isinstance(col, str):
        return col
    if isinstance(col, Column):
        if col.name:
            return col.name
        raise ValueError("partitionBy(Column) requires a named column reference")
    raise TypeError(
        f"partitionBy expects str or Column, got {type(col).__name__}"
    )


def _order_spec_for(col: Union[str, Column]) -> OrderSpec:
    if isinstance(col, str):
        col_name = col
        return ((lambda d, n=col_name: list(d[n])), False, False, col_name)
    if isinstance(col, Column):
        if col._sort_key is not None:
            return (
                col._sort_key,
                col._sort_descending,
                col._sort_nulls_last,
                col.name,
            )
        return (col._eval, False, False, col.name)
    raise TypeError(f"orderBy expects str or Column, got {type(col).__name__}")


class WindowSpec:
    """Specification for a window: partition keys, order keys, and frame bounds."""

    def __init__(self) -> None:
        self._partition_by: List[str] = []
        self._order_specs: List[OrderSpec] = []
        self._frame_start: Optional[int] = None
        self._frame_end: Optional[int] = None
        # Distinguishes "user gave no orderBy" (default whole-partition frame for
        # aggs) from "user gave orderBy but no rowsBetween" (default
        # unboundedPreceding-to-currentRow running frame).
        self._has_explicit_frame: bool = False

    def _copy(self) -> "WindowSpec":
        new = WindowSpec()
        new._partition_by = list(self._partition_by)
        new._order_specs = list(self._order_specs)
        new._frame_start = self._frame_start
        new._frame_end = self._frame_end
        new._has_explicit_frame = self._has_explicit_frame
        return new

    def partitionBy(self, *cols: Union[str, Column]) -> "WindowSpec":  # noqa: N802 — PySpark API
        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = tuple(cols[0])
        new = self._copy()
        new._partition_by = [_name_for_partition(c) for c in cols]
        return new

    def orderBy(self, *cols: Union[str, Column]) -> "WindowSpec":  # noqa: N802 — PySpark API
        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = tuple(cols[0])
        new = self._copy()
        new._order_specs = [_order_spec_for(c) for c in cols]
        return new

    def rowsBetween(self, start: int, end: int) -> "WindowSpec":  # noqa: N802 — PySpark API
        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError("rowsBetween requires integer start and end values")
        if not isinstance(end, int) or isinstance(end, bool):
            raise TypeError("rowsBetween requires integer start and end values")
        new = self._copy()
        new._frame_start = start
        new._frame_end = end
        new._has_explicit_frame = True
        return new

    def rangeBetween(self, start: int, end: int) -> "WindowSpec":  # noqa: N802 — PySpark API
        # We model rangeBetween the same way as rowsBetween for the offsets we
        # support (Spark behaviour diverges for non-zero offsets with multi-col
        # orderBy or non-numeric order keys; that's out of scope here).
        return self.rowsBetween(start, end)

    # --- read-only accessors (used by tests + Column.over) ---

    @property
    def partition_cols(self) -> List[str]:
        return list(self._partition_by)

    @property
    def order_cols(self) -> List[Union[str, Column]]:
        # The polarsdf test asserts ``spec.order_cols == [..]``. We can't return
        # the original Column objects (they may not be Columns at all once
        # converted to OrderSpec tuples), so return the captured names where
        # available (matching the polarsdf shape for string-keyed orderBy).
        return [name if name is not None else "<expr>" for _, _, _, name in self._order_specs]

    @property
    def frame_start(self) -> Optional[int]:
        return self._frame_start

    @property
    def frame_end(self) -> Optional[int]:
        return self._frame_end


class Window:
    """Factory for :class:`WindowSpec` instances."""

    _JAVA_MIN_LONG = -(1 << 63)
    _JAVA_MAX_LONG = (1 << 63) - 1

    unboundedPreceding: int = _JAVA_MIN_LONG
    unboundedFollowing: int = _JAVA_MAX_LONG
    currentRow: int = 0

    @staticmethod
    def partitionBy(*cols: Union[str, Column]) -> WindowSpec:  # noqa: N802 — PySpark API
        return WindowSpec().partitionBy(*cols)

    @staticmethod
    def orderBy(*cols: Union[str, Column]) -> WindowSpec:  # noqa: N802 — PySpark API
        return WindowSpec().orderBy(*cols)

    @staticmethod
    def rowsBetween(start: int, end: int) -> WindowSpec:  # noqa: N802 — PySpark API
        return WindowSpec().rowsBetween(start, end)

    @staticmethod
    def rangeBetween(start: int, end: int) -> WindowSpec:  # noqa: N802 — PySpark API
        return WindowSpec().rangeBetween(start, end)
