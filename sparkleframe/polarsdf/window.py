from __future__ import annotations
from typing import List, Union

from sparkleframe.polarsdf.column import Column


class WindowSpec:
    """
    Mimics PySpark's WindowSpec object for defining windowing configurations.

    Supports partitioning, ordering, and frame specification for window functions.
    """

    def __init__(self):
        self._partition_by: List[str] = []
        self._order_by: List[Union[str, Column]] = []
        self._frame_start: int | None = None
        self._frame_end: int | None = None

    def partitionBy(self, *cols: Union[str, Column]) -> WindowSpec:
        """
        Specifies the columns to partition the data by.

        Args:
            *cols (str or Column): Column names or Column objects to partition by.

        Returns:
            WindowSpec: The updated WindowSpec instance.
        """
        if len(cols) == 1:
            if isinstance(cols[0], list):
                cols = cols[0]
            else:
                cols = [cols[0]]
        else:
            cols = list(cols)

        self._partition_by = [col if isinstance(col, str) else col.to_native().meta.root_names()[0] for col in cols]
        return self

    def orderBy(self, *cols: Union[str, Column]) -> WindowSpec:
        """
        Specifies the columns to sort the data within partitions.

        Args:
            *cols (str or Column): Column names or Column objects to sort by.

        Returns:
            WindowSpec: The updated WindowSpec instance.
        """
        if len(cols) == 1:
            if isinstance(cols[0], list):
                cols = cols[0]
            else:
                cols = [cols[0]]
        else:
            cols = list(cols)
        self._order_by = cols
        return self

    def rangeBetween(self, start: int, end: int) -> WindowSpec:
        """
        Specifies the range frame boundaries for the window.

        This sets the start and end of the frame relative to the current row's value
        (based on ordered column), inclusive.

        Args:
            start (int): Start of the frame (inclusive).
            end (int): End of the frame (inclusive).

        Returns:
            WindowSpec: The updated WindowSpec instance.
        """
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("rangeBetween requires integer start and end values")
        self._frame_start = start
        self._frame_end = end
        return self

    @property
    def partition_cols(self) -> List[str]:
        """Returns the list of partition columns."""
        return self._partition_by

    @property
    def order_cols(self) -> List[Union[str, Column]]:
        """Returns the list of ordering columns."""
        return self._order_by

    @property
    def frame_start(self) -> int | None:
        """Returns the start of the window frame if defined."""
        return self._frame_start

    @property
    def frame_end(self) -> int | None:
        """Returns the end of the window frame if defined."""
        return self._frame_end


class Window:
    """
    Mimics PySpark's Window object to build WindowSpec expressions.
    """

    _JAVA_MIN_LONG = -(1 << 63)  # -9223372036854775808
    _JAVA_MAX_LONG = (1 << 63) - 1  # 9223372036854775807

    unboundedPreceding: int = _JAVA_MIN_LONG
    unboundedFollowing: int = _JAVA_MAX_LONG
    currentRow: int = 0

    @staticmethod
    def partitionBy(*cols: Union[str, Column]) -> WindowSpec:
        """
        Returns a WindowSpec partitioned by the given columns.

        Args:
            *cols (str or Column): Columns to partition by.

        Returns:
            WindowSpec: A WindowSpec with partitioning applied.
        """
        return WindowSpec().partitionBy(*cols)

    @staticmethod
    def orderBy(*cols: Union[str, Column]) -> WindowSpec:
        """
        Returns a WindowSpec ordered by the given columns.

        Args:
            *cols (str or Column): Columns to sort by.

        Returns:
            WindowSpec: A WindowSpec with ordering applied.
        """
        return WindowSpec().orderBy(*cols)
