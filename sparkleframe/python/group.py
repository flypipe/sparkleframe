"""``GroupedData`` for the pure-Python engine — produced by ``DataFrame.groupBy``.

The constructor stores the grouping columns; the aggregations are slots that raise until the
analyze/evaluate phases land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.column import Column

if TYPE_CHECKING:  # avoid a runtime import cycle with dataframe.py
    from sparkleframe.python.dataframe import DataFrame


class GroupedData:
    def __init__(self, df: "DataFrame", group_cols: list[Union[str, Column]]) -> None:
        self.df = df
        self.group_cols = group_cols

    def agg(self, *exprs: Union[str, Column]) -> "DataFrame":
        not_implemented_yet("GroupedData.agg")

    def count(self) -> "DataFrame":
        not_implemented_yet("GroupedData.count")

    def sum(self) -> "DataFrame":
        not_implemented_yet("GroupedData.sum")

    def mean(self) -> "DataFrame":
        not_implemented_yet("GroupedData.mean")

    def max(self) -> "DataFrame":
        not_implemented_yet("GroupedData.max")

    def min(self) -> "DataFrame":
        not_implemented_yet("GroupedData.min")
