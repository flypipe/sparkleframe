"""``GroupedData`` for the pure-Python engine — produced by ``DataFrame.groupBy``.

The grouping keys and aggregate expressions are handed to the DataFrame layer, which groups the
rows and evaluates each aggregate per group (see ``DataFrame._aggregate``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Union

import sparkleframe.python.functions as F
from sparkleframe.python.column import Column

if TYPE_CHECKING:  # avoid a runtime import cycle with dataframe.py
    from sparkleframe.python.dataframe import DataFrame

_DICT_AGGREGATES = {"count": F.count, "sum": F.sum, "avg": F.avg, "mean": F.avg, "max": F.max, "min": F.min}


class GroupedData:
    def __init__(self, df: "DataFrame", group_cols: list[Union[str, Column]]) -> None:
        self.df = df
        self.group_cols = group_cols

    def agg(self, *exprs: Union[Column, Dict[str, str]]) -> "DataFrame":
        if len(exprs) == 1 and isinstance(exprs[0], dict):
            exprs = tuple(
                _DICT_AGGREGATES[func](column).alias(f"{func}({column})") for column, func in exprs[0].items()
            )
        return self.df._aggregate(self.group_cols, list(exprs))

    def count(self) -> "DataFrame":
        return self.agg(F.count("*").alias("count"))

    def _numeric(self, func: Any, name: str, cols: tuple) -> "DataFrame":
        from sparkleframe.python.ast.coercion import is_numeric
        from sparkleframe.python.types import DecimalType

        columns = list(cols) or [
            f.name for f in self.df.schema.fields if is_numeric(f.dataType) or isinstance(f.dataType, DecimalType)
        ]
        return self.agg(*(func(c).alias(f"{name}({c})") for c in columns))

    def sum(self, *cols: str) -> "DataFrame":
        return self._numeric(F.sum, "sum", cols)

    def avg(self, *cols: str) -> "DataFrame":
        return self._numeric(F.avg, "avg", cols)

    mean = avg

    def max(self, *cols: str) -> "DataFrame":
        return self._numeric(F.max, "max", cols)

    def min(self, *cols: str) -> "DataFrame":
        return self._numeric(F.min, "min", cols)
