"""``DataFrame`` for the pure-Python engine — a walking skeleton.

The constructor stores the rows and the (PySpark) schema verbatim — that is the data the
build → analyze → evaluate flow will consume. Every transformation and action is a raising slot
for now; they land incrementally as the analyze (type resolution) and evaluate (execution) phases
are implemented (see ``docs/design/python-engine-ast.md``).

Type handling: per the engine's first-pass decision the schema is a PySpark ``StructType``
consumed directly (see ``docs/design/python-engine-ast.md`` and the plan).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

from sparkleframe.base.dataframe import DataFrame as BaseDataFrame
from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.column import Column


class DataFrame(BaseDataFrame):
    def __init__(self, data: Any, schema: Optional[Any] = None) -> None:
        super().__init__(df=None)
        # Stored verbatim for the analyze/evaluate phases to consume.
        self._rows: list = list(data) if data is not None else []
        self._schema = schema

    def to_records(self) -> list:
        not_implemented_yet("DataFrame.to_records")

    def columns(self) -> List[str]:
        not_implemented_yet("DataFrame.columns")

    def schema(self) -> Any:
        not_implemented_yet("DataFrame.schema")

    def dtypes(self) -> List[Tuple[str, str]]:
        not_implemented_yet("DataFrame.dtypes")

    def __getitem__(self, item: Union[int, str, Column, List, Tuple]) -> Union[Column, "DataFrame"]:
        not_implemented_yet("DataFrame.__getitem__")

    def alias(self, name: str) -> "DataFrame":
        not_implemented_yet("DataFrame.alias")

    def select(self, *cols: Union[str, Column, List[str], List[Column]]) -> "DataFrame":
        not_implemented_yet("DataFrame.select")

    def filter(self, condition: Union[str, Column]) -> "DataFrame":
        not_implemented_yet("DataFrame.filter")

    def where(self, condition: Union[str, Column]) -> "DataFrame":
        not_implemented_yet("DataFrame.where")

    def withColumn(self, name: str, col: Any) -> "DataFrame":
        not_implemented_yet("DataFrame.withColumn")

    def withColumns(self, colsMap: dict[str, Column]) -> "DataFrame":
        not_implemented_yet("DataFrame.withColumns")

    def withColumnRenamed(self, existing: str, new: str) -> "DataFrame":
        not_implemented_yet("DataFrame.withColumnRenamed")

    def drop(self, *cols: Union[str, Column]) -> "DataFrame":
        not_implemented_yet("DataFrame.drop")

    def distinct(self) -> "DataFrame":
        not_implemented_yet("DataFrame.distinct")

    def dropDuplicates(self, subset: Optional[List[str]] = None) -> "DataFrame":
        not_implemented_yet("DataFrame.dropDuplicates")

    def union(self, other: "DataFrame") -> "DataFrame":
        not_implemented_yet("DataFrame.union")

    def unionByName(self, other: "DataFrame", allowMissingColumns: bool = False) -> "DataFrame":
        not_implemented_yet("DataFrame.unionByName")

    def join(self, other: "DataFrame", on: Any = None, how: str = "inner") -> "DataFrame":
        not_implemented_yet("DataFrame.join")

    def groupBy(self, *cols: Union[str, Column]) -> Any:
        not_implemented_yet("DataFrame.groupBy")

    def groupby(self, *cols: Union[str, Column]) -> Any:
        not_implemented_yet("DataFrame.groupby")

    def sort(self, *cols: Union[str, Column, int, List[Union[str, Column, int]]]) -> "DataFrame":
        not_implemented_yet("DataFrame.sort")

    def orderBy(self, *cols: Union[str, Column, int, List[Union[str, Column, int]]]) -> "DataFrame":
        not_implemented_yet("DataFrame.orderBy")

    def fillna(self, value: Union[Any, dict], subset: Union[str, List[str], None] = None) -> "DataFrame":
        not_implemented_yet("DataFrame.fillna")

    def count(self) -> int:
        not_implemented_yet("DataFrame.count")

    def show(self, n: int = 20, truncate: bool = True, vertical: bool = False) -> None:
        not_implemented_yet("DataFrame.show")

    def toPandas(self) -> Any:
        not_implemented_yet("DataFrame.toPandas")

    def to_arrow(self) -> Any:
        not_implemented_yet("DataFrame.to_arrow")
