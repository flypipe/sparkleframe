"""``DataFrame`` for the pure-Python engine.

The constructor stores the rows and the (PySpark) schema verbatim — that is the data the
build → analyze → evaluate flow consumes. A transformation resolves its column expressions through
``analyzer.analyze`` (type resolution, against the schema) then ``evaluator.evaluate`` (execution),
producing a new frame. Operations beyond ``select`` are not implemented yet and raise.

Type handling: per the engine's first-pass decision the schema is a PySpark ``StructType``
consumed directly (see ``docs/design/python-engine-ast.md`` and the plan).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

from sparkleframe.base.dataframe import DataFrame as BaseDataFrame
from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast.analyzer import analyze
from sparkleframe.python.ast.evaluator import evaluate
from sparkleframe.python.column import Column
from sparkleframe.python.dataframe_helpers import output_name, rows_as_dicts
from sparkleframe.python.functions import col as _col

try:  # pragma: no cover - exercised only with the real pyspark installed
    from pyspark.sql.types import StructField, StructType
except Exception:  # pragma: no cover - mock pyspark (under activate) has no real types
    pass


class DataFrame(BaseDataFrame):
    def __init__(self, data: Any, schema: Optional[Any] = None) -> None:
        super().__init__(df=None)
        # Stored verbatim for the analyze/evaluate phases to consume.
        self._rows: list = list(data) if data is not None else []
        self._schema = schema

    def to_records(self) -> list:
        """Engine → ``list[dict]`` for the parity oracle: zip schema names with each row tuple."""
        return rows_as_dicts(self._rows, self._schema)

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
        columns = [c if isinstance(c, Column) else _col(c) if isinstance(c, str) else None for c in cols]
        if any(c is None for c in columns):
            not_implemented_yet("DataFrame.select for non-str/Column arguments")

        input_rows = rows_as_dicts(self._rows, self._schema)
        out_fields = []
        out_values = []  # one value list per output column
        for column in columns:
            resolved = analyze(column._expr, self._schema)
            out_fields.append(StructField(output_name(resolved), resolved.data_type))
            out_values.append(evaluate(resolved, input_rows))

        out_rows = [tuple(values[i] for values in out_values) for i in range(len(self._rows))]
        return DataFrame(out_rows, StructType(out_fields))

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
