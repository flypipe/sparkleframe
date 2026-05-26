from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import pandas as pd
import polars as pl
import pyarrow as pa

from sparkleframe.base.dataframe import DataFrame as BaseDataFrame
from sparkleframe.polarsdf import types as sft
from sparkleframe.polarsdf.column import Column
from sparkleframe.polarsdf.column_helpers import _polars_schema_for
from sparkleframe.polarsdf.dataframe_helpers import (
    PYSPARK_TO_POLARS_JOIN_MAP,
    coalesce_outer_join_keys,
    convert_pandas_value,
    map_polars_dtype_to_spark_name,
    perform_expression_join,
    polars_dtype_to_spark_structfield,
    resolve_equi_join_keys,
)
from sparkleframe.polarsdf.group import GroupedData
from sparkleframe.polarsdf.types import DataType, StructField, StructType
from sparkleframe.polarsdf.types_utils import _MapTypeUtils


class DataFrame(BaseDataFrame):

    def __init__(
        self,
        data: Union[Iterable[Any], pd.DataFrame, pl.DataFrame, pa.Table, list],
        schema: Optional[Union[DataType, StructType, str]] = None,
    ):
        """
        Schema-aware constructor:
        - If schema is StructType and data is list/tuple/records -> build with exact names/dtypes.
        - If schema is a single DataType -> wrap as StructType([StructField("value", ...)]).
        - MapType fields are automatically materialized as Polars Structs with fields equal
          to the ordered union of keys observed in the provided rows.
        - StructType fields remain Structs (dot access works natively).
        """
        self._schema = schema

        # --- Build the Polars DataFrame according to schema and data types ---
        if isinstance(schema, StructType) and isinstance(data, (list, tuple)):
            self.df = _MapTypeUtils.build_df_from_struct_rows(data, schema)

        elif isinstance(schema, StructType) and isinstance(data, pd.DataFrame):
            rows = data.to_dict(orient="records")
            self.df = _MapTypeUtils.build_df_from_struct_rows(rows, schema)

        elif isinstance(schema, DataType) and isinstance(data, (list, tuple)):
            # Wrap single logical type as a single-field StructType named "value" (Spark-like)
            wrapped = StructType([StructField("value", schema)])
            self.df = _MapTypeUtils.build_df_from_struct_rows(data, wrapped)

        elif isinstance(schema, (list, tuple)) and all(isinstance(col_name, str) for col_name in schema):
            # Spark-style createDataFrame(data=[(...), (...)], schema=["c1", "c2", ...]) provides row-oriented tuples.
            # Force row orientation so Polars does not infer tuple values as column vectors.
            if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
                self.df = pl.DataFrame(data, schema=list(schema), orient="row")
            else:
                self.df = pl.DataFrame(data, schema=list(schema))

        elif isinstance(data, pd.DataFrame):
            # No (or non-StructType) schema: let Polars infer
            self.df = pl.DataFrame(data)

        elif isinstance(data, pl.DataFrame):
            self.df = data

        elif isinstance(data, pa.Table):
            self.df = pl.from_arrow(data)

        elif isinstance(data, list):
            # No schema provided: let Polars infer
            self.df = pl.DataFrame(data)

        else:
            raise TypeError(
                "createDataFrame only supports polars.DataFrame, pandas.DataFrame, pyarrow.Table, or row iterables"
            )

        # --- Automatically materialize MapType columns into Structs for dot access ---
        try:
            if isinstance(self._schema, StructType):
                for f in self._schema:
                    # Only apply to MapType fields
                    if isinstance(f.dataType, sft.MapType):
                        dtype = self.df.schema.get(f.name)
                        if dtype is not None and _MapTypeUtils.is_map_dtype(dtype):
                            # Convert the map column to a Struct (overwrite same name)
                            self.df = _MapTypeUtils.map_to_struct(self.df, f.name)
                    # StructType fields are already Structs — no action needed
        except Exception as e:
            # Never break construction on auto-materialization errors
            import warnings

            warnings.warn(f"MapType materialization skipped due to error: {e}", RuntimeWarning)

        # >>> NEW: enforce primitive casts per provided schema
        if isinstance(self._schema, StructType):
            self.df = _MapTypeUtils.apply_schema_casts(self.df, self._schema)

        # --- Call parent constructor (BaseDataFrame) ---
        super().__init__(self.df)

    # -------------------- Helpers for schema-aware construction --------------------

    # inside your DataFrame class

    # -------------------- Selection / projection --------------------

    def __getitem__(self, item: Union[int, str, Column, List, Tuple]) -> Union[Column, "DataFrame"]:
        if isinstance(item, str):
            # Return a single column by name (``pl.col``, not a materialized Series, so ``Column`` ops work).
            return Column(pl.col(item))
        elif isinstance(item, int):
            # Return a column by index
            return Column(pl.col(self.df.columns[item]))
        elif isinstance(item, Column):
            # Return a filtered DataFrame
            with _polars_schema_for(self.df.schema):
                return DataFrame(self.df.filter(item.to_native()))
        elif isinstance(item, (list, tuple)):
            # Return a DataFrame with selected columns
            with _polars_schema_for(self.df.schema):
                cols = [col.to_native() if isinstance(col, Column) else col for col in item]
                return DataFrame(self.df.select(cols))
        else:
            raise TypeError(f"Unexpected type: {type(item)}")

    @property
    def columns(self) -> List[str]:
        """
        Returns the list of column names in the DataFrame.

        Mimics PySpark's DataFrame.columns property.

        Returns:
            List[str]: List of column names.
        """
        return self.df.columns

    def __getattr__(self, item: str):
        if item in self.df.columns:
            return Column(pl.col(item))
        raise AttributeError(f"'DataFrame' object has no attribute '{item}'")

    def alias(self, name: str) -> DataFrame:
        """
        Mimics PySpark's DataFrame.alias(name).

        While Polars doesn't use DataFrame aliases directly, this method
        stores the alias internally for potential use in more complex query building.

        Args:
            name (str): The alias to assign to this DataFrame.

        Returns:
            DataFrame: The same DataFrame instance with alias stored.
        """
        df = DataFrame(self.df)
        df._alias = name
        return df

    def filter(self, condition: Union[str, Column]) -> DataFrame:
        """
        Mimics PySpark's DataFrame.filter() method using Polars.

        Args:
            condition (Union[str, Column]): A filter condition either as a string or a Column expression.

        Returns:
            DataFrame: A new DataFrame containing only the rows that match the filter condition.
        """
        if isinstance(condition, str):
            try:
                # Support Spark-style SQL predicates, e.g. "rn = 1", "col is null and other is null".
                filtered_df = self.df.filter(pl.sql_expr(condition))
            except Exception:
                filtered_df = self.df.filter(pl.col(condition))
        elif isinstance(condition, Column):
            with _polars_schema_for(self.df.schema):
                filtered_df = self.df.filter(condition.to_native())
        else:
            raise TypeError("filter() expects a string column name or a Column expression")

        return DataFrame(filtered_df)

    where = filter  # Alias for .filter()

    def union(self, other: "DataFrame") -> "DataFrame":
        """
        Mimics PySpark's DataFrame.union (UNION ALL by position).
        """
        if not isinstance(other, DataFrame):
            raise TypeError("union() expects a DataFrame")

        left_cols = self.columns
        right_cols = other.columns
        if len(left_cols) != len(right_cols):
            raise ValueError(
                f"union() requires same number of columns; left={len(left_cols)}, right={len(right_cols)}"
            )

        right_df = other.df.select([pl.col(col_name).alias(left_cols[idx]) for idx, col_name in enumerate(right_cols)])
        return DataFrame(pl.concat([self.df, right_df], how="vertical_relaxed"))

    unionAll = union

    def unionByName(self, other: "DataFrame", allowMissingColumns: bool = False) -> "DataFrame":
        """
        Mimics PySpark ``DataFrame.unionByName`` (UNION ALL by column name).

        When ``allowMissingColumns`` is false, both frames must have the same set of
        column names; the right side is reordered to match the left frame's column order.

        When true, columns present in only one frame are filled with null in the other,
        matching Spark's relaxed union-by-name.

        Args:
            other: Another sparkleframe :class:`DataFrame`.
            allowMissingColumns: If true, allow disjoint schemas with null padding.

        Returns:
            DataFrame: Row-wise concatenation aligned by name.

        Raises:
            TypeError: If ``other`` is not a :class:`DataFrame`.
            ValueError: If schemas differ and ``allowMissingColumns`` is false.
        """
        if not isinstance(other, DataFrame):
            raise TypeError("unionByName() expects a DataFrame")

        left_cols = self.columns
        left_set = set(left_cols)
        right_set = set(other.columns)

        if not allowMissingColumns:
            if left_set != right_set:
                missing_in_right = sorted(left_set - right_set)
                extra_in_right = sorted(right_set - left_set)
                msg_parts: list[str] = []
                if missing_in_right:
                    msg_parts.append(f"columns not in the right frame: {missing_in_right}")
                if extra_in_right:
                    msg_parts.append(f"columns only in the right frame: {extra_in_right}")
                raise ValueError(
                    "unionByName() requires the same column names when allowMissingColumns=False ("
                    + "; ".join(msg_parts)
                    + "). Use allowMissingColumns=True to union with null padding."
                )
            right_aligned = other.df.select([pl.col(name) for name in left_cols])
            return DataFrame(pl.concat([self.df, right_aligned], how="vertical"))

        return DataFrame(pl.concat([self.df, other.df], how="diagonal_relaxed"))

    def distinct(self) -> "DataFrame":
        """
        Mimics PySpark's DataFrame.distinct.
        """
        return DataFrame(self.df.unique())

    def dropDuplicates(self, subset: Optional[List[str]] = None) -> "DataFrame":
        """
        Mimics PySpark's DataFrame.dropDuplicates.
        """
        if subset is None:
            return self.distinct()
        return DataFrame(self.df.unique(subset=subset))

    def select(self, *cols: Union[str, Column, List[str], List[Column]]) -> "DataFrame":
        """
        Mimics PySpark's select method using Polars.
        Select columns or expressions.
        Supports:
          - "colname"
          - "col.field" and deeper paths like "col.a.b.c" (struct/map-derived structs)
          - Column or list of Columns
        For dotted paths, the resulting column is aliased to the last path segment.

        Args:
            *cols: Column names or Column wrapper objects.

        Returns:
            A new DataFrame with selected columns.
        """
        cols = list(cols)
        cols = cols[0] if cols and isinstance(cols[0], list) else cols
        pl_expressions: List[Any] = []

        with _polars_schema_for(self.df.schema):
            for c in cols:
                if isinstance(c, Column):
                    pl_expressions.append(c.to_native())
                    continue

                if isinstance(c, str):
                    if "." in c:
                        parts = c.split(".")
                        base, tail = parts[0], parts[1:]
                        expr = pl.col(base)
                        for seg in tail:
                            expr = expr.struct.field(seg)
                        # Alias to the last segment ("id2" for "col.id.id2")
                        expr = expr.alias(tail[-1])
                        pl_expressions.append(expr)
                    else:
                        pl_expressions.append(pl.col(c))
                    continue

                # fallback: assume it's already a polars expr or valid selector
                pl_expressions.append(c)

        return DataFrame(self.df.select(*pl_expressions))

    def withColumn(self, name: str, col: Any) -> DataFrame:
        """
        Mimics PySpark's withColumn method using Polars.

        Args:
            name: Name of the new or updated column.
            col: A Column object representing the expression for the new column.

        Returns:
            A new DataFrame with the added or updated column.
        """
        if hasattr(col, "branches") and hasattr(col, "otherwise"):
            col = col.otherwise(None)
        col = Column(col) if not isinstance(col, Column) else col

        if getattr(col, "_is_explode", False):
            source_name = getattr(col, "_explode_source_name", None)
            if source_name and source_name in self.df.columns:
                exploded_df = self.df.explode(source_name)
                if name != source_name:
                    exploded_df = exploded_df.rename({source_name: name})
                return DataFrame(exploded_df)

        with _polars_schema_for(self.df.schema):
            expr = col._to_native_getitem_only().alias(name)
        updated_df = self.df.with_columns(expr)
        return DataFrame(updated_df)

    def withColumns(self, colsMap: dict[str, "Column"]) -> DataFrame:
        """
        Mimics PySpark's withColumns method using Polars.

        Args:
            colsMap: A dict mapping column names to Column expressions.

        Returns:
            A new DataFrame with all columns added or replaced.
        """
        result = self
        for name, col_expr in colsMap.items():
            result = result.withColumn(name, col_expr)
        return result

    def withColumnRenamed(self, existing: str, new: str) -> DataFrame:
        """
        Mimics PySpark's withColumnRenamed method using Polars.

        Args:
            existing: The current column name.
            new: The new name to apply.

        Returns:
            A new DataFrame with the renamed column.

        Notes:
            Matches PySpark behavior: if the source column does not exist, returns
            the original DataFrame unchanged.
        """
        if existing not in self.df.columns:
            return DataFrame(self.df)

        renamed_df = self.df.rename({existing: new})
        return DataFrame(renamed_df)

    def drop(self, *cols: Union[str, Column]) -> "DataFrame":
        """
        Mimics PySpark's DataFrame.drop.

        Removes the given columns. Columns that are not in the schema are ignored
        (same as PySpark). With no arguments, returns a new DataFrame wrapper over
        the same underlying Polars frame (same pattern as ``select`` / ``sort``).

        Args:
            *cols: Column names as strings or Column references (e.g. ``col("x")``).
                Use ``df.drop(*["a", "b"])`` to drop from a list (same as PySpark).

        Returns:
            DataFrame: A new DataFrame without the dropped columns.
        """
        if not cols:
            return DataFrame(self.df)

        to_drop: List[str] = []
        for c in cols:
            if isinstance(c, str):
                to_drop.append(c)
            elif isinstance(c, Column):
                roots = c.to_native().meta.root_names()
                if not roots:
                    raise TypeError("drop() Column expression must reference a named column")
                to_drop.append(roots[0])
            else:
                raise TypeError(f"drop() expected str or Column, got {type(c).__name__}")

        existing = [name for name in to_drop if name in self.df.columns]
        if not existing:
            return DataFrame(self.df)
        return DataFrame(self.df.drop(*existing))

    def toPandas(self) -> pd.DataFrame:
        """
        Convert the underlying Polars DataFrame to a Pandas DataFrame,
        ensuring nested arrays/maps/structs are JSON-friendly.
        Removes keys inside dicts where the value is None,
        but keeps the column and row structure intact.
        """
        df = self.df.to_arrow().to_pandas()
        return df.map(convert_pandas_value)

    def to_arrow(self) -> pa.Table:
        """
        Convert the Polars DataFrame to an Apache Arrow Table.

        Returns:
            pyarrow.Table: Arrow representation of the DataFrame.
        """
        return self.df.to_arrow()

    def show(self, n: int = 20, truncate: bool = True, vertical: bool = False):
        """
        Mimics PySpark's DataFrame.show() using Polars' native rendering.

        Args:
            n (int, optional, default 20): Number of rows to show.x
            truncate (bool): Ignored — Polars handles column truncation.
            vertical (bool or int, optional, default False): If True, displays rows in vertical layout.
        """
        if vertical:
            for i, row in enumerate(self.df.head(n).iter_rows(named=True)):
                print(f"-ROW {i}")
                for key, val in row.items():
                    print(f"{key}: {val}")
        else:
            pl.Config.set_tbl_cols(len(self.df.columns))
            print(self.df.head(n))
            pl.Config.restore_defaults()

    def fillna(self, value: Union[Any, dict], subset: Union[str, List[str], None] = None) -> DataFrame:
        """
        Mimics PySpark's DataFrame.fillna() using Polars.

        Args:
            value (Any or dict): The value to replace nulls with. If a dict, keys are column names.
            subset (str or list[str], optional): Subset of columns to apply fillna to.
                Ignored if value is a dict.

        Returns:
            DataFrame: A new DataFrame with nulls filled.
        """
        value_type = type(value)

        def matches_dtype(dtype: pl.DataType) -> bool:
            """Helper to determine if Polars dtype matches Python type."""
            return (
                (
                    value_type is int
                    and dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
                )
                or (value_type is float and dtype in (pl.Float32, pl.Float64))
                or (value_type is str and dtype == pl.Utf8)
                or (value_type is bool and dtype == pl.Boolean)
            )

        if isinstance(value, dict):
            # Fillna with different values per column
            exprs = [pl.col(col).fill_null(val).alias(col) for col, val in value.items() if col in self.df.columns]
            filled_df = self.df.with_columns(exprs)
        else:
            # Fillna with the same value across specified columns (or all columns)
            if subset is None:
                subset = self.df.columns
            elif isinstance(subset, str):
                subset = [subset]

            # Build expressions

            exprs = [
                pl.col(col).fill_null(value).alias(col)
                for col in subset
                if col in self.df.columns and matches_dtype(self.df.schema[col])
            ]

            filled_df = self.df.with_columns(exprs)

        return DataFrame(filled_df)

    def groupBy(self, *cols: Union[str, Column]) -> GroupedData:
        """
        Mimics PySpark's DataFrame.groupBy() using Polars.

        Args:
            *cols: One or more column names or Column objects.

        Returns:
            GroupedData: An object that can perform aggregations.
        """
        return GroupedData(self, list(cols))

    def groupby(self, *cols: Union[str, Column]) -> GroupedData:
        """
        Mimics PySpark's DataFrame.groupBy() using Polars.

        Args:
            *cols: One or more column names or Column objects.

        Returns:
            GroupedData: An object that can perform aggregations.
        """
        return self.groupBy(*cols)

    def join(
        self, other: DataFrame, on: Union[str, List[str], Column, List[Column], None] = None, how: str = "inner"
    ) -> DataFrame:
        """
        Mimics PySpark's DataFrame.join() using Polars.

        Args:
            other (DataFrame): The DataFrame to join with.
            on (str or List[str] or Column or List[Column], None): Column(s) to join on. If None, uses common column names.
            how (str): Type of join to perform. Supports all PySpark variants.

        Returns:
            DataFrame: A new DataFrame resulting from the join.
        """
        # True when the user wrapped keys in Column(...) (affects full-outer key coalescing).
        on_column_wrappers = False
        if isinstance(on, str):
            on = [on]
        elif isinstance(on, Column):
            on_column_wrappers = True
            on = [on.to_native()]
        elif isinstance(on, list):

            type_ = None
            for n in on:
                type_ = type_ or type(n)
                if type_ is not type(n):
                    raise TypeError(
                        "On columns must have the same type. str or List[str] or Column or List[Column], None)"
                    )

                if isinstance(n, Column):
                    on_column_wrappers = True
                    break
            on = [n.to_native() if isinstance(n, Column) else n for n in on]

        how = how.lower()
        if how not in PYSPARK_TO_POLARS_JOIN_MAP:
            raise ValueError(f"Unsupported join type: '{how}'")

        polars_join_type = PYSPARK_TO_POLARS_JOIN_MAP[how]
        suffix = "_" + str(uuid4()).replace("-", "")
        use_expr_join = on is not None and len(on) == 1 and isinstance(on[0], pl.Expr) and not on[0].meta.is_column()
        if use_expr_join:
            if how not in {"inner", "left", "leftouter", "left_outer"}:
                raise ValueError("Expression joins currently support only inner/left joins")
            result = perform_expression_join(self.df, other.df, on[0], how, suffix)
        else:
            equi_on: Union[None, str, list[str]] = on
            if on is not None:
                equi_on = resolve_equi_join_keys(on)
            result = self.df.join(other.df, on=equi_on, how=polars_join_type, suffix=suffix)

        if how == "outer":
            result = coalesce_outer_join_keys(result, suffix, on_column_wrappers)

        for col_name in result.columns:
            if col_name.endswith(suffix):
                result = result.rename({col_name: col_name.replace(suffix, "") + "_right"})

        return DataFrame(result)

    @property
    def dtypes(self) -> List[tuple[str, str]]:
        """
        Mimics pyspark.pandas.DataFrame.dtypes.

        Returns a list of tuples with (column name, string representation of data type).

        Returns:
            List[Tuple[str, str]]: List of (column name, data type) pairs.
        """
        return [(col, map_polars_dtype_to_spark_name(dtype)) for col, dtype in self.df.schema.items()]

    @property
    def schema(self) -> StructType:
        """
        Mimics pyspark.sql.DataFrame.schema by returning the schema as a StructType.
        """
        return StructType(
            [
                polars_dtype_to_spark_structfield(name, dtype, declared_schema=self._schema)
                for name, dtype in self.df.schema.items()
            ]
        )

    def sort(self, *cols: Union[str, Column, int, List[Union[str, Column, int]]]) -> DataFrame:
        """
        Mimics PySpark's DataFrame.orderBy using Polars.

        Args:
            *cols: Columns or Column expressions to sort by.
                Can be:
                  - strings: "col1", "col2"
                  - ints: 1-based column ordinals (negative means descending on that column)
                  - Column objects with sort metadata (e.g., from asc(), desc(), asc_nulls_first())
                  - a single list of such elements

        Returns:
            DataFrame: A new DataFrame sorted by the specified columns.
        """
        if len(cols) == 1 and isinstance(cols[0], list):
            cols = cols[0]

        column_names = self.df.columns
        sort_cols = []
        sort_descending = []
        sort_nulls_last = []
        for col in cols:
            if isinstance(col, int) and not isinstance(col, bool):
                if col == 0:
                    raise ValueError("[ZERO_INDEX] Index must be non-zero.")
                if col > 0:
                    sort_cols.append(column_names[col - 1])
                else:
                    sort_cols.append(column_names[-col - 1])
                sort_descending.append(col < 0)
                sort_nulls_last.append(True)
            elif isinstance(col, str):
                sort_cols.append(col)
                sort_descending.append(False)
                sort_nulls_last.append(True)
            elif isinstance(col, Column):
                sort_cols.append(col._sort_col)
                sort_descending.append(col._sort_descending)
                sort_nulls_last.append(col._sort_nulls_last)
            else:
                raise TypeError(f"orderBy received unsupported type: {type(col)}")

        sorted_df = self.df.sort(by=sort_cols, descending=sort_descending, nulls_last=sort_nulls_last)
        return DataFrame(sorted_df)

    def count(self) -> int:
        """
        Mimics PySpark's DataFrame.count().

        Returns:
            int: Number of rows in the DataFrame.
        """
        # Polars exposes the row count as a cheap .height property.
        return int(self.df.height)

    orderBy = sort
