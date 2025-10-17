from __future__ import annotations

from typing import Union, Any, Iterable, Tuple
from uuid import uuid4
from sparkleframe.polarsdf import types as sft
import pandas as pd
import polars as pl
import pyarrow as pa

from sparkleframe.base.dataframe import DataFrame as BaseDataFrame
from sparkleframe.polarsdf.column import Column
from sparkleframe.polarsdf.group import GroupedData

from sparkleframe.polarsdf.types import (
    DataType,
    StringType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    DateType,
    TimestampType,
    ByteType,
    ShortType,
    DecimalType,
    BinaryType,
    StructType,
    StructField,
)

from typing import List, Optional

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
            # Return a single column by name
            return Column(self.df[item])
        elif isinstance(item, int):
            # Return a column by index
            return Column(self.df[self.df.columns[item]])
        elif isinstance(item, Column):
            # Return a filtered DataFrame
            return DataFrame(self.df.filter(item.to_native()))
        elif isinstance(item, (list, tuple)):
            # Return a DataFrame with selected columns
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
            filtered_df = self.df.filter(pl.col(condition))
        elif isinstance(condition, Column):
            filtered_df = self.df.filter(condition.to_native())
        else:
            raise TypeError("filter() expects a string column name or a Column expression")

        return DataFrame(filtered_df)

    where = filter  # Alias for .filter()

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
        pl_expressions = []

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

        selected_df = self.df.select(*pl_expressions)
        return DataFrame(selected_df)

    def withColumn(self, name: str, col: Any) -> DataFrame:
        """
        Mimics PySpark's withColumn method using Polars.

        Args:
            name: Name of the new or updated column.
            col: A Column object representing the expression for the new column.

        Returns:
            A new DataFrame with the added or updated column.
        """
        col = Column(col) if not isinstance(col, Column) else col
        expr = col.to_native().alias(name)
        updated_df = self.df.with_columns(expr)
        return DataFrame(updated_df)

    def withColumnRenamed(self, existing: str, new: str) -> DataFrame:
        """
        Mimics PySpark's withColumnRenamed method using Polars.

        Args:
            existing: The current column name.
            new: The new name to apply.

        Returns:
            A new DataFrame with the renamed column.

        Raises:
            ValueError: If the existing column name is not in the DataFrame.
        """
        if existing not in self.df.columns:
            raise ValueError(f"Column '{existing}' does not exist in the DataFrame.")

        renamed_df = self.df.rename({existing: new})
        return DataFrame(renamed_df)

    def toPandas(self) -> pd.DataFrame:
        """
        Convert the underlying Polars DataFrame to a Pandas DataFrame,
        ensuring nested arrays/maps/structs are JSON-friendly.
        Removes keys inside dicts where the value is None,
        but keeps the column and row structure intact.
        """
        import math
        import numpy as np
        import pandas as pd

        df = self.df.to_arrow().to_pandas()

        def is_null(x) -> bool:
            try:
                return pd.isna(x) and not isinstance(x, (str, bytes))
            except Exception:
                return False

        def convert_number(x):
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, int):
                return x
            if isinstance(x, (np.floating, float)):
                if math.isnan(x):
                    return None
                return int(x) if float(x).is_integer() else float(x)
            return x

        # --- helpers to collapse Polars map layout(s) ---
        def _is_kv(d) -> bool:
            return isinstance(d, dict) and "key" in d and "value" in d

        def _kv_list_to_dict(kv_list: list):
            # [{"key":k,"value":v}, ...] -> {k: v}, skipping null values
            out = {}
            for item in kv_list:
                k = convert(item["key"])
                v = convert(item.get("value"))
                if v is not None:
                    out[k] = v
            return out

        def convert(val):
            if is_null(val):
                return None

            # NumPy arrays -> list
            if isinstance(val, np.ndarray):
                return [convert(v) for v in val.tolist()]

            # collapse map and array<map> encodings
            if isinstance(val, list):
                # Case 1: a single map encoded as a list of {"key","value"} dicts
                if val and all(_is_kv(item) for item in val):
                    return _kv_list_to_dict(val)

                # Case 2: array<map<...>> encoded as list of kv-lists
                if val and all(isinstance(item, list) and (not item or all(_is_kv(x) for x in item)) for item in val):
                    return [_kv_list_to_dict(item) for item in val]

                # General list -> recurse
                return [convert(v) for v in val]

            if isinstance(val, tuple):
                return [convert(v) for v in val]

            if isinstance(val, dict):
                # Drop only keys whose converted value is None
                out = {}
                for k, v in val.items():
                    cv = convert(v)
                    if cv is not None:
                        out[k] = cv
                return out

            return convert_number(val)

        return df.applymap(convert)

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
        has_col = False
        if isinstance(on, str):
            on = [on]
        elif isinstance(on, Column):
            has_col = True
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
                    has_col = True
                    break
            on = [n.to_native() if isinstance(n, Column) else n for n in on]

        # Mapping of PySpark join types to Polars join types
        PYSPARK_TO_POLARS_JOIN_MAP = {
            "inner": "inner",
            "cross": "cross",
            "outer": "full",
            "full": "full",
            "fullouter": "full",
            "full_outer": "full",
            "left": "left",
            "leftouter": "left",
            "left_outer": "left",
            "right": "right",
            "rightouter": "right",
            "right_outer": "right",
            "semi": "semi",
            "leftsemi": "semi",
            "left_semi": "semi",
            "anti": "anti",
            "leftanti": "anti",
            "left_anti": "anti",
        }

        how = how.lower()
        if how not in PYSPARK_TO_POLARS_JOIN_MAP:
            raise ValueError(f"Unsupported join type: '{how}'")

        polars_join_type = PYSPARK_TO_POLARS_JOIN_MAP[how]
        suffix = "_" + str(uuid4()).replace("-", "")
        result = self.df.join(other.df, on=on, how=polars_join_type, suffix=suffix)

        if how == "outer":
            """
            Polars does not automatically coalesce join keys (e.g., id) in a full outer join because it retains both left and right keys explicitly, especially when:
                * There are mismatches in the keys (e.g., id exists only on one side).
                * It needs to distinguish between matching and non-matching keys.

            Why this happens?
            Polars must preserve all information during a full (outer) join:
                * If the key is missing on one side, it will still be included in the output, but with nulls on the missing side.
                * Rather than overwrite or merge the column into one, it creates:
                    - id from the left table
                    - id_right (or similar suffix) from the right table

            This ensures no loss of data or ambiguity, which is particularly important for:
                * Asymmetric joins (like one-to-many).
                * Duplicated key values or nulls.
            """

            for col in result.columns:
                if col.endswith(suffix):

                    # TODO: for some reason pyspark results from outer differs when `on_keys` are Column or str, wheter
                    # the col is dropped or a coalesce happens
                    if has_col:
                        result = result.drop(col)
                    else:
                        result = result.with_columns(
                            pl.coalesce(col.replace(suffix, ""), col).alias(col.replace(suffix, ""))
                        ).drop(col)

        for col in result.columns:
            if col.endswith(suffix):
                result = result.rename({col: col.replace(suffix, "") + "_right"})

        return DataFrame(result)

    @property
    def dtypes(self) -> List[tuple[str, str]]:
        """
        Mimics pyspark.pandas.DataFrame.dtypes.

        Returns a list of tuples with (column name, string representation of data type).

        Returns:
            List[Tuple[str, str]]: List of (column name, data type) pairs.
        """
        POLARS_TO_PYSPARK_DTYPE_MAP = {
            pl.Int8: "tinyint",
            pl.Int16: "smallint",
            pl.Int32: "int",
            pl.Int64: "bigint",
            pl.UInt8: "tinyint",
            pl.UInt16: "smallint",
            pl.UInt32: "int",
            pl.UInt64: "bigint",
            pl.Float32: "float",
            pl.Float64: "double",
            pl.Boolean: "boolean",
            pl.Utf8: "string",
            pl.Date: "date",
            pl.Datetime: "timestamp",
            pl.Time: "time",
            pl.Duration: "interval",
            pl.Object: "binary",
            pl.List: "array",
            pl.Struct: "struct",
            pl.Decimal: "decimal",
            pl.Binary: "binary",
        }

        def map_dtype(dtype: pl.DataType) -> str:
            if isinstance(dtype, pl.Decimal):
                return f"decimal({dtype.precision},{dtype.scale})"

            if isinstance(dtype, pl.Struct):
                # Recursively describe fields
                fields_str = ",".join(f"{field.name}:{map_dtype(field.dtype)}" for field in dtype.fields)
                return f"struct<{fields_str}>"

            for polars_type, spark_type in POLARS_TO_PYSPARK_DTYPE_MAP.items():
                if isinstance(dtype, polars_type):
                    return spark_type

            return str(dtype)

        return [(col, map_dtype(dtype)) for col, dtype in self.df.schema.items()]

    @property
    def schema(self) -> StructType:
        """
        Mimics pyspark.sql.DataFrame.schema by returning the schema as a StructType.
        """

        def _declared_type_for(col_name: str) -> Optional[DataType]:
            if isinstance(self._schema, StructType):
                for f in self._schema:
                    if f.name == col_name:
                        return f.dataType
            return None

        def _to_spark_datatype(dtype: pl.DataType, col_name: Optional[str] = None) -> DataType:
            # --- NEW: detect native map layout (List(Struct["key","value"])) ---
            if _MapTypeUtils.is_map_dtype(dtype):
                key_dt_pl = dtype.inner.fields[0].dtype
                val_dt_pl = dtype.inner.fields[1].dtype
                key_dt = _to_spark_datatype(key_dt_pl)  # typically StringType()
                val_dt = _to_spark_datatype(val_dt_pl)  # may itself be a Map/Array/Struct/etc.

                # preserve declared valueContainsNull if we have it
                value_contains_null = True
                if col_name is not None:
                    decl = _declared_type_for(col_name)
                    if isinstance(decl, sft.MapType):
                        value_contains_null = decl.valueContainsNull

                return sft.MapType(key_dt, val_dt, valueContainsNull=value_contains_null)

            # Decimals
            if isinstance(dtype, pl.Decimal):
                return DecimalType(dtype.precision, dtype.scale)

            # Structs (recursive)
            if isinstance(dtype, pl.Struct):
                nested_fields = [StructField(f.name, _to_spark_datatype(f.dtype)) for f in dtype.fields]
                return StructType(nested_fields)

            # Arrays
            if isinstance(dtype, pl.List):
                elem_dtype = _to_spark_datatype(dtype.inner)
                contains_null = True
                if col_name is not None:
                    decl = _declared_type_for(col_name)
                    if isinstance(decl, sft.ArrayType):
                        contains_null = decl.containsNull
                return sft.ArrayType(elem_dtype, containsNull=contains_null)

            # Scalars
            POLARS_TO_SPARK = {
                pl.Utf8: StringType(),
                pl.Int32: IntegerType(),
                pl.UInt32: IntegerType(),
                pl.Int64: LongType(),
                pl.UInt64: LongType(),
                pl.Float32: FloatType(),
                pl.Float64: DoubleType(),
                pl.Boolean: BooleanType(),
                pl.Date: DateType(),
                pl.Datetime: TimestampType(),
                pl.Int8: ByteType(),
                pl.UInt8: ByteType(),
                pl.Int16: ShortType(),
                pl.UInt16: ShortType(),
                pl.Binary: BinaryType(),
            }
            for pl_type, spark_type in POLARS_TO_SPARK.items():
                if isinstance(dtype, pl_type):
                    return spark_type

            raise TypeError(f"Unsupported dtype '{dtype}'")

        def polars_dtype_to_spark_structfield(name: str, dtype: pl.DataType) -> StructField:
            decl = _declared_type_for(name)
            # Preserve declared ArrayType(MapType(...)) exactly as provided
            if isinstance(decl, sft.ArrayType) and isinstance(decl.elementType, sft.MapType):
                return StructField(name, decl)
            # Preserve declared MapType exactly as provided
            if isinstance(decl, sft.MapType):
                return StructField(name, decl)
            return StructField(name, _to_spark_datatype(dtype, col_name=name))

        return StructType([polars_dtype_to_spark_structfield(name, dtype) for name, dtype in self.df.schema.items()])

    def sort(self, *cols: Union[str, Column, List[Union[str, Column]]]) -> DataFrame:
        """
        Mimics PySpark's DataFrame.orderBy using Polars.

        Args:
            *cols: Columns or Column expressions to sort by.
                Can be:
                  - strings: "col1", "col2"
                  - Column objects with sort metadata (e.g., from asc(), desc(), asc_nulls_first())
                  - a single list of such elements

        Returns:
            DataFrame: A new DataFrame sorted by the specified columns.
        """
        if len(cols) == 1 and isinstance(cols[0], list):
            cols = cols[0]

        sort_cols = []
        sort_descending = []
        sort_nulls_last = []
        for i, col in enumerate(cols):
            if isinstance(col, int):
                sort_cols.append(self.df.columns[i])
                sort_descending.append(True if col < 0 else False)
                sort_nulls_last.append(True)

            if isinstance(col, str):
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
