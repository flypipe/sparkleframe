from __future__ import annotations

from typing import Union, Any, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import polars as pl
import pyarrow as pa
from pandas import DataFrame as PandasDataFrame

from sparkleframe.base.dataframe import DataFrame as BaseDataFrame
from sparkleframe.polarsdf.column import Column
from sparkleframe.polarsdf.group import GroupedData

from sparkleframe.polarsdf.types import (
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


class DataFrame(BaseDataFrame):

    def __init__(self, df: Union[pl.DataFrame, pd.DataFrame, pa.Table], schema: Optional[StructType] = None):
        if isinstance(df, pl.DataFrame):
            self.df = df
        elif isinstance(df, pd.DataFrame):
            self.df = pl.DataFrame(df)
        elif isinstance(df, pa.Table):
            self.df = pl.from_arrow(df)
        else:
            raise TypeError("DataFrame constructor accepts polars.DataFrame, pandas.DataFrame, or pyarrow.Table")
        self._schema = schema
        super().__init__(self.df)

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

        Args:
            *cols: Column names or Column wrapper objects.

        Returns:
            A new DataFrame with selected columns.
        """
        cols = list(cols)
        cols = cols[0] if isinstance(cols[0], list) else cols
        pl_expressions = [col.to_native() if isinstance(col, Column) else col for col in cols]
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

    def toPandas(self) -> PandasDataFrame:
        """Convert the underlying Polars DataFrame to a Pandas DataFrame."""
        return self.df.to_arrow().to_pandas()

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

        Returns:
            StructType: Spark-like schema derived from the Polars DataFrame schema.
        """

        def polars_dtype_to_spark_dtype(name: str, dtype: pl.DataType) -> StructField:

            # Handle decimals
            if isinstance(dtype, pl.Decimal):
                return StructField(name, DecimalType(dtype.precision, dtype.scale))

            # Handle structs (recursively)
            if isinstance(dtype, pl.Struct):
                nested_fields = [polars_dtype_to_spark_dtype(field.name, field.dtype) for field in dtype.fields]
                return StructField(name, StructType(nested_fields))

            # Basic type mappings
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
                    return StructField(name, spark_type)

            raise TypeError(f"Unsupported dtype '{dtype}' for column '{name}'")

        return StructType([polars_dtype_to_spark_dtype(name, dtype) for name, dtype in self.df.schema.items()])

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

    orderBy = sort
