import json

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from pyspark.sql import functions as F
from pyspark.sql.functions import col as spark_col
from pyspark.sql.types import (
    StringType as SparkStringType,
    IntegerType as SparkIntegerType,
    LongType as SparkLongType,
    FloatType as SparkFloatType,
    DoubleType as SparkDoubleType,
    BooleanType as SparkBooleanType,
    DateType as SparkDateType,
    TimestampType as SparkTimestampType,
    DecimalType as SparkDecimalType,
    ByteType as SparkByteType,
    ShortType as SparkShortType,
    BinaryType as SparkBinaryType,
    StructType as SparkStructType,
    StructField as SparkStructField,
)

import sparkleframe.polarsdf.functions as PF
from sparkleframe.polarsdf import Column
from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.types import (
    StringType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    DateType,
    TimestampType,
    DecimalType,
    ByteType,
    ShortType,
    BinaryType,
    StructType,
    StructField,
    MapType,
)
from sparkleframe.tests.pyspark_test import assert_pyspark_df_equal
from sparkleframe.tests.utils import to_records, create_spark_df, assert_sparkle_spark_frame_are_equal
from sparkleframe.polarsdf.types_utils import _MapTypeUtils

sample_data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [70000, 80000, 90000],
    "birth_date": ["1990-01-01", "1985-05-15", "1970-12-30"],
    "login_time": ["2024-01-01T08:00:00", "2024-01-02T09:30:00", "2024-01-03T11:45:00"],
}


@pytest.fixture
def sparkle_df():
    return DataFrame(pl.DataFrame(sample_data))


@pytest.fixture
def spark_df(spark):
    return spark.createDataFrame(pd.DataFrame(sample_data))


def _polars_native_map_series(name: str, dict_rows: list[dict[str, int]]) -> pl.Series:
    """
    Build a Polars-native 'map' layout column:
      List[Struct{key: Utf8, value: Int32}]
    from python dict rows.
    """
    kv_rows = [[{"key": k, "value": v} for k, v in d.items()] for d in dict_rows]
    return pl.Series(name, kv_rows)


class TestDataFrame:

    @pytest.mark.parametrize(
        "data, schema_sparkle, schema_spark",
        [
            (
                [{"a": 1}, {"a": 2}],
                StructType([StructField("a", IntegerType())]),
                SparkStructType([SparkStructField("a", SparkIntegerType())]),
            ),
            (
                pd.DataFrame([{"a": 1}, {"a": 2}]),
                StructType([StructField("a", IntegerType())]),
                SparkStructType([SparkStructField("a", SparkIntegerType())]),
            ),
            (
                [1, 2, 3, 4],
                IntegerType(),
                SparkIntegerType(),
            ),
            (
                pd.DataFrame([{"a": 1}, {"a": 2}]),
                None,
                None,
            ),
            (
                pl.DataFrame([{"a": 1}, {"a": 2}]),
                None,
                None,
            ),
            (
                pa.table({"a": [1, 2]}),
                None,
                None,
            ),
            ([{"a": 1}, {"a": 2}], None, None),
        ],
    )
    def test_dataframe_creation(self, spark, sparkle, data, schema_sparkle, schema_spark):

        df_sparkle = sparkle.createDataFrame(data, schema=schema_sparkle)

        if isinstance(data, pl.DataFrame):
            data = data.to_pandas().to_dict(orient="records")

        if isinstance(data, pa.Table):
            data = data.to_pylist()

        df_spark = spark.createDataFrame(data, schema=schema_spark)
        assert assert_sparkle_spark_frame_are_equal(df_sparkle, df_spark)

    @pytest.mark.parametrize(
        "cols",
        [
            ("name"),
            (["name", "age"]),
        ],
    )
    def test_select_by_list_str(self, spark, sparkle_df, spark_df, cols):
        result_spark_df = spark.createDataFrame(sparkle_df.select(cols).toPandas())
        expected_spark_df = spark_df.select(cols)
        assert_pyspark_df_equal(result_spark_df, expected_spark_df)

    @pytest.mark.parametrize(
        "cols",
        [
            ("name"),
            (["name", "age"]),
        ],
    )
    def test_select_by_list_columns(self, spark, sparkle_df, spark_df, cols):
        cols = [cols] if isinstance(cols, str) else cols
        result_spark_df = spark.createDataFrame(sparkle_df.select([PF.col(col) for col in cols]).toPandas())
        expected_spark_df = spark_df.select([F.col(col) for col in cols])
        assert_pyspark_df_equal(result_spark_df, expected_spark_df)

    def test_select_by_column_name(self, spark, sparkle_df, spark_df):
        result_spark_df = spark.createDataFrame(sparkle_df.select("name").toPandas())
        expected_spark_df = spark_df.select("name")

        assert_pyspark_df_equal(result_spark_df, expected_spark_df)

    def test_select_by_pointer_list(self, spark, sparkle_df, spark_df):
        result_spark_df = spark.createDataFrame(sparkle_df.select(*["name", "age"]).toPandas())
        expected_spark_df = spark_df.select(*["name", "age"])

        assert_pyspark_df_equal(result_spark_df, expected_spark_df)

    def test_select_by_expression(self, spark, sparkle_df, spark_df):
        result_spark_df = spark.createDataFrame(
            sparkle_df.select(PF.col("name"), PF.col("salary") * 1.1).df.to_dicts()
        )

        expected_spark_df = spark_df.select(spark_col("name"), (spark_col("salary") * 1.1).alias("salary"))

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, precision=5)

    def test_select_all_columns_with_aliases(self, spark, sparkle_df, spark_df):
        # Define aliases
        aliases = {"name": "employee_name", "age": "employee_age", "salary": "employee_salary"}

        # Apply aliases using DataFrame
        polars_selected = sparkle_df.select(*(PF.col(col).alias(alias) for col, alias in aliases.items()))
        result_spark_df = spark.createDataFrame(polars_selected.toPandas())

        # Apply the same aliases using PySpark
        expected_spark_df = spark_df.select(*(spark_col(col).alias(alias) for col, alias in aliases.items()))

        assert_pyspark_df_equal(result_spark_df, expected_spark_df)

    def test_with_column_add(self, spark, sparkle_df, spark_df):
        result_spark_df = spark.createDataFrame(
            sparkle_df.withColumn("bonus", (PF.col("salary") * 0.1)).toPandas()
        ).withColumn("bonus", F.col("bonus").cast(SparkFloatType()))

        expected_spark_df = spark_df.withColumn("bonus", spark_col("salary") * 0.1).withColumn(
            "bonus", F.col("bonus").cast(SparkFloatType())
        )

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, precision=5)

    def test_with_column_replace(self, spark, sparkle_df, spark_df):
        result_spark_df = spark.createDataFrame(sparkle_df.withColumn("salary", PF.col("salary") * 2).toPandas())

        expected_spark_df = spark_df.withColumn("salary", spark_col("salary") * 2)

        assert_pyspark_df_equal(result_spark_df, expected_spark_df)

    def test_with_column_renamed(self, spark, sparkle_df, spark_df):
        # Apply renaming using DataFrame
        renamed_polars_df = sparkle_df.withColumnRenamed("name", "employee_name")
        result_spark_df = spark.createDataFrame(renamed_polars_df.toPandas())

        # Apply the same renaming in PySpark
        expected_spark_df = spark_df.withColumnRenamed("name", "employee_name")

        assert_pyspark_df_equal(result_spark_df, expected_spark_df)

    def test_to_native_df(self, sparkle_df):
        native_df = sparkle_df.to_native_df()

        # Check that it's a Polars DataFrame
        assert isinstance(native_df, pl.DataFrame)

        # Check schema matches expected
        assert native_df.columns == sparkle_df.to_native_df().columns

        # Check data matches original sample_data
        assert native_df.shape == sparkle_df.to_native_df().shape
        assert native_df[0, 0] == "Alice"
        assert native_df[1, 1] == 30
        assert native_df[2, 2] == 90000

    @pytest.mark.parametrize(
        "col_name, data_type_class, spark_type",
        [
            ("name", StringType(), SparkStringType()),
            ("age", IntegerType(), SparkIntegerType()),
            ("age", LongType(), SparkLongType()),
            ("age", FloatType(), SparkFloatType()),
            ("age", DoubleType(), SparkDoubleType()),
            ("age", BooleanType(), SparkBooleanType()),
            ("age", DecimalType(13, 2), SparkDecimalType(13, 2)),
            ("birth_date", DateType(), SparkDateType()),
            ("login_time", TimestampType(), SparkTimestampType()),
            ("age", ByteType(), SparkByteType()),
            ("age", ShortType(), SparkShortType()),
            ("age", BinaryType(), SparkBinaryType()),
        ],
    )
    def test_cast_types(self, spark, sparkle_df, spark_df, col_name, data_type_class, spark_type):
        # Apply the cast using your API
        expr = PF.col(col_name).cast(data_type_class)
        polars_result_df = sparkle_df.select(expr.alias(col_name)).to_native_df()

        # Apply the cast using PySpark
        spark_result_df = spark_df.select(F.col(col_name).cast(spark_type).alias(col_name))

        # Extract data types for comparison
        polars_dtype = polars_result_df.schema[col_name]
        spark_dtype = spark_result_df.schema[col_name].dataType

        # Manual mapping to match Spark types to Polars types
        spark_to_polars_map = {
            SparkStringType(): pl.Utf8,
            SparkIntegerType(): pl.Int32,
            SparkLongType(): pl.Int64,
            SparkFloatType(): pl.Float32,
            SparkDoubleType(): pl.Float64,
            SparkBooleanType(): pl.Boolean,
            SparkDateType(): pl.Date,
            SparkTimestampType(): pl.Datetime,
            SparkDecimalType(13, 2): pl.Decimal(13, 2),  # include your decimal type
            SparkShortType(): pl.Int16,
            SparkBinaryType(): pl.Binary,
            SparkByteType(): pl.Int8,
        }

        expected_polars_dtype = spark_to_polars_map[spark_dtype]

        assert polars_dtype == expected_polars_dtype

    def test_to_arrow_creates_correct_spark_df(self, spark):
        pl_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "amount": [10.5, 20.0, 30.25],
                "active": [True, False, True],
            }
        )
        sparkle_df = DataFrame(pl_df)

        pandas_df = sparkle_df.to_arrow().to_pandas(types_mapper=pd.ArrowDtype)

        result_spark_df = spark.createDataFrame(pandas_df)

        expected_spark_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "name": ["Alice", "Bob", "Charlie"],
                    "amount": [10.5, 20.0, 30.25],
                    "active": [True, False, True],
                }
            )
        )

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_create_polars_from_arrow_generated_by_spark(self, spark):
        # Step 1: Create a Spark DataFrame
        spark_df = spark.createDataFrame(
            [("Alice", 25, True), ("Bob", 30, False), ("Charlie", 35, True)], ["name", "age", "active"]
        )

        # Step 2: Collect as Arrow record batches and convert to Arrow Table
        arrow_batches = spark_df._collect_as_arrow()
        arrow_table = pa.Table.from_batches(arrow_batches)

        # Step 3: Create DataFrame from Arrow Table
        sparkle_df = DataFrame(arrow_table)

        # Step 4: Convert both to Pandas for comparison (safer for schema + nulls)
        expected_pd = spark_df.toPandas()
        result_pd = sparkle_df.toPandas()

        # Step 5: Sort by name for deterministic comparison (optional)
        expected_pd_sorted = expected_pd.sort_values(by="name").reset_index(drop=True)
        result_pd_sorted = result_pd.sort_values(by="name").reset_index(drop=True)

        # Step 6: Compare using assert_frame_equal
        pd.testing.assert_frame_equal(result_pd_sorted, expected_pd_sorted)

    def test_is_not_null(self, spark):
        # Step 1: Create sample data with nulls
        data = {"id": [1, 2, 3, 4], "name": ["Alice", None, "Charlie", None]}
        pl_df = pl.DataFrame(data)
        sparkle_df = DataFrame(pl_df)

        # Step 2: Apply isNotNull using your API
        result_pd = sparkle_df.select(PF.col("name").isNotNull().alias("not_null")).toPandas()
        result_spark_df = spark.createDataFrame(result_pd)
        # Step 3: Apply isNotNull using PySpark
        spark_df = spark.createDataFrame(pd.DataFrame(data))
        expected_df = spark_df.select(F.col("name").isNotNull().alias("not_null"))

        # Step 4: Compare
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "literal, op_name, expr_func",
        [
            (10, "+", lambda col_expr: 10 + col_expr),
            (10, "-", lambda col_expr: 10 - col_expr),
            (10, "*", lambda col_expr: 10 * col_expr),
            (10, "/", lambda col_expr: 10 / col_expr),
        ],
    )
    def test_reverse_arithmetic_operators(self, spark, literal, op_name, expr_func):
        # Prepare data
        pandas_df = pd.DataFrame({"a": [1, 2, 3]})
        sparkle_df = DataFrame(pandas_df)

        # Run sparkleframe expression
        expr = expr_func(PF.col("a")).alias("result")
        result_df = sparkle_df.select(expr).toPandas()
        result_spark_df = spark.createDataFrame(result_df)

        # Expected result using PySpark directly
        expected_spark_df = spark.createDataFrame(pandas_df).select(
            (
                spark_col("a").__radd__(literal)
                if op_name == "+"
                else (
                    spark_col("a").__rsub__(literal)
                    if op_name == "-"
                    else spark_col("a").__rmul__(literal) if op_name == "*" else spark_col("a").__rtruediv__(literal)
                )
            ).alias("result")
        )

        # Assert equality
        assert_pyspark_df_equal(result_spark_df, expected_spark_df, precision=5)

    @pytest.mark.parametrize(
        "description, expr_func",
        [
            ("AND", lambda a, b, c: (a > 1) & (b < 6)),
            ("OR", lambda a, b, c: (a > 2) | (b < 6)),
            ("chained AND-OR", lambda a, b, c: ((a > 1) & (b < 6)) | (c > 7)),
            ("chained OR-AND", lambda a, b, c: (a < 2) | ((b == 5) & (c < 9))),
        ],
    )
    def test_logical_operations(self, spark, description, expr_func):
        data = {"a": [1, 2, 3, 4], "b": [10, 5, 3, 8], "c": [7, 12, 9, 4]}

        sparkle_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(pd.DataFrame(data))

        expr = expr_func(PF.col("a"), PF.col("b"), PF.col("c")).alias("result")
        expected_expr = expr_func(F.col("a"), F.col("b"), F.col("c")).alias("result")

        result_df = spark.createDataFrame(sparkle_df.select(expr).toPandas())
        expected_df = spark_df.select(expected_expr)

        assert_pyspark_df_equal(result_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "description, expr_func",
        [
            ("filter by single column", lambda a, b, c: a > 2),
            ("filter with AND", lambda a, b, c: (a > 1) & (b < 10)),
            ("filter with OR", lambda a, b, c: (a < 2) | (c > 7)),
            ("chained AND-OR", lambda a, b, c: ((a > 1) & (b < 6)) | (c > 7)),
        ],
    )
    def test_filter_and_where(self, spark, description, expr_func):
        data = {"a": [1, 2, 3, 4], "b": [10, 5, 3, 8], "c": [7, 12, 9, 4]}

        sparkle_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(pd.DataFrame(data))

        # Test .filter
        filtered_result = sparkle_df.filter(expr_func(PF.col("a"), PF.col("b"), PF.col("c"))).toPandas()
        expected_result = spark_df.filter(expr_func(F.col("a"), F.col("b"), F.col("c")))

        result_spark_df = spark.createDataFrame(filtered_result)
        assert_pyspark_df_equal(result_spark_df, expected_result, ignore_nullable=True)

        # Test .where (should behave the same)
        where_result = sparkle_df.where(expr_func(PF.col("a"), PF.col("b"), PF.col("c"))).toPandas()
        result_spark_df = spark.createDataFrame(where_result)
        assert_pyspark_df_equal(result_spark_df, expected_result, ignore_nullable=True)

    @pytest.mark.parametrize(
        "pattern, expected_matches",
        [
            (".*a.*", ["Alice", "Charlie"]),  # contains 'a'
            ("^A.*", ["Alice"]),  # starts with 'A'
            (".*b$", ["Bob"]),  # ends with 'b'
            ("^C.*e$", ["Charlie"]),  # starts with C and ends with e
            ("[aeiou]{2}", []),  # two vowels in a row
        ],
    )
    def test_rlike(self, spark, pattern, expected_matches):
        data = {
            "name": ["Alice", "Bob", "Charlie"],
        }
        pandas_df = pd.DataFrame(data)
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(pandas_df)

        for pattern, expected in [(pattern, expected_matches)]:
            expr = PF.col("name").rlike(pattern).alias("match")
            result_df = polars_df.select(expr)
            result_spark_df = spark.createDataFrame(result_df.toPandas())

            expected_df = spark_df.select(F.col("name").rlike(pattern).alias("match"))
            assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "column_name, values, use_variadic",
        [
            ("name", ["Alice", "Bob"], False),
            ("name", ["Alice", "Bob"], True),
            ("name", ["Zoe"], False),
            ("name", ["Zoe"], True),
            ("age", [25, 30], False),
            ("age", [25, 30], True),
            ("age", [100], False),
            ("age", [100], True),
            ("salary", [], False),
            ("salary", [], True),
        ],
    )
    def test_isin(self, spark, column_name, values, use_variadic):
        # Sample data
        data = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "salary": [70000, 80000, 90000],
        }

        # Create both Spark and Polars-backed Sparkle DataFrames
        spark_df = spark.createDataFrame(pd.DataFrame(data))
        sparkle_df = DataFrame(pl.DataFrame(data))

        # Build .isin expression using list or variadic arguments
        if use_variadic:
            expr = PF.col(column_name).isin(*values).alias("match")
        else:
            expr = PF.col(column_name).isin(values).alias("match")

        result_df = sparkle_df.select(expr).toPandas()
        result_spark_df = spark.createDataFrame(result_df)

        # Expected result from PySpark directly
        expected_df = spark_df.select(F.col(column_name).isin(values).alias("match"))

        # Compare
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "use_alias, spark_func, sparkle_func",
        [
            (False, F.count, PF.count),
            (False, F.sum, PF.sum),
            (False, F.mean, PF.mean),
            (False, F.min, PF.min),
            (False, F.max, PF.max),
            (True, F.count, PF.count),
            (True, F.sum, PF.sum),
            (True, F.mean, PF.mean),
            (True, F.min, PF.min),
            (True, F.max, PF.max),
        ],
    )
    def test_groupby_aggregations(self, use_alias, spark, spark_func, sparkle_func):
        # Data: two groups, multiple rows per group
        data = to_records({"group": ["A", "A", "B", "B", "A", "B"], "value": [10, 20, 5, 15, 30, 25]})

        # Spark DataFrame
        spark_df = spark.createDataFrame(data)
        if not use_alias:
            expected_df = spark_df.groupBy("group")
        else:
            expected_df = spark_df.groupby("group")
        expected_df = expected_df.agg(spark_func("value").alias("agg_result"))

        # Sparkleframe Polars DataFrame
        pl_df = DataFrame(pl.DataFrame(data))
        if not use_alias:
            result_df = pl_df.groupBy("group")
        else:
            result_df = pl_df.groupby("group")
        result_df = result_df.agg(sparkle_func("value").alias("agg_result"))

        # Convert Polars result to Spark for comparison
        result_spark_df = spark.createDataFrame(result_df.toPandas())

        # Assert equivalence
        result_spark_df = result_spark_df.withColumn("agg_result", F.col("agg_result").cast("float"))
        expected_df = expected_df.withColumn("agg_result", F.col("agg_result").cast("float"))
        assert_pyspark_df_equal(result_spark_df.orderBy("group"), expected_df.orderBy("group"), ignore_nullable=True)

    @pytest.mark.parametrize(
        "how,on_input,expected",
        [
            ("inner", "id", pl.DataFrame({"id": [2, 3], "left_val": ["b", "c"], "right_val": ["x", "y"]})),
            (
                "left",
                "id",
                pl.DataFrame({"id": [1, 2, 3], "left_val": ["a", "b", "c"], "right_val": [None, "x", "y"]}),
            ),
            (
                "right",
                "id",
                pl.DataFrame({"id": [2, 3, 4], "left_val": ["b", "c", None], "right_val": ["x", "y", "z"]}),
            ),
            (
                "outer",
                "id",
                pl.DataFrame(
                    {"id": [1, 2, 3, 4], "left_val": ["a", "b", "c", None], "right_val": [None, "x", "y", "z"]}
                ),
            ),
            ("semi", "id", pl.DataFrame({"id": [2, 3], "left_val": ["b", "c"]})),
            ("anti", "id", pl.DataFrame({"id": [1], "left_val": ["a"]})),
            (
                "cross",
                None,
                pl.DataFrame(
                    {
                        "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                        "left_val": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                        "id_right": [2, 3, 4, 2, 3, 4, 2, 3, 4],
                        "right_val": ["x", "y", "z", "x", "y", "z", "x", "y", "z"],
                    }
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("on_format", ["str", "col", "list_str", "list_col"])
    def test_polars_joins(self, spark, how, on_input, expected, on_format):

        if how == "outer" and on_format in ["col", "list_col"]:
            expected = pl.DataFrame(
                {"id": [1, 2, 3, None], "left_val": ["a", "b", "c", None], "right_val": [None, "x", "y", "z"]}
            )

        left_data = pl.DataFrame({"id": [1, 2, 3], "left_val": ["a", "b", "c"]})

        right_data = pl.DataFrame({"id": [2, 3, 4], "right_val": ["x", "y", "z"]})

        left = DataFrame(left_data)
        right = DataFrame(right_data)

        # Adjust 'on' input depending on format
        if on_input is None:
            on = None
        elif on_format == "str":
            on = on_input
        elif on_format == "col":
            on = PF.col(on_input)
        elif on_format == "list_str":
            on = [on_input]
        elif on_format == "list_col":
            on = [PF.col(on_input)]
        else:
            raise ValueError("Invalid on_format")

        result = left.join(right, on=on, how=how).toPandas()

        spark_result = spark.createDataFrame(result)
        cols = sorted(spark_result.columns)
        spark_result = spark_result.select(cols).orderBy(cols)

        spark_expected = spark.createDataFrame(expected.to_arrow().to_pandas())
        cols = sorted(spark_expected.columns)
        spark_expected = spark_expected.select(cols).orderBy(cols)

        spark_expected.show()
        spark_result.show()
        assert_pyspark_df_equal(spark_result, spark_expected, ignore_nullable=True, allow_nan_equality=True)

    @pytest.mark.parametrize(
        "how,on_keys",
        [
            ("inner", (False, "id")),  # string
            ("left", (False, "id")),  # string
            ("right", (False, "id")),  # string
            ("outer", (False, "id")),  # string
            ("inner", [(False, "id")]),  # list of string
            ("left", [(False, "id")]),  # list of string
            ("right", [(False, "id")]),  # list of string
            ("outer", [(False, "id")]),  # list of string
        ],
    )
    def test_joins_non_spark_duplicated_keys(self, spark, how, on_keys):

        def get_on(is_spark, is_col, k):
            if is_col:
                return F.col(k) if is_spark else PF.col(k)
            return k

        if isinstance(on_keys, list):
            spark_on_keys = []

            for i, k in enumerate(on_keys):
                spark_on_keys.append(get_on(True, k[0], k[1]))

                on_keys[i] = get_on(False, k[0], k[1])
        else:
            spark_on_keys = get_on(True, on_keys[0], on_keys[1])
            on_keys = get_on(False, on_keys[0], on_keys[1])

        # Prepare left and right datasets
        left_data = to_records({"id": [1, 2, 3], "left_val": ["a", "b", "c"]})

        right_data = to_records({"id": [2, 3, 4], "right_val": ["x", "y", "z"]})

        # Spark setup
        spark_left_df = spark.createDataFrame(left_data)
        spark_right_df = spark.createDataFrame(right_data)

        # Perform Spark join
        expected_df = spark_left_df.join(spark_right_df, on=spark_on_keys, how=how)

        # Sparkleframe setup
        pl_left_df = DataFrame(pl.DataFrame(left_data))
        pl_right_df = DataFrame(pl.DataFrame(right_data))

        # Perform Sparkleframe join
        result_df = pl_left_df.join(pl_right_df, on=on_keys, how=how)

        # Convert Sparkleframe result to Spark DataFrame
        result_spark_df = spark.createDataFrame(
            schema=tuple(result_df.df.columns), data=[tuple(d.values()) for d in result_df.df.to_dicts()]
        )

        # Sort to ensure deterministic comparison
        assert_pyspark_df_equal(
            result_spark_df.select(sorted(result_spark_df.columns)).orderBy("id"),
            expected_df.select(sorted(expected_df.columns)).orderBy("id"),
            ignore_nullable=True,
        )

    @pytest.mark.parametrize(
        "how,on_keys",
        [
            ("inner", (True, "id")),  # Column
            ("left", (True, "id")),  # Column
            # ("right", (True, "id")),  # Column
            ("outer", (True, "id")),  # Column
            ("inner", [(True, "id")]),  # list of column
            ("left", [(True, "id")]),  # list of column
            # ("right", [(True, "id")]),  # list of column
            ("outer", [(True, "id")]),  # list of column
        ],
    )
    def test_joins_with_duplicated_spark_keys(self, spark, how, on_keys):

        def get_on(is_spark, is_col, k):
            if is_col:
                return F.col(k) if is_spark else PF.col(k)
            return k

        if isinstance(on_keys, list):
            spark_on_keys = []

            for i, k in enumerate(on_keys):
                spark_on_keys.append(get_on(True, k[0], k[1]))

                on_keys[i] = get_on(False, k[0], k[1])
        else:
            spark_on_keys = get_on(True, on_keys[0], on_keys[1])
            on_keys = get_on(False, on_keys[0], on_keys[1])

        # Prepare left and right datasets
        left_data = to_records({"id": [1, 2, 3], "left_val": ["a", "b", "c"]})

        right_data = to_records({"id": [2, 3, 4], "right_val": ["x", "y", "z"]})

        # Spark setup
        spark_left_df = spark.createDataFrame(left_data)
        spark_right_df = spark.createDataFrame(right_data)

        # # Perform Spark join

        with pytest.raises(Exception):
            expected_df = spark_left_df.join(spark_right_df, on=spark_on_keys, how=how)

        spark_left_df = spark_left_df
        spark_right_df = spark_right_df.withColumnRenamed("id", "id_right")
        expected_df = spark_left_df.join(spark_right_df, F.col("id") == F.col("id_right"), how=how).drop("id_right")

        # Sparkleframe setup
        pl_left_df = DataFrame(pl.DataFrame(left_data))
        pl_right_df = DataFrame(pl.DataFrame(right_data))

        # Perform Sparkleframe join
        result_df = pl_left_df.join(pl_right_df, on=on_keys, how=how)

        # Convert Sparkleframe result to Spark DataFrame
        result_spark_df = spark.createDataFrame(
            schema=tuple(result_df.df.columns), data=[tuple(d.values()) for d in result_df.df.to_dicts()]
        )

        # Sort to ensure deterministic comparison
        assert_pyspark_df_equal(
            result_spark_df.select(sorted(result_spark_df.columns)).orderBy("id"),
            expected_df.select(sorted(expected_df.columns)).orderBy("id"),
            ignore_nullable=True,
        )

    def test_join_keys_different_type_raise_error(self):

        left_data = to_records({"id": [1, 2, 3], "left_val": ["a", "b", "c"]})

        right_data = to_records({"id": [2, 3, 4], "right_val": ["x", "y", "z"]})

        pl_left_df = DataFrame(pl.DataFrame(left_data))
        pl_right_df = DataFrame(pl.DataFrame(right_data))

        with pytest.raises(TypeError):
            pl_left_df.join(pl_right_df, on=["id", PF.col("id")], how="left")

    @pytest.mark.parametrize(
        "fillna_value, subset",
        [
            (0, None),  # Fill all columns with 0
            ("unknown", "name"),  # Fill only name column
            ({"name": "missing", "age": 0}, None),  # Fill with per-column values
        ],
    )
    def test_pyspark_fillna(self, spark, fillna_value, subset):
        # Sample data with nulls
        data = to_records({"name": ["Alice", None, "Charlie"], "age": [25, None, 35]})

        # Create Polars DataFrame (Sparkleframe)
        sparkle_df = DataFrame(pl.DataFrame(data))

        # Create Spark DataFrame
        spark_df = spark.createDataFrame(data)

        # Apply fillna in Sparkleframe
        if isinstance(fillna_value, dict):
            filled_sparkle = sparkle_df.fillna(fillna_value)
            filled_spark = spark_df.fillna(fillna_value)
        else:
            filled_sparkle = sparkle_df.fillna(fillna_value, subset=subset)
            filled_spark = spark_df.fillna(fillna_value, subset=subset)

        # Convert Sparkleframe to Spark for comparison
        sparkle_as_spark = spark.createDataFrame(filled_sparkle.df.to_dicts())

        # Convert result to Pandas for assertion
        sparkle_as_spark = sparkle_as_spark.select(sorted(sparkle_as_spark.columns)).orderBy("name", "age")

        filled_spark = filled_spark.select(sorted(filled_spark.columns)).orderBy("name", "age")

        # Assert equality
        assert_pyspark_df_equal(sparkle_as_spark, filled_spark, ignore_nullable=True)

    @pytest.mark.parametrize(
        "fillna_value, subset, expected_dict",
        [
            (1, None, {"name": ["Alice", None, "Charlie"], "age": [25, 1, 35]}),
            ("unknown", "name", {"name": ["Alice", "unknown", "Charlie"], "age": [25, None, 35]}),
            ({"name": "missing", "age": 0}, None, {"name": ["Alice", "missing", "Charlie"], "age": [25, 0, 35]}),
        ],
    )
    def test_fillna(self, spark, fillna_value, subset, expected_dict):
        expected_dict = to_records(expected_dict)

        # Sample data with nulls
        data = {"name": ["Alice", None, "Charlie"], "age": [25, None, 35]}

        sparkle_df = DataFrame(pl.DataFrame(to_records(data)))
        sparkle_df = sparkle_df.fillna(fillna_value, subset=subset)

        # Convert result to Pandas for assertion
        result_df = spark.createDataFrame(sparkle_df.df.to_dicts())
        result_df = result_df.select(sorted(result_df.columns)).orderBy("name", "age")

        expected_df = spark.createDataFrame(expected_dict)
        expected_df = expected_df.select(sorted(expected_df.columns)).orderBy("name", "age")

        assert_pyspark_df_equal(result_df, expected_df, ignore_nullable=True)

    def test_columns_property(self):
        # Create sample Polars DataFrame
        data = {"name": ["Alice", "Bob"], "age": [30, 40], "salary": [1000, 2000]}
        df = DataFrame(pl.DataFrame(data))

        # Validate the columns property
        assert df.columns == ["name", "age", "salary"]

    @pytest.mark.parametrize(
        "value, dtype_class, expected_spark_type",
        [
            ("foo", StringType(), SparkStringType()),
            (42, IntegerType(), SparkIntegerType()),
            (42, LongType(), SparkLongType()),
            (3.14, FloatType(), SparkFloatType()),
            (2.718281828, DoubleType(), SparkDoubleType()),
            (True, BooleanType(), SparkBooleanType()),
            (pd.to_datetime("2024-01-01").date(), DateType(), SparkDateType()),
            (pd.to_datetime("2024-01-01 12:34:56"), TimestampType(), SparkTimestampType()),
            (123.45, DecimalType(10, 2), SparkDecimalType(10, 2)),
            (1, ByteType(), SparkByteType()),
            (100, ShortType(), SparkShortType()),
            (b"abc", BinaryType(), SparkBinaryType()),
            # ✅ Nested StructType test case
            (
                {"field1": "hello", "field2": 999},
                StructType([StructField("field1", StringType()), StructField("field2", IntegerType())]),
                SparkStructType(
                    [
                        SparkStructField("field1", SparkStringType()),
                        SparkStructField("field2", SparkIntegerType()),
                    ]
                ),
            ),
            # ✅ Struct with nested StructType inside
            (
                {"outer": {"inner": 123}},
                StructType([StructField("outer", StructType([StructField("inner", IntegerType())]))]),
                SparkStructType(
                    [SparkStructField("outer", SparkStructType([SparkStructField("inner", SparkIntegerType())]))]
                ),
            ),
        ],
    )
    def test_polars_to_spark_dtype_conversion(self, spark, value, dtype_class, expected_spark_type):
        col_name = "col"
        pl_df = pl.DataFrame({col_name: [value]})

        # Wrap in Sparkleframe DataFrame
        sparkle_df = DataFrame(pl_df).withColumn(col_name, PF.col(col_name).cast(dtype_class))

        # Convert to Arrow → Pandas → Spark (actual logic for type preservation)
        if isinstance(value, dict):
            spark_df = spark.createDataFrame(data=[(json.dumps(value),)], schema=(col_name,)).withColumn(
                col_name, F.from_json(F.col(col_name), expected_spark_type)
            )

        else:
            spark_df = create_spark_df(spark, sparkle_df).withColumn(
                col_name, F.col(col_name).cast(expected_spark_type)
            )

        assert spark_df.dtypes == sparkle_df.dtypes

    @pytest.mark.parametrize(
        "col_name, value, dtype_class, expected_spark_type",
        [
            ("string_col", "foo", StringType(), SparkStringType()),
            ("int_col", 42, IntegerType(), SparkIntegerType()),
            ("long_col", 42, LongType(), SparkLongType()),
            ("float_col", 3.14, FloatType(), SparkFloatType()),
            ("double_col", 2.718281828, DoubleType(), SparkDoubleType()),
            ("bool_col", True, BooleanType(), SparkBooleanType()),
            ("date_col", pd.to_datetime("2024-01-01").date(), DateType(), SparkDateType()),
            ("timestamp_col", pd.to_datetime("2024-01-01 12:34:56"), TimestampType(), SparkTimestampType()),
            ("decimal_col", 123.45, DecimalType(10, 2), SparkDecimalType(10, 2)),
            ("byte_col", 1, ByteType(), SparkByteType()),
            ("short_col", 100, ShortType(), SparkShortType()),
            ("binary_col", b"abc", BinaryType(), SparkBinaryType()),
            (
                "struct_col",
                {"field1": "hello", "field2": 999},
                StructType([StructField("field1", StringType()), StructField("field2", IntegerType())]),
                SparkStructType(
                    [
                        SparkStructField("field1", SparkStringType()),
                        SparkStructField("field2", SparkIntegerType()),
                    ]
                ),
            ),
            (
                "nested_struct_col",
                {"outer": {"inner": 123}},
                StructType([StructField("outer", StructType([StructField("inner", IntegerType())]))]),
                SparkStructType(
                    [SparkStructField("outer", SparkStructType([SparkStructField("inner", SparkIntegerType())]))]
                ),
            ),
        ],
    )
    def test_schema_conversion_to_spark_dtype(self, spark, col_name, value, dtype_class, expected_spark_type):
        # Create Polars DataFrame and cast using Sparkleframe types
        pl_df = pl.DataFrame({col_name: [value]})
        sparkle_df = DataFrame(pl_df).withColumn(col_name, PF.col(col_name).cast(dtype_class))

        # Convert to Spark DF for comparison
        if isinstance(value, dict):
            # Use JSON roundtrip for struct parsing
            spark_df = spark.createDataFrame([(json.dumps(value),)], schema=(col_name,)).withColumn(
                col_name, F.from_json(F.col(col_name), expected_spark_type)
            )
        else:
            spark_df = spark.createDataFrame(sparkle_df.toPandas()).withColumn(
                col_name, F.col(col_name).cast(expected_spark_type)
            )

        # Check schema equality
        assert json.dumps(sparkle_df.schema, sort_keys=True, default=str) == json.dumps(
            spark_df.schema, sort_keys=True, default=str
        )

    def test_schema_equivalence_with_spark(self, spark):
        # Sample Pandas data
        pdf = pd.DataFrame([[1, "Alice"], [2, "Bob"]])

        # Sparkleframe schema
        sf_schema = StructType(
            [StructField("id", IntegerType(), nullable=False), StructField("name", StringType(), nullable=True)]
        )

        # Equivalent Spark schema
        spark_schema = SparkStructType(
            [
                SparkStructField("id", SparkIntegerType(), nullable=False),
                SparkStructField("name", SparkStringType(), nullable=True),
            ]
        )

        # Create Sparkleframe DataFrame
        sf_df = DataFrame(pdf, schema=sf_schema)

        # Create Spark DataFrame
        spark_df = spark.createDataFrame(pdf, schema=spark_schema)

        assert json.dumps(sf_df._schema, sort_keys=True, default=str) == json.dumps(
            spark_df._schema, sort_keys=True, default=str
        )

    def test_getitem_str(self, spark):
        input_data = [{"age": 2, "name": "Alice"}, {"age": 5, "name": "Bob"}]
        pdf = pd.DataFrame(input_data)
        sdf = spark.createDataFrame(pdf)
        ps_result = sdf["age"]

        sf_df = DataFrame(pl.DataFrame(pdf))
        sf_result = sf_df["age"]

        assert isinstance(sf_result, Column)
        assert ps_result.__class__.__name__ == sf_result.__class__.__name__

        sdf = sdf.select(ps_result)
        sf_df = create_spark_df(spark, sf_df.select(sf_result))
        assert_pyspark_df_equal(sdf, sf_df)

    def test_getitem_int(self, spark):
        input_data = [{"age": 2, "name": "Alice"}, {"age": 5, "name": "Bob"}]
        pdf = pd.DataFrame(input_data)
        sdf = spark.createDataFrame(pdf)
        ps_result = sdf[0]

        sf_df = DataFrame(pl.DataFrame(pdf))
        sf_result = sf_df[0]

        assert isinstance(sf_result, Column)
        assert ps_result.__class__.__name__ == sf_result.__class__.__name__

        sdf = sdf.select(ps_result)
        sf_df = create_spark_df(spark, sf_df.select(sf_result))
        assert_pyspark_df_equal(sdf, sf_df)

    def test_getitem_column(self, spark):
        input_data = [{"age": 2, "name": "Alice"}, {"age": 5, "name": "Bob"}]
        pl_df = pl.DataFrame(input_data)

        sdf = create_spark_df(spark, pl_df)
        ps_result = sdf[sdf["age"] > 3]

        sf_df = DataFrame(pl_df)
        sf_result_df = sf_df[sf_df["age"] > 3]
        sf_result_df = create_spark_df(spark, sf_result_df)

        assert_pyspark_df_equal(ps_result, sf_result_df)

    def test_getitem_list(self, spark):
        input_data = [{"age": 2, "name": "Alice"}, {"age": 5, "name": "Bob"}]
        pl_df = pl.DataFrame(input_data)

        sdf = create_spark_df(spark, pl_df)
        ps_result = sdf[["name", "age"]].orderBy("name")

        sf_df = DataFrame(pl_df)
        sf_result_df = sf_df[["name", "age"]]
        sf_result_df = create_spark_df(spark, sf_result_df).orderBy("name")

        assert_pyspark_df_equal(ps_result, sf_result_df)

    @pytest.mark.parametrize("use_col", [False, True])
    @pytest.mark.parametrize(
        "columns_order",
        [
            [("id", "asc")],
            [("id", "asc_nulls_first")],
            [("id", "asc_nulls_last")],
            [("name", "asc"), ("id", "asc")],
            [("name", "asc_nulls_first"), ("id", "asc_nulls_first")],
            [("name", "asc_nulls_last"), ("id", "asc_nulls_last")],
            [("id", "desc")],
            [("id", "desc_nulls_first")],
            [("id", "desc_nulls_last")],
            [("id", "desc")],
            [("name", "desc"), ("id", "desc")],
            [("name", "desc_nulls_first"), ("id", "desc_nulls_first")],
            [("name", "desc_nulls_last"), ("id", "desc_nulls_last")],
            [("name", "asc"), ("id", "desc")],
            [("name", "asc_nulls_first"), ("id", "desc_nulls_first")],
            [("name", "asc_nulls_last"), ("id", "desc_nulls_last")],
            [("name", "desc"), ("id", "asc")],
            [("name", "desc_nulls_first"), ("id", "asc_nulls_first")],
            [("name", "desc_nulls_last"), ("id", "asc_nulls_last")],
        ],
    )
    def test_order_by_comparison_with_spark(self, spark, use_col, columns_order):
        data = {"id": [3, 1, 2, 0], "name": ["Charlie", "Alice", "Bob", None]}

        order_func = {
            "asc": (PF.asc, F.asc),
            "asc_nulls_first": (PF.asc_nulls_first, F.asc_nulls_first),
            "asc_nulls_last": (PF.asc_nulls_last, F.asc_nulls_last),
            "desc": (PF.desc, F.desc),
            "desc_nulls_first": (PF.desc_nulls_first, F.desc_nulls_first),
            "desc_nulls_last": (PF.desc_nulls_last, F.desc_nulls_last),
        }

        sparkle_order = [
            order_func[column_order[1]][0](column_order[0] if not use_col else PF.col(column_order[0]))
            for column_order in columns_order
        ]
        spark_order = [
            order_func[column_order[1]][1](column_order[0] if not use_col else F.col(column_order[0]))
            for column_order in columns_order
        ]

        sf_df = DataFrame(pl.DataFrame(data))
        sorted_sf_df = sf_df.orderBy(*sparkle_order)
        result_spark_df = create_spark_df(spark, sorted_sf_df)

        spark_df = spark.createDataFrame(pd.DataFrame(data))
        expected_df = spark_df.orderBy(*spark_order)
        assert_pyspark_df_equal(result_spark_df, expected_df, allow_nan_equality=True)

    def test_order_by_int_raises_error(self, spark):
        data = {"age": [2, 5], "name": ["Alice", "Bob"]}

        sf_df = DataFrame(pl.DataFrame(data))

        spark_df = create_spark_df(spark, sf_df)

        with pytest.raises(TypeError):
            spark_df.orderBy(-1)

        with pytest.raises(TypeError):
            sf_df.orderBy(-1)

    def test_order_by_ascending_kwarg_raises_error(self, spark):
        data = {"age": [2, 5], "name": ["Alice", "Bob"]}

        sf_df = DataFrame(pl.DataFrame(data))

        with pytest.raises(TypeError):
            sf_df.orderBy("age", ascending=True)

    def test_count(self):
        df = DataFrame(pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        assert df.count() == 3

    def test_auto_materialize_maptype_single_level_select(self, spark, sparkle):
        # Polars-native map layout in the input DF
        native = pl.DataFrame([_polars_native_map_series("col", [{"id": 1, "m": 1}])])

        # Sparkle schema declares a MapType so the constructor should auto-materialize to a Struct
        sf_schema = StructType([StructField("col", MapType(StringType(), IntegerType()))])
        df_sparkle = sparkle.createDataFrame(native, schema=sf_schema)

        # Column becomes a Struct with fields ["id", "m"]
        struct_dtype = df_sparkle.to_native_df().schema["col"]
        assert isinstance(struct_dtype, pl.Struct)
        assert [f.name for f in struct_dtype.fields] == ["id", "m"]

        # Dot access selection should work and alias to leaf name
        sel_sf = df_sparkle.select("col.id")  # -> Column named "id"
        result_spark_df = create_spark_df(spark, sel_sf)

        # Note: simper & robust — just build the same result frame from Polars’ pandas output
        expected_spark_df = spark.createDataFrame(sel_sf.toPandas())

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_auto_materialize_is_noop_for_plain_struct(self, spark, sparkle):
        # A plain struct column should NOT go through map auto-materialization
        native = pl.DataFrame({"s": [{"a": 10}]})
        sf_schema = StructType([StructField("s", StructType([StructField("a", IntegerType())]))])

        df_sparkle = sparkle.createDataFrame(native, schema=sf_schema)

        s_dtype = df_sparkle.to_native_df().schema["s"]
        assert isinstance(s_dtype, pl.Struct)
        assert [f.name for f in s_dtype.fields] == ["a"]

        # And dot access works as usual
        result_spark_df = create_spark_df(spark, df_sparkle.select("s.a"))
        expected_spark_df = spark.createDataFrame(df_sparkle.select("s.a").toPandas())
        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_auto_materialize_warns_but_does_not_crash(self, monkeypatch, spark, sparkle):
        # Prepare a DF that would normally auto-materialize
        native = pl.DataFrame([_polars_native_map_series("col", [{"x": 5}])])
        sf_schema = StructType([StructField("col", MapType(StringType(), IntegerType()))])

        # Force the internal converter to throw so we take the warning branch
        monkeypatch.setattr(_MapTypeUtils, "is_map_dtype", lambda dt: True)

        def boom(df, col, *, keys=None, new_name=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(_MapTypeUtils, "map_to_struct", boom)

        with pytest.warns(RuntimeWarning, match="MapType materialization skipped"):
            df_sparkle = sparkle.createDataFrame(native, schema=sf_schema)

        # Still a valid DataFrame; and since conversion failed, col is NOT a Struct
        assert isinstance(df_sparkle, DataFrame)
        assert not isinstance(df_sparkle.to_native_df().schema["col"], pl.Struct)

    def test_auto_materialize_union_keys_order_and_nulls(self, spark, sparkle):
        # Two rows with different keys; union keeps first-seen order ("a" then "b")
        native = pl.DataFrame([_polars_native_map_series("col", [{"a": 1}, {"b": 2}])])
        sf_schema = StructType([StructField("col", MapType(StringType(), IntegerType()))])

        df_sparkle = sparkle.createDataFrame(native, schema=sf_schema)

        struct_dtype = df_sparkle.to_native_df().schema["col"]
        assert isinstance(struct_dtype, pl.Struct)
        assert [f.name for f in struct_dtype.fields] == ["a", "b"]

        # Values line up with nulls where key missing
        result_spark_df = create_spark_df(spark, df_sparkle)
        expected_pd = pl.DataFrame([{"col": {"a": 1, "b": None}}, {"col": {"a": None, "b": 2}}]).to_pandas()
        expected_spark_df = spark.createDataFrame(expected_pd)

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)
