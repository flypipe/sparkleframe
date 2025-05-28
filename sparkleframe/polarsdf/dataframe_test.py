import numpy as np
import pytest
import polars as pl
import pandas as pd
from pyspark.sql.functions import col as spark_col

from sparkleframe.tests.pyspark_test import assert_pyspark_df_equal
from sparkleframe.polarsdf.dataframe import DataFrame
import sparkleframe.polarsdf.functions as PF

from pyspark.sql import functions as F
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
    BinaryType as SparkBinaryType
)

from sparkleframe.polarsdf.types import (
    StringType, IntegerType, LongType, FloatType,
    DoubleType, BooleanType, DateType, TimestampType, DecimalType, ByteType, ShortType, BinaryType
)
import pyarrow as pa

sample_data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [70000, 80000, 90000],
    "birth_date": ["1990-01-01", "1985-05-15", "1970-12-30"],
    "login_time": ["2024-01-01T08:00:00", "2024-01-02T09:30:00", "2024-01-03T11:45:00"]
}

@pytest.fixture
def sparkle_df():
    return DataFrame(pl.DataFrame(sample_data))

@pytest.fixture
def spark_df(spark):
    return spark.createDataFrame(pd.DataFrame(sample_data))

class TestDataFrame:
    def test_select_by_column_name(self, spark, sparkle_df, spark_df):
        result_spark_df = spark.createDataFrame(sparkle_df.select("name").toPandas())
        expected_spark_df = spark_df.select("name")

        assert_pyspark_df_equal(result_spark_df, expected_spark_df)

    def test_select_by_expression(self, spark, sparkle_df, spark_df):
        result_spark_df = spark.createDataFrame(
            sparkle_df.select(
                PF.col("name"),
                PF.col("salary") * 1.1
            ).toPandas()
        )

        expected_spark_df = spark_df.select(
            spark_col("name"),
            (spark_col("salary") * 1.1).alias("salary")
        )

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, precision=5)

    def test_select_all_columns_with_aliases(self, spark, sparkle_df, spark_df):
        # Define aliases
        aliases = {
            "name": "employee_name",
            "age": "employee_age",
            "salary": "employee_salary"
        }

        # Apply aliases using DataFrame
        polars_selected = sparkle_df.select(
            *(PF.col(col).alias(alias) for col, alias in aliases.items())
        )
        result_spark_df = spark.createDataFrame(polars_selected.toPandas())

        # Apply the same aliases using PySpark
        expected_spark_df = spark_df.select(
            *(spark_col(col).alias(alias) for col, alias in aliases.items())
        )

        assert_pyspark_df_equal(result_spark_df, expected_spark_df)

    def test_with_column_add(self, spark, sparkle_df, spark_df):
        result_spark_df = spark.createDataFrame(
            sparkle_df.withColumn("bonus", PF.col("salary") * 0.1).toPandas()
        )

        expected_spark_df = spark_df.withColumn("bonus", spark_col("salary") * 0.1)

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, precision=5)

    def test_with_column_replace(self, spark, sparkle_df, spark_df):
        result_spark_df = spark.createDataFrame(
            sparkle_df.withColumn("salary", PF.col("salary") * 2).toPandas()
        )

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

    @pytest.mark.parametrize("col_name, data_type_class, spark_type", [
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
    ])
    def test_cast_types(self, spark, sparkle_df, spark_df, col_name, data_type_class, spark_type):
        # Apply the cast using your API
        expr = PF.col(col_name).cast(data_type_class)
        polars_result_df = sparkle_df.select(expr.alias(col_name)).to_native_df()

        # Apply the cast using PySpark
        spark_result_df = spark_df.select(
            F.col(col_name).cast(spark_type).alias(col_name)
        )

        # Extract data types for comparison
        polars_dtype = polars_result_df.schema[col_name]
        spark_dtype = spark_result_df.schema[col_name].dataType

        print(f"Polars dtype: {polars_dtype}")
        print(f"Spark dtype: {spark_dtype}")

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
        pl_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "amount": [10.5, 20.0, 30.25],
            "active": [True, False, True]
        })
        sparkle_df = DataFrame(pl_df)

        arrow_table = sparkle_df.to_arrow()
        pandas_df = arrow_table.to_pandas(types_mapper=pd.ArrowDtype)

        result_spark_df = spark.createDataFrame(pandas_df)

        expected_spark_df = spark.createDataFrame(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "amount": [10.5, 20.0, 30.25],
            "active": [True, False, True]
        }))

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_create_polars_from_arrow_generated_by_spark(self, spark):
        # Step 1: Create a Spark DataFrame
        spark_df = spark.createDataFrame([
            ("Alice", 25, True),
            ("Bob", 30, False),
            ("Charlie", 35, True)
        ], ["name", "age", "active"])

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
        data = {
            "id": [1, 2, 3, 4],
            "name": ["Alice", None, "Charlie", None]
        }
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

    @pytest.mark.parametrize("literal, op_name, expr_func", [
        (10, "+", lambda col_expr: 10 + col_expr),
        (10, "-", lambda col_expr: 10 - col_expr),
        (10, "*", lambda col_expr: 10 * col_expr),
        (10, "/", lambda col_expr: 10 / col_expr),
    ])
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
            (spark_col("a").__radd__(literal) if op_name == "+" else
             spark_col("a").__rsub__(literal) if op_name == "-" else
             spark_col("a").__rmul__(literal) if op_name == "*" else
             spark_col("a").__rtruediv__(literal)).alias("result")
        )

        # Assert equality
        assert_pyspark_df_equal(result_spark_df, expected_spark_df, precision=5)

    @pytest.mark.parametrize("description, expr_func", [
        ("AND", lambda a, b, c: (a > 1) & (b < 6)),
        ("OR", lambda a, b, c: (a > 2) | (b < 6)),
        ("chained AND-OR", lambda a, b, c: ((a > 1) & (b < 6)) | (c > 7)),
        ("chained OR-AND", lambda a, b, c: (a < 2) | ((b == 5) & (c < 9))),
    ])
    def test_logical_operations(self, spark, description, expr_func):
        data = {
            "a": [1, 2, 3, 4],
            "b": [10, 5, 3, 8],
            "c": [7, 12, 9, 4]
        }

        sparkle_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(pd.DataFrame(data))

        expr = expr_func(PF.col("a"), PF.col("b"), PF.col("c")).alias("result")
        expected_expr = expr_func(F.col("a"), F.col("b"), F.col("c")).alias("result")

        result_df = spark.createDataFrame(sparkle_df.select(expr).toPandas())
        expected_df = spark_df.select(expected_expr)

        assert_pyspark_df_equal(result_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize("description, expr_func", [
        ("filter by single column", lambda a, b, c: a > 2),
        ("filter with AND", lambda a, b, c: (a > 1) & (b < 10)),
        ("filter with OR", lambda a, b, c: (a < 2) | (c > 7)),
        ("chained AND-OR", lambda a, b, c: ((a > 1) & (b < 6)) | (c > 7)),
    ])
    def test_filter_and_where(self, spark, description, expr_func):
        data = {
            "a": [1, 2, 3, 4],
            "b": [10, 5, 3, 8],
            "c": [7, 12, 9, 4]
        }

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

    import re
    from pyspark.sql.functions import col as spark_col, rlike as spark_rlike

    @pytest.mark.parametrize("pattern, expected_matches", [
        (".*a.*", ["Alice", "Charlie"]),  # contains 'a'
        ("^A.*", ["Alice"]),  # starts with 'A'
        (".*b$", ["Bob"]),  # ends with 'b'
        ("^C.*e$", ["Charlie"]),  # starts with C and ends with e
        ("[aeiou]{2}", []),  # two vowels in a row
    ])
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

