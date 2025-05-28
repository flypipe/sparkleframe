import json

import pandas as pd
import pandas.testing as pdt
import polars as pl
import pytest
from pyspark.sql.functions import get_json_object as spark_get_json_object, lit as spark_lit, when as spark_when, \
    col as spark_col, coalesce as spark_coalesce

from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.functions import col, when, get_json_object, lit, coalesce
from sparkleframe.tests.pyspark_test import assert_pyspark_df_equal
from sparkleframe.tests.utils import to_records

sample_data = {
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": [7, 8, 9]
}

@pytest.fixture
def sparkle_df():
    return DataFrame(pl.DataFrame(sample_data))

@pytest.fixture
def spark_df(spark):
    return spark.createDataFrame(pd.DataFrame(sample_data))

class TestFunctions:
    def test_when(self, spark, sparkle_df, spark_df):
        expr = when(col("a") > 2, "yes").otherwise("no")

        # Add the result column to the full Polars DataFrame
        result_spark_df = spark.createDataFrame(
            sparkle_df.withColumn("result", expr).toPandas()
        )

        # Add result column to full Spark DataFrame
        expected_spark_df = spark_df.withColumn(
            "result", spark_when(spark_col("a") > 2, "yes").otherwise("no")
        )

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    @pytest.mark.parametrize("json_data, path, expected_values", [
        ([json.dumps({"a": 1}), json.dumps({"a": 2})], "$.a", ["1", "2"]),
        ([json.dumps({"a": {"b": 3}}), json.dumps({"a": {"b": 4}})], "$.a.b", ["3", "4"]),
        ([json.dumps({"arr": [10, 20]}), json.dumps({"arr": [30, 40]})], "$.arr[1]", ["20", "40"]),
        ([json.dumps({"a": {"b": [5, 6]}}), json.dumps({"a": {"b": [7, 8]}})], "$.a.b[0]", ["5", "7"]),
        (
        [json.dumps({"items": [{"id": 1}, {"id": 2}]}), json.dumps({"items": [{"id": 3}, {"id": 4}]})], "$.items[1].id",
        ["2", "4"]),
    ])
    def test_get_json_object(self, spark, json_data, path, expected_values):
        df = pd.DataFrame({"json_col": json_data})

        spark_df = spark.createDataFrame(df)
        expected_df = spark_df.select(spark_get_json_object("json_col", path).alias("result"))

        polars_df = DataFrame(pl.DataFrame(df))
        result_df = polars_df.select(get_json_object("json_col", path).alias("result"))
        result_spark_df = spark.createDataFrame(result_df.toPandas())

        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize("literal_value", [
        42,  # int
        3.14,  # float
        "hello",  # string
        True,  # boolean
        None,  # null
    ])
    def test_lit_against_spark(self, spark, literal_value):
        df = pl.DataFrame({"x": [1, 2, 3]})
        sparkle_df = DataFrame(df)
        result_df = sparkle_df.select(lit(literal_value).alias("value")).toPandas()

        # Result using Spark
        spark_df = spark.createDataFrame(pd.DataFrame({"x": [1, 2, 3]}))
        expected_df = spark_df.select(spark_lit(literal_value).alias("value")).toPandas()

        # Compare using pandas
        pdt.assert_frame_equal(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_dtype=False  # Important: ignores schema/type mismatches
        )

    @pytest.mark.parametrize("literal_value", [
        42,  # int
        3.14,  # float
        "hello",  # string
        True,  # boolean
        None,  # null
    ])
    def test_lit_against_spark(self, spark, literal_value):
        df = pl.DataFrame({"x": [1, 2, 3]})
        sparkle_df = DataFrame(df)
        result_df = sparkle_df.select(lit(literal_value).alias("value")).toPandas()

        # Result using Spark
        spark_df = spark.createDataFrame(pd.DataFrame({"x": [1, 2, 3]}))
        expected_df = spark_df.select(spark_lit(literal_value).alias("value")).toPandas()

        # Compare using pandas
        pdt.assert_frame_equal(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_dtype=False  # Important: ignores schema/type mismatches
        )

    @pytest.mark.parametrize("a_vals, b_vals, expected_vals", [
        ([None, 2, None], [1, None, 3], [1, 2, 3]),
        ([None, None, None], [None, None, None], [None, None, None]),
        ([None, 5, 6], ["x", "y", None], ["x", 5, 6]),
        (["", None, "z"], ["a", "b", None], ["", "b", "z"]),
    ])
    def test_coalesce_against_spark(self, spark, a_vals, b_vals, expected_vals):
        # Build pandas DataFrame for both Spark and Polars
        data = to_records({"a": a_vals, "b": b_vals})

        # Spark setup
        spark_df = spark.createDataFrame(data)
        expected_spark_df = spark_df.select(spark_coalesce(spark_col("a"), spark_col("b")).alias("result"))

        # sparkleframe setup
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(coalesce(col("a"), col("b")).alias("result"))
        result_spark_df = spark.createDataFrame(result_df.toPandas())

        # Compare using PySpark equality
        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)