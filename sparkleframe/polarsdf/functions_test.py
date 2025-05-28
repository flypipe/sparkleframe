import json
from pyspark.sql.functions import get_json_object as spark_get_json_object
import pytest
import polars as pl
import pandas as pd
from pyspark.sql import functions as F
from sparkleframe.polarsdf.functions import col, when, get_json_object
from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.tests.pyspark_test import assert_pyspark_df_equal

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
            "result", F.when(F.col("a") > 2, "yes").otherwise("no")
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