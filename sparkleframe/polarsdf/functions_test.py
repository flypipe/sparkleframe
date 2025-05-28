import json

import pandas as pd
import pandas.testing as pdt
import polars as pl
import pytest
from pyspark.sql.functions import get_json_object as spark_get_json_object, lit as spark_lit, when as spark_when, \
    col as spark_col, coalesce as spark_coalesce, round as spark_round
from pyspark.sql.types import StringType, StructField

from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.functions import col, when, get_json_object, lit, coalesce, round
from sparkleframe.polarsdf.types import StructType
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
    # def test_when(self, spark, sparkle_df, spark_df):
    #     expr = when(col("a") > 2, "yes").otherwise("no")
    #
    #     # Add the result column to the full Polars DataFrame
    #     result_spark_df = spark.createDataFrame(
    #         sparkle_df.withColumn("result", expr).toPandas()
    #     )
    #
    #     # Add result column to full Spark DataFrame
    #     expected_spark_df = spark_df.withColumn(
    #         "result", spark_when(spark_col("a") > 2, "yes").otherwise("no")
    #     )
    #
    #     assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)
    #
    # def test_chained_when_boolean_output(self, spark):
    #     # Input data
    #     data = to_records({
    #         "b": ["A", "B", "C", "D"],
    #         "c": ["b", "e", "g", "z"]
    #     })
    #
    #
    #     polars_df = DataFrame(pl.DataFrame(data))
    #     expr = (
    #         when((col("b") == "A") & (col("c").isin("A", "b", "c")), True)
    #         .when((col("b") == "B") & (col("c").isin("d", "e")), True)
    #         .when((col("b") == "C") & (col("c").isin("f", "g", "h", "i")), True)
    #         .otherwise(False)
    #     )
    #
    #     result_df = polars_df.withColumn("result", expr)
    #     result_spark_df = spark.createDataFrame(result_df.df.to_dicts())
    #
    #     # Expected result using PySpark chained when()
    #     expected_df = spark.createDataFrame(data).withColumn("result",
    #         spark_when((spark_col("b") == "A") & (spark_col("c").isin("A", "b", "c")), True)
    #         .when((spark_col("b") == "B") & (spark_col("c").isin("d", "e")), True)
    #         .when((spark_col("b") == "C") & (spark_col("c").isin("f", "g", "h", "i")), True)
    #         .otherwise(False)
    #     )
    #
    #     # Compare results
    #     assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)
    #
    # @pytest.mark.parametrize("json_data, path, expected_values", [
    #     ([json.dumps({"a": 1}), json.dumps({"a": 2})], "$.a", ["1", "2"]),
    #     ([json.dumps({"a": {"b": 3}}), json.dumps({"a": {"b": 4}})], "$.a.b", ["3", "4"]),
    #     ([json.dumps({"arr": [10, 20]}), json.dumps({"arr": [30, 40]})], "$.arr[1]", ["20", "40"]),
    #     ([json.dumps({"a": {"b": [5, 6]}}), json.dumps({"a": {"b": [7, 8]}})], "$.a.b[0]", ["5", "7"]),
    #     (
    #     [json.dumps({"items": [{"id": 1}, {"id": 2}]}), json.dumps({"items": [{"id": 3}, {"id": 4}]})], "$.items[1].id",
    #     ["2", "4"]),
    # ])
    # def test_get_json_object(self, spark, json_data, path, expected_values):
    #     df = pd.DataFrame({"json_col": json_data})
    #
    #     spark_df = spark.createDataFrame(df)
    #     expected_df = spark_df.select(spark_get_json_object("json_col", path).alias("result"))
    #
    #     polars_df = DataFrame(pl.DataFrame(df))
    #     result_df = polars_df.select(get_json_object("json_col", path).alias("result"))
    #     result_spark_df = spark.createDataFrame(result_df.toPandas())
    #
    #     assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)
    #
    # @pytest.mark.parametrize("literal_value", [
    #     42,  # int
    #     3.14,  # float
    #     "hello",  # string
    #     True,  # boolean
    #     None,  # null
    # ])
    # def test_lit_against_spark(self, spark, literal_value):
    #     df = pl.DataFrame({"x": [1, 2, 3]})
    #     sparkle_df = DataFrame(df)
    #     result_df = sparkle_df.select(lit(literal_value).alias("value")).toPandas()
    #
    #     # Result using Spark
    #     spark_df = spark.createDataFrame(pd.DataFrame({"x": [1, 2, 3]}))
    #     expected_df = spark_df.select(spark_lit(literal_value).alias("value")).toPandas()
    #
    #     # Compare using pandas
    #     pdt.assert_frame_equal(
    #         result_df.reset_index(drop=True),
    #         expected_df.reset_index(drop=True),
    #         check_dtype=False  # Important: ignores schema/type mismatches
    #     )
    #
    # @pytest.mark.parametrize("literal_value", [
    #     42,  # int
    #     3.14,  # float
    #     "hello",  # string
    #     True,  # boolean
    #     None,  # null
    # ])
    # def test_lit_against_spark(self, spark, literal_value):
    #     df = pl.DataFrame({"x": [1, 2, 3]})
    #     sparkle_df = DataFrame(df)
    #     result_df = sparkle_df.select(lit(literal_value).alias("value")).toPandas()
    #
    #     # Result using Spark
    #     spark_df = spark.createDataFrame(pd.DataFrame({"x": [1, 2, 3]}))
    #     expected_df = spark_df.select(spark_lit(literal_value).alias("value")).toPandas()
    #
    #     # Compare using pandas
    #     pdt.assert_frame_equal(
    #         result_df.reset_index(drop=True),
    #         expected_df.reset_index(drop=True),
    #         check_dtype=False  # Important: ignores schema/type mismatches
    #     )
    #
    # @pytest.mark.parametrize("a_vals, b_vals, expected_vals", [
    #     ([None, 2, None], [1, None, 3], [1, 2, 3]),
    #     ([None, None, None], [None, None, None], [None, None, None]),
    #     ([None, 5, 6], ["x", "y", None], ["x", 5, 6]),
    #     (["", None, "z"], ["a", "b", None], ["", "b", "z"]),
    # ])
    # def test_coalesce_against_spark(self, spark, a_vals, b_vals, expected_vals):
    #     # Build pandas DataFrame for both Spark and Polars
    #     data = to_records({"a": a_vals, "b": b_vals})
    #
    #     # Spark setup
    #     if expected_vals == [None, None, None]:
    #         spark_df = spark.createDataFrame(data=[("1", "1")], schema="a: string, b: string")
    #         spark_df = (
    #             spark_df
    #             .withColumn("a", spark_lit(None))
    #             .withColumn("b", spark_lit(None))
    #         )
    #
    #         expected_spark_df = spark.createDataFrame(data=[("1", "1")], schema="a: string, b: string")
    #         expected_spark_df = (
    #             expected_spark_df
    #             .withColumn("a", spark_lit(None))
    #             .withColumn("b", spark_lit(None))
    #         )
    #     else:
    #         spark_df = spark.createDataFrame(data)
    #         expected_spark_df = spark_df.select(spark_coalesce(spark_col("a"), spark_col("b")).alias("result"))
    #
    #     # sparkleframe setup
    #     polars_df = DataFrame(pl.DataFrame(data))
    #     result_df = polars_df.select(coalesce(col("a"), col("b")).alias("result"))
    #
    #     if result_df.df.to_dicts() == [{'result': None}, {'result': None}, {'result': None}]:
    #         result_spark_df = (
    #             spark_df
    #             .withColumn("a", spark_lit(None))
    #             .withColumn("b", spark_lit(None))
    #         )
    #     else:
    #         result_spark_df = spark.createDataFrame(result_df.df.to_dicts())
    #
    #     # Compare using PySpark equality
    #     assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)


    @pytest.mark.parametrize("values, scale", [
        ([1.234, 2.345, 3.456], 0),  # round to integer
        ([1.234, 2.345, 3.456], 1),  # round to 1 decimal
        ([1.234, 2.345, 3.456], 2),  # round to 2 decimals
        ([None, 2.555, 3.666], 1),  # include None
    ])
    def test_round_against_spark(self, spark, values, scale):
        data = to_records({"x": values})

        # Sparkleframe / Polars
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(round(col("x"), scale).alias("rounded")).toPandas()

        # PySpark
        spark_df = spark.createDataFrame(data)
        expected_df = spark_df.select(spark_round(spark_col("x"), scale).alias("rounded")).toPandas()

        # Compare using pandas
        pdt.assert_frame_equal(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-5
        )
