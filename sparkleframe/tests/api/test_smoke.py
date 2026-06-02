"""Smoke parity tests covering Batch A composition primitives.

These exist to validate the test infrastructure (backend fixture, parity helper,
xfail registry) before any sub-agent batches run. If these pass on both backends
with PySpark parity, the infrastructure is good.
"""
from __future__ import annotations

import pyspark.sql.functions as SF

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


class TestSmokeParity:
    def test_lit_and_arithmetic(self, session, F, spark):
        data = [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]
        df = session.createDataFrame(data).withColumn("c", F.col("a") + F.col("b"))
        spark_df = spark.createDataFrame(data).withColumn("c", SF.col("a") + SF.col("b"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_filter_equality(self, session, F, spark):
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        df = session.createDataFrame(data).filter(F.col("a") == F.lit(2))
        spark_df = spark.createDataFrame(data).filter(SF.col("a") == SF.lit(2))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_when_otherwise(self, session, F, spark):
        data = [{"a": 1}, {"a": 3}, {"a": 5}]
        df = session.createDataFrame(data).withColumn(
            "r", F.when(F.col("a") > F.lit(2), "big").otherwise("small")
        )
        spark_df = spark.createDataFrame(data).withColumn(
            "r", SF.when(SF.col("a") > SF.lit(2), "big").otherwise("small")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_coalesce(self, session, F, spark):
        data = [{"a": None, "b": 1}, {"a": 2, "b": None}, {"a": None, "b": None}]
        df = session.createDataFrame(data).withColumn("c", F.coalesce(F.col("a"), F.col("b")))
        spark_df = spark.createDataFrame(data).withColumn("c", SF.coalesce(SF.col("a"), SF.col("b")))
        assert_sparkle_spark_frame_are_equal(df, spark_df)
