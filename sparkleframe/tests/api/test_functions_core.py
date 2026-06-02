"""Batch 13 parity tests: core ``functions`` primitives.

Covers the already-implemented building blocks shared by every other functions
batch:

    - ``F.col`` — identity in ``select``, aliasing, arithmetic composition
    - ``F.lit`` — int / float / str / bool / None literals, row-count broadcast
    - ``F.when(...).otherwise(...)`` — single, chained, no-otherwise (NULL),
      conditions using ``F.col`` comparisons, ``lit``/``col`` branch values
    - ``F.coalesce`` — 2-arg, 3-arg, all-null row, mix of col + lit
    - ``F.least`` / ``F.greatest`` — 2-arg, 3-arg, null skipping, numeric and
      string types
    - Edge: ``least`` / ``greatest`` with a single argument raises

Every test runs against both backends (``polarsdf`` and ``pythondf``) and is
asserted against native PySpark via :func:`assert_sparkle_spark_frame_are_equal`.
"""
from __future__ import annotations

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import (
    BooleanType as SparkBooleanType,
    DoubleType as SparkDoubleType,
    IntegerType as SparkIntegerType,
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


def _schema(*fields):
    return SparkStructType(
        [SparkStructField(name, dtype, nullable=True) for name, dtype in fields]
    )


# -----------------------------------------------------------------------------
# F.col
# -----------------------------------------------------------------------------


class TestCol:
    def test_col_identity_in_select(self, session, F, spark):
        data = [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]
        df = session.createDataFrame(data).select(F.col("a"))
        spark_df = spark.createDataFrame(data).select(SF.col("a"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_col_with_alias(self, session, F, spark):
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        df = session.createDataFrame(data).select(F.col("a").alias("renamed"))
        spark_df = spark.createDataFrame(data).select(SF.col("a").alias("renamed"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_col_with_arithmetic(self, session, F, spark):
        data = [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]
        df = session.createDataFrame(data).withColumn(
            "r", F.col("a") * F.col("b") + F.col("a")
        )
        spark_df = spark.createDataFrame(data).withColumn(
            "r", SF.col("a") * SF.col("b") + SF.col("a")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# F.lit
# -----------------------------------------------------------------------------


_LIT_LITERALS = [
    ("int", 42),
    ("float", 3.14),
    ("string", "hello"),
    ("bool_true", True),
    ("bool_false", False),
]


class TestLit:
    @pytest.mark.parametrize("label, value", _LIT_LITERALS)
    def test_lit_broadcast(self, session, F, spark, label, value):
        data = [{"x": 1}, {"x": 2}, {"x": 3}]
        df = session.createDataFrame(data).select(F.lit(value).alias("v"))
        spark_df = spark.createDataFrame(data).select(SF.lit(value).alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_lit_none_typed_as_int(self, session, F, spark):
        """``lit(None)`` standalone needs an explicit type for parity (Spark
        defaults to StringType; we cast both sides to int to avoid that
        backend-specific divergence which is out of scope here)."""
        data = [{"x": 1}, {"x": 2}]
        # Wrap None inside a branch whose other side is a typed numeric so both
        # backends agree on int type.
        df = session.createDataFrame(data).withColumn(
            "n", F.when(F.col("x") < F.lit(0), F.lit(1)).otherwise(F.lit(None))
        )
        spark_df = spark.createDataFrame(data).withColumn(
            "n", SF.when(SF.col("x") < SF.lit(0), SF.lit(1)).otherwise(SF.lit(None))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_lit_combined_with_col(self, session, F, spark):
        data = [{"x": 1}, {"x": 2}, {"x": 3}]
        df = session.createDataFrame(data).withColumn("r", F.col("x") + F.lit(100))
        spark_df = spark.createDataFrame(data).withColumn("r", SF.col("x") + SF.lit(100))
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# F.when(...).otherwise(...)
# -----------------------------------------------------------------------------


class TestWhenOtherwise:
    def test_single_when_with_string_branches(self, session, F, spark):
        data = [{"a": 1}, {"a": 3}, {"a": 5}]
        df = session.createDataFrame(data).withColumn(
            "r", F.when(F.col("a") > F.lit(2), "big").otherwise("small")
        )
        spark_df = spark.createDataFrame(data).withColumn(
            "r", SF.when(SF.col("a") > SF.lit(2), "big").otherwise("small")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_chained_when_when_otherwise(self, session, F, spark):
        data = [{"a": 1}, {"a": 5}, {"a": 10}, {"a": 20}]
        df = session.createDataFrame(data).withColumn(
            "r",
            F.when(F.col("a") < F.lit(5), "tiny")
            .when(F.col("a") < F.lit(15), "mid")
            .otherwise("large"),
        )
        spark_df = spark.createDataFrame(data).withColumn(
            "r",
            SF.when(SF.col("a") < SF.lit(5), "tiny")
            .when(SF.col("a") < SF.lit(15), "mid")
            .otherwise("large"),
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_when_no_otherwise_yields_null(self, session, F, spark):
        # PySpark: when no branch matches and no .otherwise(), result is NULL.
        # Use a string branch so the inferred type is StringType on both sides.
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        df = session.createDataFrame(data).withColumn(
            "r", F.when(F.col("a") == F.lit(2), "match")
        )
        spark_df = spark.createDataFrame(data).withColumn(
            "r", SF.when(SF.col("a") == SF.lit(2), "match")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_when_with_col_branch_values(self, session, F, spark):
        data = [{"a": 1, "b": 100, "c": 999}, {"a": 5, "b": 200, "c": 888}]
        df = session.createDataFrame(data).withColumn(
            "r",
            F.when(F.col("a") < F.lit(3), F.col("b")).otherwise(F.col("c")),
        )
        spark_df = spark.createDataFrame(data).withColumn(
            "r",
            SF.when(SF.col("a") < SF.lit(3), SF.col("b")).otherwise(SF.col("c")),
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_when_with_lit_branch_values(self, session, F, spark):
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        df = session.createDataFrame(data).withColumn(
            "r",
            F.when(F.col("a") == F.lit(2), F.lit(99)).otherwise(F.lit(0)),
        )
        spark_df = spark.createDataFrame(data).withColumn(
            "r",
            SF.when(SF.col("a") == SF.lit(2), SF.lit(99)).otherwise(SF.lit(0)),
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_when_condition_uses_two_col_comparison(self, session, F, spark):
        data = [{"a": 1, "b": 5}, {"a": 5, "b": 1}, {"a": 3, "b": 3}]
        df = session.createDataFrame(data).withColumn(
            "r",
            F.when(F.col("a") < F.col("b"), "lt")
            .when(F.col("a") > F.col("b"), "gt")
            .otherwise("eq"),
        )
        spark_df = spark.createDataFrame(data).withColumn(
            "r",
            SF.when(SF.col("a") < SF.col("b"), "lt")
            .when(SF.col("a") > SF.col("b"), "gt")
            .otherwise("eq"),
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# F.coalesce
# -----------------------------------------------------------------------------


class TestCoalesce:
    def test_coalesce_two_columns(self, session, F, spark):
        data = [
            {"a": None, "b": 1},
            {"a": 2, "b": None},
            {"a": 3, "b": 4},
        ]
        schema = _schema(("a", SparkIntegerType()), ("b", SparkIntegerType()))
        df = session.createDataFrame(data).withColumn(
            "r", F.coalesce(F.col("a"), F.col("b"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.coalesce(SF.col("a"), SF.col("b"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_coalesce_three_columns(self, session, F, spark):
        data = [
            {"a": None, "b": None, "c": 30},
            {"a": None, "b": 20, "c": None},
            {"a": 10, "b": 20, "c": 30},
        ]
        schema = _schema(
            ("a", SparkIntegerType()),
            ("b", SparkIntegerType()),
            ("c", SparkIntegerType()),
        )
        df = session.createDataFrame(data).withColumn(
            "r", F.coalesce(F.col("a"), F.col("b"), F.col("c"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.coalesce(SF.col("a"), SF.col("b"), SF.col("c"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_coalesce_all_null_row_is_null(self, session, F, spark):
        data = [
            {"a": None, "b": None},
            {"a": 7, "b": None},
            {"a": None, "b": 8},
        ]
        schema = _schema(("a", SparkIntegerType()), ("b", SparkIntegerType()))
        df = session.createDataFrame(data).withColumn(
            "r", F.coalesce(F.col("a"), F.col("b"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.coalesce(SF.col("a"), SF.col("b"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_coalesce_col_and_lit_fallback(self, session, F, spark):
        data = [{"a": None}, {"a": 5}, {"a": None}]
        schema = _schema(("a", SparkIntegerType()))
        df = session.createDataFrame(data).withColumn(
            "r", F.coalesce(F.col("a"), F.lit(-1))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.coalesce(SF.col("a"), SF.lit(-1))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# F.least
# -----------------------------------------------------------------------------


class TestLeast:
    def test_least_two_int_columns(self, session, F, spark):
        data = [{"a": 1, "b": 4}, {"a": 5, "b": 2}, {"a": 3, "b": 3}]
        schema = _schema(("a", SparkIntegerType()), ("b", SparkIntegerType()))
        df = session.createDataFrame(data).withColumn(
            "r", F.least(F.col("a"), F.col("b"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.least(SF.col("a"), SF.col("b"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_least_three_int_columns(self, session, F, spark):
        data = [
            {"a": 1, "b": 4, "c": 7},
            {"a": 5, "b": 2, "c": 8},
            {"a": 9, "b": 6, "c": 3},
        ]
        schema = _schema(
            ("a", SparkIntegerType()),
            ("b", SparkIntegerType()),
            ("c", SparkIntegerType()),
        )
        df = session.createDataFrame(data).withColumn(
            "r", F.least(F.col("a"), F.col("b"), F.col("c"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.least(SF.col("a"), SF.col("b"), SF.col("c"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_least_skips_nulls(self, session, F, spark):
        data = [
            {"a": None, "b": 5},
            {"a": 3, "b": None},
            {"a": None, "b": None},
            {"a": 1, "b": 9},
        ]
        schema = _schema(("a", SparkIntegerType()), ("b", SparkIntegerType()))
        df = session.createDataFrame(data).withColumn(
            "r", F.least(F.col("a"), F.col("b"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.least(SF.col("a"), SF.col("b"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_least_strings(self, session, F, spark):
        data = [
            {"a": "banana", "b": "apple"},
            {"a": "delta", "b": "echo"},
            {"a": "zebra", "b": "yak"},
        ]
        schema = _schema(("a", SparkStringType()), ("b", SparkStringType()))
        df = session.createDataFrame(data).withColumn(
            "r", F.least(F.col("a"), F.col("b"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.least(SF.col("a"), SF.col("b"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_least_single_arg_raises(self, F):
        with pytest.raises((ValueError, TypeError, Exception)):
            F.least(F.col("a"))


# -----------------------------------------------------------------------------
# F.greatest
# -----------------------------------------------------------------------------


class TestGreatest:
    def test_greatest_two_int_columns(self, session, F, spark):
        data = [{"a": 1, "b": 4}, {"a": 5, "b": 2}, {"a": 3, "b": 3}]
        schema = _schema(("a", SparkIntegerType()), ("b", SparkIntegerType()))
        df = session.createDataFrame(data).withColumn(
            "r", F.greatest(F.col("a"), F.col("b"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.greatest(SF.col("a"), SF.col("b"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_greatest_three_int_columns(self, session, F, spark):
        data = [
            {"a": 1, "b": 4, "c": 7},
            {"a": 5, "b": 2, "c": 8},
            {"a": 9, "b": 6, "c": 3},
        ]
        schema = _schema(
            ("a", SparkIntegerType()),
            ("b", SparkIntegerType()),
            ("c", SparkIntegerType()),
        )
        df = session.createDataFrame(data).withColumn(
            "r", F.greatest(F.col("a"), F.col("b"), F.col("c"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.greatest(SF.col("a"), SF.col("b"), SF.col("c"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_greatest_skips_nulls(self, session, F, spark):
        data = [
            {"a": None, "b": 5},
            {"a": 3, "b": None},
            {"a": None, "b": None},
            {"a": 1, "b": 9},
        ]
        schema = _schema(("a", SparkIntegerType()), ("b", SparkIntegerType()))
        df = session.createDataFrame(data).withColumn(
            "r", F.greatest(F.col("a"), F.col("b"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.greatest(SF.col("a"), SF.col("b"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_greatest_strings(self, session, F, spark):
        data = [
            {"a": "banana", "b": "apple"},
            {"a": "delta", "b": "echo"},
            {"a": "zebra", "b": "yak"},
        ]
        schema = _schema(("a", SparkStringType()), ("b", SparkStringType()))
        df = session.createDataFrame(data).withColumn(
            "r", F.greatest(F.col("a"), F.col("b"))
        )
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", SF.greatest(SF.col("a"), SF.col("b"))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_greatest_single_arg_raises(self, F):
        with pytest.raises((ValueError, TypeError, Exception)):
            F.greatest(F.col("a"))
