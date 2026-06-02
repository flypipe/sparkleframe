"""Batch 15 parity tests: math functions in ``F``.

Covered:
    - ``F.abs(col)`` — absolute value, null preservation, expression input.
    - ``F.floor(col)`` — Spark returns LongType.
    - ``F.round(col, scale)`` — Spark HALF_UP rounding (NOT banker's; that's bround).
    - ``F.pow(base, exp)`` — accepts col/str/lit on either side; always Double.
    - ``F.isnan(col)`` — True only for IEEE 754 NaN; False for null in Spark.
    - ``F.try_divide(a, b)`` — null on divide-by-zero / null operand.
    - ``F.nullif(e1, e2)`` — null when e1 == e2 (both non-null); else e1.

All tests run against both backends via the ``(session, F, spark)`` fixtures
and assert parity with native PySpark.
"""
from __future__ import annotations

import pyspark.sql.functions as SF
from pyspark.sql.types import (
    DoubleType as SparkDoubleType,
    IntegerType as SparkIntegerType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


_DOUBLE_NULLABLE_SCHEMA = SparkStructType(
    [SparkStructField("v", SparkDoubleType(), nullable=True)]
)
_INT_NULLABLE_SCHEMA = SparkStructType(
    [SparkStructField("v", SparkIntegerType(), nullable=True)]
)


# ---------------------------------------------------------------------------
# F.abs
# ---------------------------------------------------------------------------


class TestAbs:
    def test_abs_doubles_positive_negative_zero_null(self, session, F, spark):
        rows = [{"v": -3.5}, {"v": 0.0}, {"v": 2.1}, {"v": -7.0}, {"v": None}]
        sdf = session.createDataFrame(rows, schema="v DOUBLE").select(F.abs(F.col("v")).alias("a"))
        pdf = spark.createDataFrame(rows, schema=_DOUBLE_NULLABLE_SCHEMA).select(SF.abs(SF.col("v")).alias("a"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_abs_integers(self, session, F, spark):
        rows = [{"v": -10}, {"v": 0}, {"v": 5}, {"v": -1}, {"v": None}]
        sdf = session.createDataFrame(rows, schema="v INT").select(F.abs("v").alias("a"))
        pdf = spark.createDataFrame(rows, schema=_INT_NULLABLE_SCHEMA).select(SF.abs("v").alias("a"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_abs_of_subtraction_expression(self, session, F, spark):
        rows = [{"a": 1.0, "b": 4.0}, {"a": 5.0, "b": 2.0}, {"a": 3.0, "b": 3.0}]
        sdf = session.createDataFrame(rows).withColumn("d", F.abs(F.col("a") - F.col("b")))
        pdf = spark.createDataFrame(rows).withColumn("d", SF.abs(SF.col("a") - SF.col("b")))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.floor
# ---------------------------------------------------------------------------


class TestFloor:
    def test_floor_doubles(self, session, F, spark):
        rows = [{"v": 1.9}, {"v": 2.1}, {"v": -0.5}, {"v": 0.0}, {"v": -1.5}, {"v": None}]
        sdf = session.createDataFrame(rows, schema="v DOUBLE").select(F.floor("v").alias("f"))
        pdf = spark.createDataFrame(rows, schema=_DOUBLE_NULLABLE_SCHEMA).select(SF.floor("v").alias("f"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_floor_integers_passthrough(self, session, F, spark):
        rows = [{"v": 3}, {"v": -2}, {"v": 0}, {"v": None}]
        sdf = session.createDataFrame(rows, schema="v INT").select(F.floor("v").alias("f"))
        pdf = spark.createDataFrame(rows, schema=_INT_NULLABLE_SCHEMA).select(SF.floor("v").alias("f"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.round (HALF_UP)
# ---------------------------------------------------------------------------


class TestRound:
    def test_round_default_scale_zero(self, session, F, spark):
        # Spark uses HALF_UP: 2.5 -> 3, -2.5 -> -3
        rows = [{"v": 1.4}, {"v": 1.5}, {"v": 2.5}, {"v": -2.5}, {"v": -1.4}, {"v": 0.0}, {"v": None}]
        sdf = session.createDataFrame(rows, schema="v DOUBLE").select(F.round("v").alias("r"))
        pdf = spark.createDataFrame(rows, schema=_DOUBLE_NULLABLE_SCHEMA).select(SF.round("v").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_round_scale_two(self, session, F, spark):
        rows = [{"v": 1.235}, {"v": 1.225}, {"v": -1.235}, {"v": 0.005}, {"v": None}]
        sdf = session.createDataFrame(rows, schema="v DOUBLE").select(F.round("v", 2).alias("r"))
        pdf = spark.createDataFrame(rows, schema=_DOUBLE_NULLABLE_SCHEMA).select(SF.round("v", 2).alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_round_negative_scale(self, session, F, spark):
        # scale=-1 rounds to nearest 10; HALF_UP
        rows = [{"v": 15.0}, {"v": 24.0}, {"v": 25.0}, {"v": -25.0}, {"v": None}]
        sdf = session.createDataFrame(rows, schema="v DOUBLE").select(F.round("v", -1).alias("r"))
        pdf = spark.createDataFrame(rows, schema=_DOUBLE_NULLABLE_SCHEMA).select(SF.round("v", -1).alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_round_column_arg(self, session, F, spark):
        rows = [{"v": 2.345}, {"v": 2.355}, {"v": None}]
        sdf = session.createDataFrame(rows, schema="v DOUBLE").select(F.round(F.col("v"), 1).alias("r"))
        pdf = spark.createDataFrame(rows, schema=_DOUBLE_NULLABLE_SCHEMA).select(SF.round(SF.col("v"), 1).alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.pow
# ---------------------------------------------------------------------------


class TestPow:
    def test_pow_col_col_floats(self, session, F, spark):
        rows = [{"base": 2.0, "exp": 3.0}, {"base": 3.0, "exp": 2.0}, {"base": 10.0, "exp": 0.0}]
        sdf = session.createDataFrame(rows).select(F.pow("base", "exp").alias("p"))
        pdf = spark.createDataFrame(rows).select(SF.pow("base", "exp").alias("p"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_pow_col_col_ints(self, session, F, spark):
        rows = [{"base": 2, "exp": 3}, {"base": 5, "exp": 2}, {"base": 10, "exp": 0}]
        sdf = session.createDataFrame(rows).select(F.pow("base", "exp").alias("p"))
        pdf = spark.createDataFrame(rows).select(SF.pow("base", "exp").alias("p"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_pow_col_lit(self, session, F, spark):
        rows = [{"base": 2.0}, {"base": 3.0}, {"base": 4.0}]
        sdf = session.createDataFrame(rows).select(F.pow(F.col("base"), F.lit(2)).alias("p"))
        pdf = spark.createDataFrame(rows).select(SF.pow(SF.col("base"), SF.lit(2)).alias("p"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_pow_lit_col(self, session, F, spark):
        rows = [{"exp": 0.0}, {"exp": 1.0}, {"exp": 2.0}, {"exp": 3.0}]
        sdf = session.createDataFrame(rows).select(F.pow(F.lit(2), F.col("exp")).alias("p"))
        pdf = spark.createDataFrame(rows).select(SF.pow(SF.lit(2), SF.col("exp")).alias("p"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_pow_with_nulls(self, session, F, spark):
        # Use short column names because pythondf's createDataFrame doesn't parse
        # multi-column schema strings; PySpark schema string accepts "a DOUBLE, b DOUBLE".
        rows = [{"a": 2.0, "b": 3.0}, {"a": None, "b": 2.0}, {"a": 3.0, "b": None}]
        schema = SparkStructType([
            SparkStructField("a", SparkDoubleType(), nullable=True),
            SparkStructField("b", SparkDoubleType(), nullable=True),
        ])
        sdf = session.createDataFrame(rows, schema="a DOUBLE, b DOUBLE").select(F.pow("a", "b").alias("p"))
        pdf = spark.createDataFrame(rows, schema=schema).select(SF.pow("a", "b").alias("p"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.isnan
# ---------------------------------------------------------------------------


class TestIsnan:
    def test_isnan_finite_nan_zero(self, session, F, spark):
        # Spark contract: isnan(null) -> False (not null). Plain values False.
        rows = [{"v": 1.0}, {"v": float("nan")}, {"v": 0.0}, {"v": -1.0}]
        sdf = session.createDataFrame(rows, schema="v DOUBLE").select(F.isnan("v").alias("n"))
        pdf = spark.createDataFrame(rows, schema=_DOUBLE_NULLABLE_SCHEMA).select(SF.isnan("v").alias("n"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_isnan_null_input(self, session, F, spark):
        rows = [{"v": 1.0}, {"v": None}, {"v": float("nan")}]
        sdf = session.createDataFrame(rows, schema="v DOUBLE").select(F.isnan("v").alias("n"))
        pdf = spark.createDataFrame(rows, schema=_DOUBLE_NULLABLE_SCHEMA).select(SF.isnan("v").alias("n"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.try_divide
# ---------------------------------------------------------------------------


class TestTryDivide:
    def test_try_divide_normal(self, session, F, spark):
        rows = [{"a": 10.0, "b": 2.0}, {"a": 9.0, "b": 3.0}, {"a": 1.0, "b": 4.0}]
        sdf = session.createDataFrame(rows).select(F.try_divide("a", "b").alias("d"))
        pdf = spark.createDataFrame(rows).select(SF.try_divide("a", "b").alias("d"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_try_divide_by_zero_returns_null(self, session, F, spark):
        rows = [{"a": 10.0, "b": 0.0}, {"a": 7.0, "b": 0.0}]
        sdf = session.createDataFrame(rows).select(F.try_divide("a", "b").alias("d"))
        pdf = spark.createDataFrame(rows).select(SF.try_divide("a", "b").alias("d"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_try_divide_with_nulls(self, session, F, spark):
        rows = [{"a": 10.0, "b": 2.0}, {"a": 9.0, "b": 0.0}, {"a": None, "b": 3.0}, {"a": 5.0, "b": None}]
        schema = SparkStructType([
            SparkStructField("a", SparkDoubleType(), nullable=True),
            SparkStructField("b", SparkDoubleType(), nullable=True),
        ])
        sdf = session.createDataFrame(rows, schema="a DOUBLE, b DOUBLE").select(F.try_divide("a", "b").alias("d"))
        pdf = spark.createDataFrame(rows, schema=schema).select(SF.try_divide("a", "b").alias("d"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.nullif
# ---------------------------------------------------------------------------


class TestNullif:
    def test_nullif_equal_becomes_null(self, session, F, spark):
        rows = [{"a": 1, "b": 1}, {"a": 2, "b": 3}, {"a": 3, "b": 3}]
        sdf = session.createDataFrame(rows).select(F.nullif("a", "b").alias("n"))
        pdf = spark.createDataFrame(rows).select(SF.nullif("a", "b").alias("n"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_nullif_with_nulls_returns_e1(self, session, F, spark):
        # Either side null -> equality unknown -> return e1 (Spark semantics).
        rows = [{"a": 5, "b": None}, {"a": None, "b": 2}, {"a": 3, "b": None}, {"a": None, "b": None}]
        schema = SparkStructType([
            SparkStructField("a", SparkIntegerType(), nullable=True),
            SparkStructField("b", SparkIntegerType(), nullable=True),
        ])
        sdf = session.createDataFrame(rows, schema="a INT, b INT").select(F.nullif("a", "b").alias("n"))
        pdf = spark.createDataFrame(rows, schema=schema).select(SF.nullif("a", "b").alias("n"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_nullif_strings(self, session, F, spark):
        rows = [{"a": "x", "b": "x"}, {"a": "x", "b": "y"}, {"a": "z", "b": "z"}]
        sdf = session.createDataFrame(rows).select(F.nullif("a", "b").alias("n"))
        pdf = spark.createDataFrame(rows).select(SF.nullif("a", "b").alias("n"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)
