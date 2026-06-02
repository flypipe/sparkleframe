"""Batch 2 parity tests: Column arithmetic, comparisons, and unary ops.

Ported from ``sparkleframe/polarsdf/column_test.py``'s ``TestColumn`` and
``TestArithmeticParityWithSpark`` classes, scoped to:
    - binary arithmetic: ``+ - * / % **`` and their reflected forms
    - unary ops: ``- +``
    - binary comparisons: ``== != < <= > >=``

Every test runs against both backends (``polarsdf``, ``pythondf``) and asserts
parity against native PySpark via ``assert_sparkle_spark_frame_are_equal``.
"""
from __future__ import annotations

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import (
    DoubleType as SparkDoubleType,
    FloatType as SparkFloatType,
    IntegerType as SparkIntegerType,
    LongType as SparkLongType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


# -----------------------------------------------------------------------------
# Basic arithmetic / comparison parity
# -----------------------------------------------------------------------------


_SAMPLE_DATA = [
    {"a": 1, "b": 4, "c": 7, "d": "cat"},
    {"a": 2, "b": 5, "c": 8, "d": "dog"},
    {"a": 3, "b": 6, "c": 9, "d": "bird"},
]


_BINARY_ARITH_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("truediv", lambda a, b: a / b),
    ("mod", lambda a, b: a % b),
    ("pow", lambda a, b: a ** b),
]

_REVERSE_ARITH_OPS = [
    ("radd", lambda c: 10 + c),
    ("rsub", lambda c: 10 - c),
    ("rmul", lambda c: 10 * c),
    ("rtruediv", lambda c: 10 / c),
    ("rmod", lambda c: 10 % c),
    ("rpow", lambda c: 2 ** c),
]

_COMPARISON_OPS = [
    ("eq", lambda a, b: a == b),
    ("ne", lambda a, b: a != b),
    ("lt", lambda a, b: a < b),
    ("le", lambda a, b: a <= b),
    ("gt", lambda a, b: a > b),
    ("ge", lambda a, b: a >= b),
]


class TestColumnArithmetic:
    @pytest.mark.parametrize("op_name, op_func", _BINARY_ARITH_OPS)
    def test_arithmetic_col_col(self, session, F, spark, op_name, op_func):
        df = session.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", op_func(F.col("a"), F.col("b"))
        ).select("r")
        spark_df = spark.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", op_func(SF.col("a"), SF.col("b"))
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    @pytest.mark.parametrize("op_name, op_func", _REVERSE_ARITH_OPS)
    def test_reverse_arithmetic_literal_col(self, session, F, spark, op_name, op_func):
        df = session.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", op_func(F.col("a"))
        ).select("r")
        spark_df = spark.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", op_func(SF.col("a"))
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_chained_expression(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", (F.col("c") - F.col("a") + F.col("b")) * F.col("b") / F.col("a")
        ).select("r")
        spark_df = spark.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", (SF.col("c") - SF.col("a") + SF.col("b")) * SF.col("b") / SF.col("a")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_unary_neg_column(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", -F.col("a")
        ).select("r")
        spark_df = spark.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", -SF.col("a")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_unary_pos_column_is_identity(self, session, F, spark):
        """PySpark has no ``+col`` operator on ``Column``, so we just verify the
        backend's ``+col`` is a no-op equivalent to ``col``. Spark baseline uses
        plain ``SF.col`` for the parity assertion."""
        df = session.createDataFrame(_SAMPLE_DATA).withColumn("r", +F.col("a")).select("r")
        spark_df = spark.createDataFrame(_SAMPLE_DATA).withColumn("r", SF.col("a")).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)


class TestColumnComparison:
    @pytest.mark.parametrize("op_name, op_func", _COMPARISON_OPS)
    def test_comparison_col_col(self, session, F, spark, op_name, op_func):
        df = session.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", op_func(F.col("a"), F.col("b"))
        ).select("r")
        spark_df = spark.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", op_func(SF.col("a"), SF.col("b"))
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    @pytest.mark.parametrize("op_name, op_func", _COMPARISON_OPS)
    def test_comparison_col_lit(self, session, F, spark, op_name, op_func):
        df = session.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", op_func(F.col("a"), F.lit(2))
        ).select("r")
        spark_df = spark.createDataFrame(_SAMPLE_DATA).withColumn(
            "r", op_func(SF.col("a"), SF.lit(2))
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# Null propagation
# -----------------------------------------------------------------------------


_NULL_ROWS_INT = [
    {"a": 1, "b": 4},
    {"a": 2, "b": None},
    {"a": None, "b": 6},
    {"a": None, "b": None},
]

_NULL_ROWS_DOUBLE = [
    {"a": 1.0, "b": 2.0},
    {"a": 2.0, "b": None},
    {"a": None, "b": 6.0},
    {"a": None, "b": None},
]


def _typed_schema(spark_type, *names):
    return SparkStructType([SparkStructField(n, spark_type, nullable=True) for n in names])


class TestArithmeticWithNulls:
    @pytest.mark.parametrize("op_name, op_func", _BINARY_ARITH_OPS)
    def test_int_nulls(self, session, F, spark, op_name, op_func):
        # b=0 isn't present, so div/mod/pow don't error
        data = [(r["a"], r["b"]) for r in _NULL_ROWS_INT]
        schema = _typed_schema(SparkIntegerType(), "a", "b")
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", op_func(SF.col("a"), SF.col("b"))
        ).select("r")
        df = session.createDataFrame(_NULL_ROWS_INT).withColumn(
            "r", op_func(F.col("a"), F.col("b"))
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    @pytest.mark.parametrize("op_name, op_func", _BINARY_ARITH_OPS)
    def test_double_nulls(self, session, F, spark, op_name, op_func):
        data = [(r["a"], r["b"]) for r in _NULL_ROWS_DOUBLE]
        schema = _typed_schema(SparkDoubleType(), "a", "b")
        spark_df = spark.createDataFrame(data, schema=schema).withColumn(
            "r", op_func(SF.col("a"), SF.col("b"))
        ).select("r")
        df = session.createDataFrame(_NULL_ROWS_DOUBLE).withColumn(
            "r", op_func(F.col("a"), F.col("b"))
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# Typed parity: numeric types
# -----------------------------------------------------------------------------


_NUMERIC_TYPE_FIXTURES = [
    ("int", SparkIntegerType(), [2, 5, 10], [1, 3, 4]),
    ("long", SparkLongType(), [2, 5, 10], [1, 3, 4]),
    ("float", SparkFloatType(), [1.5, 2.5, 3.5], [0.5, 1.0, 2.0]),
    ("double", SparkDoubleType(), [1.5, 2.5, 3.5], [0.5, 1.0, 2.0]),
]


class TestArithmeticTypedParity:
    @pytest.mark.parametrize(
        "dtype_label, spark_type, values_a, values_b", _NUMERIC_TYPE_FIXTURES
    )
    @pytest.mark.parametrize("op_name, op_func", _BINARY_ARITH_OPS)
    def test_arithmetic_col_col_typed(
        self, session, F, spark, dtype_label, spark_type, values_a, values_b, op_name, op_func
    ):
        rows = list(zip(values_a, values_b))
        schema = _typed_schema(spark_type, "a", "b")
        spark_df = spark.createDataFrame(rows, schema=schema).withColumn(
            "r", op_func(SF.col("a"), SF.col("b"))
        ).select("r")
        dict_rows = [{"a": a, "b": b} for a, b in rows]
        df = session.createDataFrame(dict_rows).withColumn(
            "r", op_func(F.col("a"), F.col("b"))
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    @pytest.mark.parametrize(
        "dtype_label, spark_type, values_a, values_b", _NUMERIC_TYPE_FIXTURES
    )
    @pytest.mark.parametrize("op_name, op_func", _COMPARISON_OPS)
    def test_comparison_col_col_typed(
        self, session, F, spark, dtype_label, spark_type, values_a, values_b, op_name, op_func
    ):
        rows = list(zip(values_a, values_b))
        schema = _typed_schema(spark_type, "a", "b")
        spark_df = spark.createDataFrame(rows, schema=schema).withColumn(
            "r", op_func(SF.col("a"), SF.col("b"))
        ).select("r")
        dict_rows = [{"a": a, "b": b} for a, b in rows]
        df = session.createDataFrame(dict_rows).withColumn(
            "r", op_func(F.col("a"), F.col("b"))
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    @pytest.mark.parametrize(
        "dtype_label, spark_type, values_a, values_b", _NUMERIC_TYPE_FIXTURES
    )
    def test_unary_neg_typed(
        self, session, F, spark, dtype_label, spark_type, values_a, values_b
    ):
        rows = list(zip(values_a, values_b))
        schema = _typed_schema(spark_type, "a", "b")
        spark_df = spark.createDataFrame(rows, schema=schema).withColumn(
            "r", -SF.col("a")
        ).select("r")
        dict_rows = [{"a": a, "b": b} for a, b in rows]
        df = session.createDataFrame(dict_rows).withColumn(
            "r", -F.col("a")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)
