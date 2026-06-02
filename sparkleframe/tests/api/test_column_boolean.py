"""Batch 3 parity tests: Column boolean ops, null predicates, isin, between.

Ported and extended from ``sparkleframe/polarsdf/column_test.py``, covering:
    - ``&`` / ``|`` / ``~`` with Kleene 3-valued logic
    - ``isNull`` / ``isNotNull``
    - ``isin(*values)`` and ``isin([values])`` (incl. null-in-list and
      include-null-on-non-match semantics)
    - ``between(lower, upper)`` (inclusive both ends, null-propagating)

Every test runs against both backends (``polarsdf``, ``pythondf``) and asserts
parity against native PySpark via ``assert_sparkle_spark_frame_are_equal``.
"""
from __future__ import annotations

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import (
    BooleanType as SparkBooleanType,
    IntegerType as SparkIntegerType,
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


def _schema(spark_type, *names):
    return SparkStructType([SparkStructField(n, spark_type, nullable=True) for n in names])


# -----------------------------------------------------------------------------
# Boolean operators: column-vs-column on non-null booleans
# -----------------------------------------------------------------------------


_BOOL_ROWS = [
    {"a": True, "b": True},
    {"a": True, "b": False},
    {"a": False, "b": True},
    {"a": False, "b": False},
]


class TestBooleanOps:
    def test_and_col_col(self, session, F, spark):
        df = session.createDataFrame(_BOOL_ROWS).withColumn(
            "r", F.col("a") & F.col("b")
        ).select("r")
        spark_df = spark.createDataFrame(_BOOL_ROWS).withColumn(
            "r", SF.col("a") & SF.col("b")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_or_col_col(self, session, F, spark):
        df = session.createDataFrame(_BOOL_ROWS).withColumn(
            "r", F.col("a") | F.col("b")
        ).select("r")
        spark_df = spark.createDataFrame(_BOOL_ROWS).withColumn(
            "r", SF.col("a") | SF.col("b")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_invert(self, session, F, spark):
        df = session.createDataFrame(_BOOL_ROWS).withColumn(
            "r", ~F.col("a")
        ).select("r")
        spark_df = spark.createDataFrame(_BOOL_ROWS).withColumn(
            "r", ~SF.col("a")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_and_lit_left(self, session, F, spark):
        # lit(True) & col
        df = session.createDataFrame(_BOOL_ROWS).withColumn(
            "r", F.lit(True) & F.col("a")
        ).select("r")
        spark_df = spark.createDataFrame(_BOOL_ROWS).withColumn(
            "r", SF.lit(True) & SF.col("a")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_or_lit_left(self, session, F, spark):
        df = session.createDataFrame(_BOOL_ROWS).withColumn(
            "r", F.lit(False) | F.col("a")
        ).select("r")
        spark_df = spark.createDataFrame(_BOOL_ROWS).withColumn(
            "r", SF.lit(False) | SF.col("a")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_chained_and_or_not(self, session, F, spark):
        data = [
            {"a": 1, "b": 4, "c": 7},
            {"a": 2, "b": 5, "c": 8},
            {"a": 3, "b": 6, "c": 9},
        ]
        df = session.createDataFrame(data).withColumn(
            "r", (~(F.col("a") > F.lit(1))) | ((F.col("b") < F.lit(6)) & (F.col("c") >= F.lit(7)))
        ).select("r")
        spark_df = spark.createDataFrame(data).withColumn(
            "r", (~(SF.col("a") > SF.lit(1))) | ((SF.col("b") < SF.lit(6)) & (SF.col("c") >= SF.lit(7)))
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# Kleene 3-valued logic — booleans containing null
# -----------------------------------------------------------------------------


# Cartesian over {True, False, None} × {True, False, None}
_KLEENE_ROWS_TUPLES = [
    (True, True),
    (True, False),
    (True, None),
    (False, True),
    (False, False),
    (False, None),
    (None, True),
    (None, False),
    (None, None),
]
_KLEENE_ROWS = [{"a": a, "b": b} for a, b in _KLEENE_ROWS_TUPLES]


class TestKleeneLogic:
    def test_and_with_nulls(self, session, F, spark):
        schema = _schema(SparkBooleanType(), "a", "b")
        spark_df = spark.createDataFrame(_KLEENE_ROWS_TUPLES, schema=schema).withColumn(
            "r", SF.col("a") & SF.col("b")
        ).select("r")
        df = session.createDataFrame(_KLEENE_ROWS).withColumn(
            "r", F.col("a") & F.col("b")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_or_with_nulls(self, session, F, spark):
        schema = _schema(SparkBooleanType(), "a", "b")
        spark_df = spark.createDataFrame(_KLEENE_ROWS_TUPLES, schema=schema).withColumn(
            "r", SF.col("a") | SF.col("b")
        ).select("r")
        df = session.createDataFrame(_KLEENE_ROWS).withColumn(
            "r", F.col("a") | F.col("b")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_invert_with_nulls(self, session, F, spark):
        rows_tuples = [(True,), (False,), (None,)]
        rows = [{"a": t[0]} for t in rows_tuples]
        schema = _schema(SparkBooleanType(), "a")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).withColumn(
            "r", ~SF.col("a")
        ).select("r")
        df = session.createDataFrame(rows).withColumn(
            "r", ~F.col("a")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_and_with_lit_null(self, session, F, spark):
        # col & lit(None) — Kleene: True & null → null, False & null → False
        rows_tuples = [(True,), (False,), (None,)]
        rows = [{"a": t[0]} for t in rows_tuples]
        schema = _schema(SparkBooleanType(), "a")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).withColumn(
            "r", SF.col("a") & SF.lit(None).cast("boolean")
        ).select("r")
        df = session.createDataFrame(rows).withColumn(
            "r", F.col("a") & F.lit(None).cast("boolean")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# isNull / isNotNull
# -----------------------------------------------------------------------------


class TestNullPredicates:
    def test_isnull_int(self, session, F, spark):
        rows_tuples = [(1,), (None,), (3,), (None,), (5,)]
        rows = [{"x": t[0]} for t in rows_tuples]
        schema = _schema(SparkIntegerType(), "x")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).withColumn(
            "r", SF.col("x").isNull()
        ).select("r")
        df = session.createDataFrame(rows).withColumn(
            "r", F.col("x").isNull()
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_isnotnull_int(self, session, F, spark):
        rows_tuples = [(1,), (None,), (3,), (None,), (5,)]
        rows = [{"x": t[0]} for t in rows_tuples]
        schema = _schema(SparkIntegerType(), "x")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).withColumn(
            "r", SF.col("x").isNotNull()
        ).select("r")
        df = session.createDataFrame(rows).withColumn(
            "r", F.col("x").isNotNull()
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_isnull_string(self, session, F, spark):
        rows_tuples = [("a",), (None,), ("",), ("b",)]
        rows = [{"x": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "x")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).withColumn(
            "r", SF.col("x").isNull()
        ).select("r")
        df = session.createDataFrame(rows).withColumn(
            "r", F.col("x").isNull()
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# isin
# -----------------------------------------------------------------------------


class TestIsin:
    @pytest.mark.parametrize("use_variadic", [False, True])
    def test_isin_int_no_nulls(self, session, F, spark, use_variadic):
        data = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        values = [1, 3]

        def build(F_, df):
            col = F_.col("a")
            return df.withColumn("r", col.isin(*values) if use_variadic else col.isin(values)).select("r")

        df = build(F, session.createDataFrame(data))
        spark_df = build(SF, spark.createDataFrame(data))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    @pytest.mark.parametrize("use_variadic", [False, True])
    def test_isin_string(self, session, F, spark, use_variadic):
        data = [{"d": "cat"}, {"d": "dog"}, {"d": "fish"}]
        values = ["cat", "fish"]

        def build(F_, df):
            col = F_.col("d")
            return df.withColumn("r", col.isin(*values) if use_variadic else col.isin(values)).select("r")

        df = build(F, session.createDataFrame(data))
        spark_df = build(SF, spark.createDataFrame(data))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_isin_empty_list(self, session, F, spark):
        data = [{"a": 1}, {"a": 2}]
        df = session.createDataFrame(data).withColumn("r", F.col("a").isin([])).select("r")
        spark_df = spark.createDataFrame(data).withColumn("r", SF.col("a").isin([])).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_isin_column_has_nulls(self, session, F, spark):
        # column has nulls; list has no null. Null values yield null result.
        rows_tuples = [(1,), (2,), (None,), (3,)]
        rows = [{"a": t[0]} for t in rows_tuples]
        schema = _schema(SparkIntegerType(), "a")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).withColumn(
            "r", SF.col("a").isin([1, 3])
        ).select("r")
        df = session.createDataFrame(rows).withColumn(
            "r", F.col("a").isin([1, 3])
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_isin_list_has_null(self, session, F, spark):
        # list has null. Spark semantics: matches→True, value-null→null, non-match→null
        rows_tuples = [(1,), (2,), (None,), (3,)]
        rows = [{"a": t[0]} for t in rows_tuples]
        schema = _schema(SparkIntegerType(), "a")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).withColumn(
            "r", SF.col("a").isin([1, None])
        ).select("r")
        df = session.createDataFrame(rows).withColumn(
            "r", F.col("a").isin([1, None])
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_isin_string_list_has_null(self, session, F, spark):
        rows_tuples = [("a",), ("b",), (None,), ("c",)]
        rows = [{"d": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "d")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).withColumn(
            "r", SF.col("d").isin(["a", None])
        ).select("r")
        df = session.createDataFrame(rows).withColumn(
            "r", F.col("d").isin(["a", None])
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# between
# -----------------------------------------------------------------------------


class TestBetween:
    def test_between_int(self, session, F, spark):
        data = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]
        df = session.createDataFrame(data).withColumn(
            "r", F.col("a").between(2, 4)
        ).select("r")
        spark_df = spark.createDataFrame(data).withColumn(
            "r", SF.col("a").between(2, 4)
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_between_string(self, session, F, spark):
        data = [{"d": "apple"}, {"d": "banana"}, {"d": "cherry"}, {"d": "date"}]
        df = session.createDataFrame(data).withColumn(
            "r", F.col("d").between("banana", "cherry")
        ).select("r")
        spark_df = spark.createDataFrame(data).withColumn(
            "r", SF.col("d").between("banana", "cherry")
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_between_with_null_value(self, session, F, spark):
        rows_tuples = [(1,), (None,), (3,), (5,)]
        rows = [{"a": t[0]} for t in rows_tuples]
        schema = _schema(SparkIntegerType(), "a")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).withColumn(
            "r", SF.col("a").between(2, 4)
        ).select("r")
        df = session.createDataFrame(rows).withColumn(
            "r", F.col("a").between(2, 4)
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_between_inclusive_bounds(self, session, F, spark):
        data = [{"a": 2}, {"a": 4}]  # both endpoints are inclusive
        df = session.createDataFrame(data).withColumn(
            "r", F.col("a").between(2, 4)
        ).select("r")
        spark_df = spark.createDataFrame(data).withColumn(
            "r", SF.col("a").between(2, 4)
        ).select("r")
        assert_sparkle_spark_frame_are_equal(df, spark_df)
