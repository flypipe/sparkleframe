"""Batch 21 parity tests: module-level sort helpers in ``F``.

Mirrors the Column-method sort helpers covered in B9 (``test_dataframe_ordering``)
but exercises the module-level ``F.asc / F.desc / F.asc_nulls_first / ...``
functions. Each helper is tested with both ``str`` and ``Column`` input, and
null placement is verified end-to-end via ``DataFrame.orderBy``.

Order-sensitive: assertions use ``assert_frame_ordered_equal``.
"""
from __future__ import annotations

import pyspark.sql.functions as SF

from sparkleframe.tests.utils import assert_frame_ordered_equal


_ORDER_DATA = [
    {"a": 3, "b": "y"},
    {"a": 1, "b": "x"},
    {"a": 2, "b": "z"},
    {"a": 4, "b": "x"},
]

_NULL_DATA = [
    {"a": 3, "b": "y"},
    {"a": None, "b": "x"},
    {"a": 2, "b": None},
    {"a": 1, "b": "z"},
    {"a": None, "b": "a"},
]


# -----------------------------------------------------------------------------
# F.asc
# -----------------------------------------------------------------------------


class TestFAsc:
    def test_asc_str(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy(F.asc("a"))
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy(SF.asc("a"))
        assert_frame_ordered_equal(df, spark_df)

    def test_asc_column(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy(F.asc(F.col("a")))
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy(SF.asc(SF.col("a")))
        assert_frame_ordered_equal(df, spark_df)

    def test_asc_nulls_default_first(self, session, F, spark):
        # PySpark default for ASC is NULLS FIRST.
        df = session.createDataFrame(_NULL_DATA).orderBy(F.asc("a"))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.asc("a"))
        assert_frame_ordered_equal(df, spark_df)


# -----------------------------------------------------------------------------
# F.desc
# -----------------------------------------------------------------------------


class TestFDesc:
    def test_desc_str(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy(F.desc("a"))
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy(SF.desc("a"))
        assert_frame_ordered_equal(df, spark_df)

    def test_desc_column(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy(F.desc(F.col("a")))
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy(SF.desc(SF.col("a")))
        assert_frame_ordered_equal(df, spark_df)

    def test_desc_nulls_default_last(self, session, F, spark):
        # PySpark default for DESC is NULLS LAST.
        df = session.createDataFrame(_NULL_DATA).orderBy(F.desc("a"))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.desc("a"))
        assert_frame_ordered_equal(df, spark_df)


# -----------------------------------------------------------------------------
# F.asc_nulls_first / F.asc_nulls_last
# -----------------------------------------------------------------------------


class TestFAscNulls:
    def test_asc_nulls_first_str(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.asc_nulls_first("a"))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.asc_nulls_first("a"))
        assert_frame_ordered_equal(df, spark_df)

    def test_asc_nulls_first_column(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.asc_nulls_first(F.col("a")))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.asc_nulls_first(SF.col("a")))
        assert_frame_ordered_equal(df, spark_df)

    def test_asc_nulls_last_str(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.asc_nulls_last("a"))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.asc_nulls_last("a"))
        assert_frame_ordered_equal(df, spark_df)

    def test_asc_nulls_last_column(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.asc_nulls_last(F.col("a")))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.asc_nulls_last(SF.col("a")))
        assert_frame_ordered_equal(df, spark_df)


# -----------------------------------------------------------------------------
# F.desc_nulls_first / F.desc_nulls_last
# -----------------------------------------------------------------------------


class TestFDescNulls:
    def test_desc_nulls_first_str(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.desc_nulls_first("a"))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.desc_nulls_first("a"))
        assert_frame_ordered_equal(df, spark_df)

    def test_desc_nulls_first_column(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.desc_nulls_first(F.col("a")))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.desc_nulls_first(SF.col("a")))
        assert_frame_ordered_equal(df, spark_df)

    def test_desc_nulls_last_str(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.desc_nulls_last("a"))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.desc_nulls_last("a"))
        assert_frame_ordered_equal(df, spark_df)

    def test_desc_nulls_last_column(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.desc_nulls_last(F.col("a")))
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.desc_nulls_last(SF.col("a")))
        assert_frame_ordered_equal(df, spark_df)


# -----------------------------------------------------------------------------
# Multi-key mixed-direction with F.* helpers
# -----------------------------------------------------------------------------


class TestFMultiKey:
    def test_multi_key_asc_then_desc(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy(F.asc("b"), F.desc("a"))
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy(SF.asc("b"), SF.desc("a"))
        assert_frame_ordered_equal(df, spark_df)

    def test_multi_key_with_nulls_mixed(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(
            F.asc_nulls_last("a"), F.desc_nulls_last("b")
        )
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(
            SF.asc_nulls_last("a"), SF.desc_nulls_last("b")
        )
        assert_frame_ordered_equal(df, spark_df)
