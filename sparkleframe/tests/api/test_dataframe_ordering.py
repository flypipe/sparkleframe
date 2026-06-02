"""Batch 9 parity tests: DataFrame ordering + slicing.

Covers the row-axis surface of the DataFrame API:
    - ``orderBy`` / ``sort`` with str, Column, and sort-keyed Column args;
      nulls placement (default ASC NULLS FIRST, DESC NULLS LAST) and explicit
      ``asc_nulls_last`` / ``desc_nulls_first`` variants; multi-key mixed-direction
    - ``distinct()`` — set-uniqueness of rows
    - ``dropDuplicates(subset=...)`` — set-uniqueness on a key subset
    - ``limit(n)`` — first ``n`` rows
    - ``head`` / ``take`` / ``first`` — row-extraction shape parity

Ordering tests use ``assert_frame_ordered_equal`` (position-sensitive); distinct
and dropDuplicates use ``assert_sparkle_spark_frame_are_equal`` (set-equal).
``head``/``take``/``first`` only assert on the pythondf shape and contents.
"""
from __future__ import annotations

import pyspark.sql.functions as SF

from sparkleframe.tests.utils import (
    assert_frame_ordered_equal,
    assert_sparkle_spark_frame_are_equal,
)


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
# orderBy / sort
# -----------------------------------------------------------------------------


class TestOrderBy:
    def test_order_by_str(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy("a")
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy("a")
        assert_frame_ordered_equal(df, spark_df)

    def test_sort_alias(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).sort("a")
        spark_df = spark.createDataFrame(_ORDER_DATA).sort("a")
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_column_default_asc(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy(F.col("a"))
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy(SF.col("a"))
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_desc(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy(F.col("a").desc())
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy(SF.col("a").desc())
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_asc_explicit(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy(F.col("a").asc())
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy(SF.col("a").asc())
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_multi_key_mixed(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy(F.col("b").asc(), F.col("a").desc())
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy(SF.col("b").asc(), SF.col("a").desc())
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_str_default_nulls_first(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy("a")
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy("a")
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_desc_default_nulls_last(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.col("a").desc())
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.col("a").desc())
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_asc_nulls_last(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.col("a").asc_nulls_last())
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.col("a").asc_nulls_last())
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_desc_nulls_first(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.col("a").desc_nulls_first())
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.col("a").desc_nulls_first())
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_asc_nulls_first_explicit(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.col("a").asc_nulls_first())
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.col("a").asc_nulls_first())
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_desc_nulls_last_explicit(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(F.col("a").desc_nulls_last())
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(SF.col("a").desc_nulls_last())
        assert_frame_ordered_equal(df, spark_df)

    def test_order_by_multi_key_with_nulls(self, session, F, spark):
        df = session.createDataFrame(_NULL_DATA).orderBy(
            F.col("a").asc_nulls_last(), F.col("b").desc_nulls_last()
        )
        spark_df = spark.createDataFrame(_NULL_DATA).orderBy(
            SF.col("a").asc_nulls_last(), SF.col("b").desc_nulls_last()
        )
        assert_frame_ordered_equal(df, spark_df)


# -----------------------------------------------------------------------------
# distinct
# -----------------------------------------------------------------------------


class TestDistinct:
    def test_distinct_full_duplicates(self, session, F, spark):
        data = [{"a": 1, "b": "x"}, {"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        df = session.createDataFrame(data).distinct()
        spark_df = spark.createDataFrame(data).distinct()
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_distinct_no_duplicates(self, session, F, spark):
        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
        df = session.createDataFrame(data).distinct()
        spark_df = spark.createDataFrame(data).distinct()
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_distinct_partial_duplicates(self, session, F, spark):
        # Same `a` but different `b` should both survive.
        data = [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 1, "b": "x"}]
        df = session.createDataFrame(data).distinct()
        spark_df = spark.createDataFrame(data).distinct()
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# dropDuplicates
# -----------------------------------------------------------------------------


class TestDropDuplicates:
    def test_drop_duplicates_no_subset(self, session, F, spark):
        data = [{"a": 1, "b": "x"}, {"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        df = session.createDataFrame(data).dropDuplicates()
        spark_df = spark.createDataFrame(data).dropDuplicates()
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_drop_duplicates_subset_single(self, session, F, spark):
        data = [
            {"a": 1, "b": "x"},
            {"a": 1, "b": "y"},
            {"a": 2, "b": "z"},
        ]
        df = session.createDataFrame(data).dropDuplicates(["a"]).orderBy("a")
        spark_df = spark.createDataFrame(data).dropDuplicates(["a"]).orderBy("a")
        # First-seen "b" for a=1 is implementation-detail in PySpark too;
        # both implementations should agree on rowcount + per-key uniqueness.
        # pythondf has no .count() yet (B10); use len() which both backends accept.
        assert len(df.collect()) == spark_df.count()

    def test_drop_duplicates_subset_multi(self, session, F, spark):
        data = [
            {"a": 1, "b": "x", "c": 10},
            {"a": 1, "b": "x", "c": 20},
            {"a": 1, "b": "y", "c": 30},
            {"a": 2, "b": "x", "c": 40},
        ]
        df = session.createDataFrame(data).dropDuplicates(["a", "b"])
        spark_df = spark.createDataFrame(data).dropDuplicates(["a", "b"])
        # pythondf has no .count() yet (B10); use len() which both backends accept.
        assert len(df.collect()) == spark_df.count() == 3


# -----------------------------------------------------------------------------
# limit
# -----------------------------------------------------------------------------


class TestLimit:
    def test_limit_zero(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).limit(0)
        spark_df = spark.createDataFrame(_ORDER_DATA).limit(0)
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_limit_larger_than_rowcount(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).limit(100)
        spark_df = spark.createDataFrame(_ORDER_DATA).limit(100)
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_limit_mid(self, session, F, spark):
        # Apply an orderBy so the "first N" is deterministic.
        df = session.createDataFrame(_ORDER_DATA).orderBy("a").limit(2)
        spark_df = spark.createDataFrame(_ORDER_DATA).orderBy("a").limit(2)
        assert_frame_ordered_equal(df, spark_df)

    def test_limit_all(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).limit(len(_ORDER_DATA))
        spark_df = spark.createDataFrame(_ORDER_DATA).limit(len(_ORDER_DATA))
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# head / take / first
# -----------------------------------------------------------------------------


class TestHead:
    def test_head_default_returns_dict(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy("a")
        row = df.head()
        assert isinstance(row, dict)
        assert row == {"a": 1, "b": "x"}

    def test_head_one_returns_list_of_one(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy("a")
        rows = df.head(1)
        assert isinstance(rows, list)
        assert rows == [{"a": 1, "b": "x"}]

    def test_head_many_returns_list(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy("a")
        rows = df.head(5)
        assert isinstance(rows, list)
        assert rows == [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "z"},
            {"a": 3, "b": "y"},
            {"a": 4, "b": "x"},
        ]

    def test_head_zero_returns_empty_list(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA)
        assert df.head(0) == []

    def test_head_default_on_empty_returns_none(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).filter(F.col("a") > F.lit(999))
        assert df.head() is None


class TestTake:
    def test_take_returns_list_of_dicts(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy("a")
        rows = df.take(2)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)
        assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "z"}]

    def test_take_more_than_rowcount(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy("a")
        rows = df.take(100)
        assert len(rows) == len(_ORDER_DATA)


class TestFirst:
    def test_first_returns_dict(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).orderBy("a")
        row = df.first()
        assert isinstance(row, dict)
        assert row == {"a": 1, "b": "x"}

    def test_first_on_empty_returns_none(self, session, F, spark):
        df = session.createDataFrame(_ORDER_DATA).filter(F.col("a") > F.lit(999))
        assert df.first() is None
