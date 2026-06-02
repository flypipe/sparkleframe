"""Batch 14 parity tests: aggregate function edge cases.

Complements ``test_groupby_agg.py`` (B11) by covering type-level and edge-case
behaviour NOT covered there:

    - Type promotion: ``F.sum(int)`` -> Long, ``F.sum(float)`` -> Double,
      ``F.mean`` -> Double, ``F.count`` -> Long (verified via the value path,
      which uses Spark's promoted types as the source of truth).
    - Empty input: aggregates on a 0-row DataFrame — ``count`` returns 0;
      ``sum/mean/min/max/first`` return NULL; ``collect_list/collect_set``
      return empty arrays.
    - All-null column: aggregates skip nulls; ``count("col")`` returns 0;
      ``collect_list/collect_set`` return empty arrays.
    - ``F.first(ignorenulls=True)`` skipping a leading null in select context.
    - ``F.count("col_name")`` vs ``F.count("*")`` distinction (non-null count
      vs row count) at the DataFrame.select level.
    - Aggregate + ``.alias()`` produces the correctly-named column.
    - Float / decimal-string sums and means.

Every test runs against both backends (``polarsdf`` and ``pythondf``) and is
asserted against native PySpark via :func:`assert_sparkle_spark_frame_are_equal`.
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


# -----------------------------------------------------------------------------
# Type promotion — verified via parity assertion (Spark types drive the
# inferred record types).
# -----------------------------------------------------------------------------


class TestAggReturnTypes:
    def test_sum_int_returns_long(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}, {"a": 3}]
        sdf = session.createDataFrame(rows).select(F.sum("a").alias("s"))
        pdf = spark.createDataFrame(rows).select(SF.sum("a").alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_sum_float_returns_double(self, session, F, spark):
        rows = [{"a": 1.5}, {"a": 2.25}, {"a": 0.25}]
        sdf = session.createDataFrame(rows).select(F.sum("a").alias("s"))
        pdf = spark.createDataFrame(rows).select(SF.sum("a").alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_mean_int_returns_double(self, session, F, spark):
        # mean over [1,2] is 1.5 — Spark mean always returns DoubleType.
        rows = [{"a": 1}, {"a": 2}]
        sdf = session.createDataFrame(rows).select(F.mean("a").alias("m"))
        pdf = spark.createDataFrame(rows).select(SF.mean("a").alias("m"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_mean_returns_double_even_for_whole_result(self, session, F, spark):
        # mean over [2, 4] is exactly 3.0 — still DoubleType in Spark.
        rows = [{"a": 2}, {"a": 4}]
        sdf = session.createDataFrame(rows).select(F.mean("a").alias("m"))
        pdf = spark.createDataFrame(rows).select(SF.mean("a").alias("m"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_count_returns_long(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}, {"a": None}]
        sdf = session.createDataFrame(rows).select(F.count("a").alias("c"))
        pdf = spark.createDataFrame(rows).select(SF.count("a").alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_count_star_returns_long(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}, {"a": None}]
        sdf = session.createDataFrame(rows).select(F.count("*").alias("c"))
        pdf = spark.createDataFrame(rows).select(SF.count("*").alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# Empty-input aggregates
# -----------------------------------------------------------------------------


_EMPTY_INT_SCHEMA = SparkStructType(
    [SparkStructField("a", SparkIntegerType(), nullable=True)]
)


class TestEmptyInputAggregates:
    """Aggregates over a 0-row DataFrame.

    PySpark guarantees one result row for every global aggregation:
        - count -> 0 (Long, non-null)
        - sum / mean / min / max / first -> NULL
        - collect_list / collect_set -> [] (empty array)
    """

    def test_empty_count_column_is_zero(self, session, F, spark):
        sdf = session.createDataFrame([], schema="a INT").select(F.count("a").alias("c"))
        pdf = spark.createDataFrame([], schema=_EMPTY_INT_SCHEMA).select(SF.count("a").alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_count_star_is_zero(self, session, F, spark):
        sdf = session.createDataFrame([], schema="a INT").select(F.count("*").alias("c"))
        pdf = spark.createDataFrame([], schema=_EMPTY_INT_SCHEMA).select(SF.count("*").alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_sum_is_null(self, session, F, spark):
        sdf = session.createDataFrame([], schema="a INT").select(F.sum("a").alias("s"))
        pdf = spark.createDataFrame([], schema=_EMPTY_INT_SCHEMA).select(SF.sum("a").alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_mean_is_null(self, session, F, spark):
        sdf = session.createDataFrame([], schema="a INT").select(F.mean("a").alias("m"))
        pdf = spark.createDataFrame([], schema=_EMPTY_INT_SCHEMA).select(SF.mean("a").alias("m"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_min_is_null(self, session, F, spark):
        sdf = session.createDataFrame([], schema="a INT").select(F.min("a").alias("mn"))
        pdf = spark.createDataFrame([], schema=_EMPTY_INT_SCHEMA).select(SF.min("a").alias("mn"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_max_is_null(self, session, F, spark):
        sdf = session.createDataFrame([], schema="a INT").select(F.max("a").alias("mx"))
        pdf = spark.createDataFrame([], schema=_EMPTY_INT_SCHEMA).select(SF.max("a").alias("mx"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_first_is_null(self, session, F, spark):
        sdf = session.createDataFrame([], schema="a INT").select(F.first("a").alias("f"))
        pdf = spark.createDataFrame([], schema=_EMPTY_INT_SCHEMA).select(SF.first("a").alias("f"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_collect_list_is_empty_array(self, session, F, spark):
        sdf = session.createDataFrame([], schema="a INT").select(F.collect_list("a").alias("cl"))
        pdf = spark.createDataFrame([], schema=_EMPTY_INT_SCHEMA).select(SF.collect_list("a").alias("cl"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_collect_set_is_empty_array(self, session, F, spark):
        sdf = session.createDataFrame([], schema="a INT").select(F.collect_set("a").alias("cs"))
        pdf = spark.createDataFrame([], schema=_EMPTY_INT_SCHEMA).select(SF.collect_set("a").alias("cs"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# All-null column aggregates
# -----------------------------------------------------------------------------


class TestAllNullColumnAggregates:
    """Aggregates where every value in the column is NULL.

    PySpark contract:
        - count("col") -> 0 (skips nulls)
        - count("*") -> N (counts rows, not values)
        - sum / mean / min / max / first(default) -> NULL
        - collect_list / collect_set -> [] (drop nulls)
    """

    @staticmethod
    def _all_null():
        return [{"a": None}, {"a": None}, {"a": None}]

    def test_count_column_skips_all_nulls(self, session, F, spark):
        rows = self._all_null()
        sdf = session.createDataFrame(rows, schema="a INT").select(F.count("a").alias("c"))
        pdf = spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA).select(SF.count("a").alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_count_star_counts_rows_with_all_nulls(self, session, F, spark):
        rows = self._all_null()
        sdf = session.createDataFrame(rows, schema="a INT").select(F.count("*").alias("c"))
        pdf = spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA).select(SF.count("*").alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_sum_all_null_is_null(self, session, F, spark):
        rows = self._all_null()
        sdf = session.createDataFrame(rows, schema="a INT").select(F.sum("a").alias("s"))
        pdf = spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA).select(SF.sum("a").alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_mean_all_null_is_null(self, session, F, spark):
        rows = self._all_null()
        sdf = session.createDataFrame(rows, schema="a INT").select(F.mean("a").alias("m"))
        pdf = spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA).select(SF.mean("a").alias("m"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_min_all_null_is_null(self, session, F, spark):
        rows = self._all_null()
        sdf = session.createDataFrame(rows, schema="a INT").select(F.min("a").alias("mn"))
        pdf = spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA).select(SF.min("a").alias("mn"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_max_all_null_is_null(self, session, F, spark):
        rows = self._all_null()
        sdf = session.createDataFrame(rows, schema="a INT").select(F.max("a").alias("mx"))
        pdf = spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA).select(SF.max("a").alias("mx"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_collect_list_all_null_is_empty(self, session, F, spark):
        rows = self._all_null()
        sdf = session.createDataFrame(rows, schema="a INT").select(F.collect_list("a").alias("cl"))
        pdf = spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA).select(SF.collect_list("a").alias("cl"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_collect_set_all_null_is_empty(self, session, F, spark):
        rows = self._all_null()
        sdf = session.createDataFrame(rows, schema="a INT").select(F.collect_set("a").alias("cs"))
        pdf = spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA).select(SF.collect_set("a").alias("cs"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# F.first ignorenulls behaviour in select (global agg)
# -----------------------------------------------------------------------------


class TestFirstIgnoreNulls:
    def test_first_ignorenulls_skips_leading_null_in_select(self, session, F, spark):
        rows = [{"a": None}, {"a": None}, {"a": 7}, {"a": 8}]
        sdf = (
            session.createDataFrame(rows, schema="a INT")
            .select(F.first("a", ignorenulls=True).alias("f"))
        )
        pdf = (
            spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA)
            .select(SF.first("a", ignorenulls=True).alias("f"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_first_default_takes_leading_null_in_select(self, session, F, spark):
        rows = [{"a": None}, {"a": 7}, {"a": 8}]
        sdf = (
            session.createDataFrame(rows, schema="a INT")
            .select(F.first("a").alias("f"))
        )
        pdf = (
            spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA)
            .select(SF.first("a").alias("f"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_first_ignorenulls_all_null_returns_null(self, session, F, spark):
        rows = [{"a": None}, {"a": None}, {"a": None}]
        sdf = (
            session.createDataFrame(rows, schema="a INT")
            .select(F.first("a", ignorenulls=True).alias("f"))
        )
        pdf = (
            spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA)
            .select(SF.first("a", ignorenulls=True).alias("f"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# F.count("col") vs F.count("*") distinction in select
# -----------------------------------------------------------------------------


class TestCountColumnVsStar:
    def test_count_column_skips_nulls_select(self, session, F, spark):
        rows = [{"a": 1}, {"a": None}, {"a": 3}, {"a": None}]
        sdf = session.createDataFrame(rows, schema="a INT").select(
            F.count("a").alias("non_null"),
            F.count("*").alias("rows"),
        )
        pdf = spark.createDataFrame(rows, schema=_EMPTY_INT_SCHEMA).select(
            SF.count("a").alias("non_null"),
            SF.count("*").alias("rows"),
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_count_no_nulls_equals_count_star(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}, {"a": 3}]
        sdf = session.createDataFrame(rows).select(
            F.count("a").alias("non_null"),
            F.count("*").alias("rows"),
        )
        pdf = spark.createDataFrame(rows).select(
            SF.count("a").alias("non_null"),
            SF.count("*").alias("rows"),
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# Aliasing produces the correct output column name
# -----------------------------------------------------------------------------


class TestAggAliasing:
    def test_sum_alias_names_column(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}, {"a": 3}]
        sdf = session.createDataFrame(rows).select(F.sum("a").alias("total"))
        pdf = spark.createDataFrame(rows).select(SF.sum("a").alias("total"))
        assert "total" in sdf.columns
        assert "total" in pdf.columns
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_multiple_aggs_distinct_aliases(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        sdf = session.createDataFrame(rows).select(
            F.sum("a").alias("total"),
            F.mean("a").alias("avg_val"),
            F.min("a").alias("low"),
            F.max("a").alias("high"),
            F.count("a").alias("n"),
        )
        pdf = spark.createDataFrame(rows).select(
            SF.sum("a").alias("total"),
            SF.mean("a").alias("avg_val"),
            SF.min("a").alias("low"),
            SF.max("a").alias("high"),
            SF.count("a").alias("n"),
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# Single-type collect_list shape parity
# -----------------------------------------------------------------------------


class TestCollectListShape:
    def test_collect_list_strings(self, session, F, spark):
        rows = [{"k": "g", "v": "a"}, {"k": "g", "v": "b"}, {"k": "g", "v": "c"}]
        sdf = (
            session.createDataFrame(rows)
            .orderBy("v")
            .groupBy("k")
            .agg(F.collect_list("v").alias("items"))
        )
        pdf = (
            spark.createDataFrame(rows)
            .orderBy("v")
            .groupBy("k")
            .agg(SF.collect_list("v").alias("items"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_collect_list_floats(self, session, F, spark):
        rows = [{"k": "g", "v": 1.5}, {"k": "g", "v": 2.5}, {"k": "g", "v": 3.5}]
        sdf = (
            session.createDataFrame(rows)
            .orderBy("v")
            .groupBy("k")
            .agg(F.collect_list("v").alias("items"))
        )
        pdf = (
            spark.createDataFrame(rows)
            .orderBy("v")
            .groupBy("k")
            .agg(SF.collect_list("v").alias("items"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# Float-input aggregates
# -----------------------------------------------------------------------------


class TestFloatAggregates:
    def test_mean_floats(self, session, F, spark):
        rows = [{"a": 1.5}, {"a": 2.5}, {"a": 3.5}]
        sdf = session.createDataFrame(rows).select(F.mean("a").alias("m"))
        pdf = spark.createDataFrame(rows).select(SF.mean("a").alias("m"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_min_max_floats(self, session, F, spark):
        rows = [{"a": 3.5}, {"a": 1.25}, {"a": 9.0}, {"a": 7.5}]
        sdf = session.createDataFrame(rows).select(
            F.min("a").alias("mn"),
            F.max("a").alias("mx"),
        )
        pdf = spark.createDataFrame(rows).select(
            SF.min("a").alias("mn"),
            SF.max("a").alias("mx"),
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_sum_floats_with_nulls(self, session, F, spark):
        rows = [{"a": 1.5}, {"a": None}, {"a": 2.5}, {"a": None}]
        schema = SparkStructType(
            [SparkStructField("a", SparkDoubleType(), nullable=True)]
        )
        sdf = session.createDataFrame(rows, schema="a DOUBLE").select(
            F.sum("a").alias("s")
        )
        pdf = spark.createDataFrame(rows, schema=schema).select(SF.sum("a").alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# String aggregates: min/max are valid on strings in Spark.
# -----------------------------------------------------------------------------


class TestStringMinMax:
    def test_min_string(self, session, F, spark):
        rows = [{"s": "banana"}, {"s": "apple"}, {"s": "cherry"}]
        sdf = session.createDataFrame(rows).select(F.min("s").alias("mn"))
        pdf = spark.createDataFrame(rows).select(SF.min("s").alias("mn"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_max_string(self, session, F, spark):
        rows = [{"s": "banana"}, {"s": "apple"}, {"s": "cherry"}]
        sdf = session.createDataFrame(rows).select(F.max("s").alias("mx"))
        pdf = spark.createDataFrame(rows).select(SF.max("s").alias("mx"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)
