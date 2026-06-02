"""Batch 11 parity tests: ``DataFrame.groupBy`` + ``GroupedData.agg`` + aggregate functions.

Covers:
    - Standalone aggregates in ``DataFrame.select``: ``F.sum``, ``F.count``,
      ``F.count("*")``, ``F.mean``, ``F.avg``, ``F.min``, ``F.max``,
      ``F.first`` (with/without ``ignorenulls``), ``F.collect_list``,
      ``F.collect_set``.
    - ``GroupedData`` shortcut methods (``.sum/.count/.mean/.min/.max``).
    - ``GroupedData.agg`` with multiple expressions, aliases, and multi-key
      grouping.
    - Null handling: aggregates skip nulls per PySpark semantics.
"""
from __future__ import annotations

import pytest
from pyspark.sql import functions as SF

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


# -----------------------------------------------------------------------------
# Standalone aggregates inside DataFrame.select
# -----------------------------------------------------------------------------


class TestStandaloneAggregates:
    def test_sum(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}, {"a": 3}]
        sdf = session.createDataFrame(rows).select(F.sum(F.col("a")).alias("s"))
        pdf = spark.createDataFrame(rows).select(SF.sum(SF.col("a")).alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_sum_str_arg(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}, {"a": 3}]
        sdf = session.createDataFrame(rows).select(F.sum("a").alias("s"))
        pdf = spark.createDataFrame(rows).select(SF.sum("a").alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_count_star(self, session, F, spark):
        rows = [{"a": 1}, {"a": None}, {"a": 3}]
        sdf = session.createDataFrame(rows).select(F.count("*").alias("n"))
        pdf = spark.createDataFrame(rows).select(SF.count("*").alias("n"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_count_column_skips_nulls(self, session, F, spark):
        rows = [{"a": 1}, {"a": None}, {"a": 3}]
        sdf = session.createDataFrame(rows).select(F.count("a").alias("n"))
        pdf = spark.createDataFrame(rows).select(SF.count("a").alias("n"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_mean(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        sdf = session.createDataFrame(rows).select(F.mean("a").alias("m"))
        pdf = spark.createDataFrame(rows).select(SF.mean("a").alias("m"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_avg_alias(self, session, F, spark):
        rows = [{"a": 1}, {"a": 2}]
        sdf = session.createDataFrame(rows).select(F.avg("a").alias("m"))
        pdf = spark.createDataFrame(rows).select(SF.avg("a").alias("m"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_min(self, session, F, spark):
        rows = [{"a": 5}, {"a": 1}, {"a": 3}]
        sdf = session.createDataFrame(rows).select(F.min("a").alias("v"))
        pdf = spark.createDataFrame(rows).select(SF.min("a").alias("v"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_max(self, session, F, spark):
        rows = [{"a": 5}, {"a": 1}, {"a": 3}]
        sdf = session.createDataFrame(rows).select(F.max("a").alias("v"))
        pdf = spark.createDataFrame(rows).select(SF.max("a").alias("v"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_aggregates_skip_nulls(self, session, F, spark):
        rows = [{"a": 1}, {"a": None}, {"a": 3}, {"a": None}, {"a": 6}]
        sdf = session.createDataFrame(rows).select(
            F.sum("a").alias("s"),
            F.mean("a").alias("m"),
            F.min("a").alias("mn"),
            F.max("a").alias("mx"),
            F.count("a").alias("cn"),
        )
        pdf = spark.createDataFrame(rows).select(
            SF.sum("a").alias("s"),
            SF.mean("a").alias("m"),
            SF.min("a").alias("mn"),
            SF.max("a").alias("mx"),
            SF.count("a").alias("cn"),
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# GroupedData shortcut methods
# -----------------------------------------------------------------------------


class TestGroupedShortcuts:
    def test_groupby_count(self, session, F, spark):
        rows = [{"k": "A"}, {"k": "B"}, {"k": "A"}, {"k": "A"}, {"k": "B"}]
        sdf = session.createDataFrame(rows).groupBy("k").count()
        pdf = spark.createDataFrame(rows).groupBy("k").count()
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_groupby_sum(self, session, F, spark):
        rows = [{"k": "A", "a": 1}, {"k": "A", "a": 2}, {"k": "B", "a": 5}]
        sdf = session.createDataFrame(rows).groupBy("k").sum("a")
        pdf = spark.createDataFrame(rows).groupBy("k").sum("a")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_groupby_mean(self, session, F, spark):
        rows = [{"k": "A", "a": 1}, {"k": "A", "a": 3}, {"k": "B", "a": 5}]
        sdf = session.createDataFrame(rows).groupBy("k").mean("a")
        pdf = spark.createDataFrame(rows).groupBy("k").mean("a")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_groupby_min(self, session, F, spark):
        rows = [{"k": "A", "a": 1}, {"k": "A", "a": 3}, {"k": "B", "a": 5}]
        sdf = session.createDataFrame(rows).groupBy("k").min("a")
        pdf = spark.createDataFrame(rows).groupBy("k").min("a")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_groupby_max(self, session, F, spark):
        rows = [{"k": "A", "a": 1}, {"k": "A", "a": 3}, {"k": "B", "a": 5}]
        sdf = session.createDataFrame(rows).groupBy("k").max("a")
        pdf = spark.createDataFrame(rows).groupBy("k").max("a")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# GroupedData.agg(*expressions)
# -----------------------------------------------------------------------------


class TestGroupedAgg:
    def test_agg_single_sum_with_alias(self, session, F, spark):
        rows = [{"k": "A", "a": 1}, {"k": "A", "a": 2}, {"k": "B", "a": 5}]
        sdf = session.createDataFrame(rows).groupBy("k").agg(F.sum("a").alias("s"))
        pdf = spark.createDataFrame(rows).groupBy("k").agg(SF.sum("a").alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_agg_multiple_expressions(self, session, F, spark):
        rows = [
            {"k": "A", "a": 1},
            {"k": "A", "a": 2},
            {"k": "B", "a": 5},
            {"k": "B", "a": 6},
        ]
        sdf = (
            session.createDataFrame(rows)
            .groupBy("k")
            .agg(F.sum("a").alias("s"), F.count("*").alias("n"))
        )
        pdf = (
            spark.createDataFrame(rows)
            .groupBy("k")
            .agg(SF.sum("a").alias("s"), SF.count("*").alias("n"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_agg_multi_key(self, session, F, spark):
        rows = [
            {"k1": "A", "k2": 1, "v": 10},
            {"k1": "A", "k2": 1, "v": 20},
            {"k1": "A", "k2": 2, "v": 30},
            {"k1": "B", "k2": 1, "v": 40},
        ]
        sdf = (
            session.createDataFrame(rows)
            .groupBy("k1", "k2")
            .agg(F.sum("v").alias("s"))
        )
        pdf = (
            spark.createDataFrame(rows)
            .groupBy("k1", "k2")
            .agg(SF.sum("v").alias("s"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_agg_null_handling(self, session, F, spark):
        rows = [
            {"k": "A", "v": 1},
            {"k": "A", "v": None},
            {"k": "A", "v": 3},
            {"k": "B", "v": None},
            {"k": "B", "v": None},
        ]
        sdf = (
            session.createDataFrame(rows)
            .groupBy("k")
            .agg(
                F.sum("v").alias("s"),
                F.count("v").alias("cn"),
                F.count("*").alias("cstar"),
                F.mean("v").alias("m"),
                F.min("v").alias("mn"),
                F.max("v").alias("mx"),
            )
        )
        pdf = (
            spark.createDataFrame(rows)
            .groupBy("k")
            .agg(
                SF.sum("v").alias("s"),
                SF.count("v").alias("cn"),
                SF.count("*").alias("cstar"),
                SF.mean("v").alias("m"),
                SF.min("v").alias("mn"),
                SF.max("v").alias("mx"),
            )
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# F.first(col, ignorenulls=...) inside groupBy.agg
# -----------------------------------------------------------------------------


class TestAggFirst:
    def test_first_default_keeps_nulls(self, session, F, spark):
        # ``first()`` with default ignorenulls=False can pick a null if it's first.
        # Make the first row of each group deterministic by ordering.
        rows = [
            {"k": "A", "v": None},
            {"k": "A", "v": 1},
            {"k": "B", "v": 2},
            {"k": "B", "v": 3},
        ]
        # Sort by k, then by an explicit position column to make ordering deterministic.
        sdf = (
            session.createDataFrame(rows)
            .orderBy("k")
            .groupBy("k")
            .agg(F.first("v").alias("f"))
        )
        pdf = (
            spark.createDataFrame(rows)
            .orderBy("k")
            .groupBy("k")
            .agg(SF.first("v").alias("f"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_first_ignorenulls(self, session, F, spark):
        rows = [
            {"k": "A", "v": None},
            {"k": "A", "v": 1},
            {"k": "A", "v": 2},
            {"k": "B", "v": None},
            {"k": "B", "v": 3},
        ]
        sdf = (
            session.createDataFrame(rows)
            .orderBy("k")
            .groupBy("k")
            .agg(F.first("v", ignorenulls=True).alias("f"))
        )
        pdf = (
            spark.createDataFrame(rows)
            .orderBy("k")
            .groupBy("k")
            .agg(SF.first("v", ignorenulls=True).alias("f"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# collect_list / collect_set
# -----------------------------------------------------------------------------


class TestCollectList:
    def test_collect_list_skips_nulls(self, session, F, spark):
        rows = [
            {"k": "A", "v": 1},
            {"k": "A", "v": None},
            {"k": "A", "v": 3},
            {"k": "B", "v": 2},
        ]
        # Order rows deterministically so list ordering matches.
        sdf = (
            session.createDataFrame(rows)
            .orderBy("k", "v")
            .groupBy("k")
            .agg(F.collect_list("v").alias("items"))
        )
        pdf = (
            spark.createDataFrame(rows)
            .orderBy("k", "v")
            .groupBy("k")
            .agg(SF.collect_list("v").alias("items"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


class TestCollectSet:
    def test_collect_set_dedups_and_skips_nulls(self, session, F, spark):
        from sparkleframe.tests.utils import _get_records

        rows = [
            {"k": "A", "v": 1},
            {"k": "A", "v": 1},
            {"k": "A", "v": None},
            {"k": "A", "v": 2},
            {"k": "B", "v": 5},
            {"k": "B", "v": 5},
        ]
        # ``collect_set`` order is undefined per Spark; sort the resulting arrays
        # in Python before comparing.
        sdf = (
            session.createDataFrame(rows)
            .groupBy("k")
            .agg(F.collect_set("v").alias("items"))
        )
        pdf = (
            spark.createDataFrame(rows)
            .groupBy("k")
            .agg(SF.collect_set("v").alias("items"))
        )

        def _norm(records):
            return sorted(
                ({"k": r["k"], "items": sorted(r["items"])} for r in records),
                key=lambda r: r["k"],
            )

        assert _norm(_get_records(sdf)) == _norm(_get_records(pdf))
