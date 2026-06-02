"""Batch 12 parity tests: ``DataFrame.join`` across every PySpark how variant.

Covers:
    - ``inner`` with single str key, list[str] key, multi-key.
    - ``left`` / ``left_outer`` / ``leftouter`` aliases.
    - ``right`` / ``right_outer`` / ``rightouter`` aliases.
    - ``outer`` / ``full`` / ``fullouter`` / ``full_outer`` aliases.
    - ``cross`` cartesian product.
    - ``leftsemi`` / ``semi`` / ``left_semi`` aliases — only left columns.
    - ``leftanti`` / ``anti`` / ``left_anti`` aliases — only left columns.
    - Predicate (``Column``) joins raise NotImplementedError on pythondf (xfailed
      on polarsdf as a no-op-not-supported in this batch).
"""
from __future__ import annotations

import pytest
from pyspark.sql import functions as SF

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


# ---------------------------------------------------------------------------
# Inner join
# ---------------------------------------------------------------------------


class TestInnerJoin:
    def test_inner_single_str_key(self, session, F, spark):
        left = [{"id": 1, "left_val": "a"}, {"id": 2, "left_val": "b"}, {"id": 3, "left_val": "c"}]
        right = [{"id": 2, "right_val": "x"}, {"id": 3, "right_val": "y"}, {"id": 4, "right_val": "z"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on="id", how="inner")
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on="id", how="inner")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_inner_list_single_key(self, session, F, spark):
        left = [{"id": 1, "left_val": "a"}, {"id": 2, "left_val": "b"}]
        right = [{"id": 1, "right_val": "x"}, {"id": 2, "right_val": "y"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on=["id"], how="inner")
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on=["id"], how="inner")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_inner_multi_key(self, session, F, spark):
        left = [
            {"k1": 1, "k2": "a", "lv": 10},
            {"k1": 1, "k2": "b", "lv": 20},
            {"k1": 2, "k2": "a", "lv": 30},
        ]
        right = [
            {"k1": 1, "k2": "a", "rv": 100},
            {"k1": 2, "k2": "a", "rv": 200},
            {"k1": 2, "k2": "b", "rv": 300},
        ]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on=["k1", "k2"], how="inner")
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on=["k1", "k2"], how="inner")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_inner_default_how(self, session, F, spark):
        left = [{"id": 1, "lv": "a"}, {"id": 2, "lv": "b"}]
        right = [{"id": 2, "rv": "x"}, {"id": 3, "rv": "y"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on="id")
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on="id")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# Left / right / outer
# ---------------------------------------------------------------------------


class TestLeftJoin:
    @pytest.mark.parametrize("how", ["left", "leftouter", "left_outer"])
    def test_left_alias(self, session, F, spark, how):
        left = [{"id": 1, "lv": "a"}, {"id": 2, "lv": "b"}, {"id": 3, "lv": "c"}]
        right = [{"id": 2, "rv": "x"}, {"id": 3, "rv": "y"}, {"id": 4, "rv": "z"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on="id", how=how)
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on="id", how=how)
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


class TestRightJoin:
    @pytest.mark.parametrize("how", ["right", "rightouter", "right_outer"])
    def test_right_alias(self, session, F, spark, how):
        left = [{"id": 1, "lv": "a"}, {"id": 2, "lv": "b"}, {"id": 3, "lv": "c"}]
        right = [{"id": 2, "rv": "x"}, {"id": 3, "rv": "y"}, {"id": 4, "rv": "z"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on="id", how=how)
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on="id", how=how)
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


class TestOuterJoin:
    @pytest.mark.parametrize("how", ["outer", "full", "fullouter", "full_outer"])
    def test_outer_alias(self, session, F, spark, how):
        left = [{"id": 1, "lv": "a"}, {"id": 2, "lv": "b"}, {"id": 3, "lv": "c"}]
        right = [{"id": 2, "rv": "x"}, {"id": 3, "rv": "y"}, {"id": 4, "rv": "z"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on="id", how=how)
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on="id", how=how)
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# Cross join
# ---------------------------------------------------------------------------


class TestCrossJoin:
    def test_cross_small_frames(self, session, F, spark):
        left = [{"a": 1}, {"a": 2}, {"a": 3}]
        right = [{"b": "x"}, {"b": "y"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), how="cross")
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), how="cross")
        # Size = m * n.
        assert sdf.count() == 3 * 2
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_cross_single_rows(self, session, F, spark):
        left = [{"a": 1}]
        right = [{"b": 10}, {"b": 20}, {"b": 30}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), how="cross")
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), how="cross")
        assert sdf.count() == 3
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# Semi / anti
# ---------------------------------------------------------------------------


class TestLeftSemiJoin:
    @pytest.mark.parametrize("how", ["semi", "leftsemi", "left_semi"])
    def test_leftsemi_alias(self, session, F, spark, how):
        left = [{"id": 1, "lv": "a"}, {"id": 2, "lv": "b"}, {"id": 3, "lv": "c"}]
        right = [{"id": 2, "rv": "x"}, {"id": 3, "rv": "y"}, {"id": 4, "rv": "z"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on="id", how=how)
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on="id", how=how)
        # Only left columns survive.
        assert sorted(sdf.columns) == ["id", "lv"]
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_leftsemi_dedupes_left(self, session, F, spark):
        # If left has dup keys and right has any match, all matching left rows survive.
        left = [{"id": 1, "lv": "a"}, {"id": 1, "lv": "b"}, {"id": 2, "lv": "c"}]
        right = [{"id": 1, "rv": "x"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on="id", how="leftsemi")
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on="id", how="leftsemi")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


class TestLeftAntiJoin:
    @pytest.mark.parametrize("how", ["anti", "leftanti", "left_anti"])
    def test_leftanti_alias(self, session, F, spark, how):
        left = [{"id": 1, "lv": "a"}, {"id": 2, "lv": "b"}, {"id": 3, "lv": "c"}]
        right = [{"id": 2, "rv": "x"}, {"id": 3, "rv": "y"}, {"id": 4, "rv": "z"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on="id", how=how)
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on="id", how=how)
        assert sorted(sdf.columns) == ["id", "lv"]
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_leftanti_all_unmatched(self, session, F, spark):
        left = [{"id": 1, "lv": "a"}, {"id": 2, "lv": "b"}]
        right = [{"id": 10, "rv": "x"}]
        sdf = session.createDataFrame(left).join(session.createDataFrame(right), on="id", how="leftanti")
        pdf = spark.createDataFrame(left).join(spark.createDataFrame(right), on="id", how="leftanti")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# Predicate (Column) join — pythondf raises; polarsdf is out of scope here.
# ---------------------------------------------------------------------------


class TestPredicateJoin:
    def test_predicate_join_raises_on_pythondf(self, session, F, backend, spark):
        if backend != "pythondf":
            pytest.skip("predicate join coverage is pythondf-only in this batch")
        left = session.createDataFrame([{"a": 1}])
        right = session.createDataFrame([{"b": 1}])
        with pytest.raises(NotImplementedError):
            left.join(right, on=(F.col("a") == F.col("b")), how="inner")


# ---------------------------------------------------------------------------
# Misc / error paths
# ---------------------------------------------------------------------------


class TestJoinErrors:
    def test_unsupported_how_raises(self, session, F, spark):
        left = session.createDataFrame([{"id": 1}])
        right = session.createDataFrame([{"id": 1}])
        with pytest.raises((ValueError, Exception)):
            left.join(right, on="id", how="not_a_real_join")

    def test_mixed_on_types_raise(self, session, F, spark):
        left = session.createDataFrame([{"id": 1}])
        right = session.createDataFrame([{"id": 1}])
        with pytest.raises(TypeError):
            left.join(right, on=["id", F.col("id")], how="inner")
