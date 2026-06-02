"""Batch 22 parity tests: Window functions.

Covers:
    - :class:`Window` builder + :class:`WindowSpec` API surface (no DataFrame).
    - Ranking functions (``rank``, ``dense_rank``, ``row_number``) over a
      partitioned/ordered window.
    - Aggregate functions (``sum``, ``count``, ``mean``, ``min``, ``max``)
      applied via ``.over(window)``:
        * Whole-partition reduction (partitionBy only).
        * Running reduction (partitionBy + orderBy).
        * Explicit ``rowsBetween`` frames.

Ranking and running-aggregate tests use the position-sensitive
``assert_frame_ordered_equal`` after an explicit ``orderBy`` to align the
sparkle/PySpark row order. Whole-partition tests use
``assert_sparkle_spark_frame_are_equal`` (set-equal).
"""
from __future__ import annotations

import importlib

import pytest
import pyspark.sql.functions as SF
from pyspark.sql.window import Window as SparkWindow

from sparkleframe.tests.utils import (
    assert_frame_ordered_equal,
    assert_sparkle_spark_frame_are_equal,
)


# -----------------------------------------------------------------------------
# WindowSpec API — pure backend tests (no DataFrame)
# -----------------------------------------------------------------------------


@pytest.fixture
def W(backend: str):  # noqa: N802 — short alias mirroring PySpark conventions
    """Backend's :class:`Window` class."""
    return importlib.import_module(f"sparkleframe.{backend}.window").Window


@pytest.fixture
def WSpec(backend: str):  # noqa: N802
    """Backend's :class:`WindowSpec` class."""
    return importlib.import_module(f"sparkleframe.{backend}.window").WindowSpec


class TestWindowSpec:
    @pytest.mark.parametrize(
        "partition_cols",
        [
            ["a"],
            ["x", "y"],
            [],
        ],
    )
    def test_partition_by(self, W, WSpec, partition_cols):
        spec = W.partitionBy(*partition_cols)
        assert isinstance(spec, WSpec)
        assert spec.partition_cols == partition_cols

    @pytest.mark.parametrize(
        "order_cols",
        [
            ["a"],
            ["x", "y"],
        ],
    )
    def test_order_by(self, W, WSpec, order_cols):
        spec = W.orderBy(*order_cols)
        assert isinstance(spec, WSpec)
        assert spec.order_cols == order_cols

    def test_partition_and_order_by(self, W):
        spec = W.partitionBy("group").orderBy("value")
        assert spec.partition_cols == ["group"]
        assert spec.order_cols == ["value"]

    @pytest.mark.parametrize(
        "start, end",
        [
            (-1, 1),
            (0, 0),
            (-5, 0),
            (0, 10),
        ],
    )
    def test_range_between_valid(self, W, start, end):
        spec = W.orderBy("timestamp").rangeBetween(start, end)
        assert spec.frame_start == start
        assert spec.frame_end == end

    @pytest.mark.parametrize(
        "start, end",
        [
            ("-1", 1),
            (0, "5"),
            ("a", "b"),
        ],
    )
    def test_range_between_invalid_raises(self, W, start, end):
        with pytest.raises(TypeError):
            W.orderBy("timestamp").rangeBetween(start, end)

    def test_rows_between_sets_frame(self, W):
        spec = W.partitionBy("p").orderBy("a").rowsBetween(-1, 1)
        assert spec.frame_start == -1
        assert spec.frame_end == 1


# -----------------------------------------------------------------------------
# Ranking functions
# -----------------------------------------------------------------------------


_RANK_DATA = [
    {"p": "x", "a": 10},
    {"p": "x", "a": 20},
    {"p": "x", "a": 20},  # tie with the previous row
    {"p": "x", "a": 40},
    {"p": "y", "a": 5},
    {"p": "y", "a": 15},
]


class TestRanking:
    def test_row_number(self, session, F, spark):
        from sparkleframe.pythondf.window import Window as PyWindow  # noqa: F401  — silence linter; backend-agnostic test uses W fixture path below
        # Build via backend-agnostic import path through `session`.
        backend_window = _backend_window(session)
        sdf = (
            session.createDataFrame(_RANK_DATA)
            .withColumn(
                "rn",
                F.row_number().over(backend_window.partitionBy("p").orderBy("a")),
            )
            .orderBy("p", "a")
        )
        pdf = (
            spark.createDataFrame(_RANK_DATA)
            .withColumn(
                "rn",
                SF.row_number().over(SparkWindow.partitionBy("p").orderBy("a")),
            )
            .orderBy("p", "a")
        )
        assert_frame_ordered_equal(sdf, pdf)

    def test_rank_with_ties(self, session, F, spark):
        backend_window = _backend_window(session)
        sdf = (
            session.createDataFrame(_RANK_DATA)
            .withColumn(
                "r",
                F.rank().over(backend_window.partitionBy("p").orderBy("a")),
            )
            .orderBy("p", "a")
        )
        pdf = (
            spark.createDataFrame(_RANK_DATA)
            .withColumn(
                "r",
                SF.rank().over(SparkWindow.partitionBy("p").orderBy("a")),
            )
            .orderBy("p", "a")
        )
        assert_frame_ordered_equal(sdf, pdf)

    def test_dense_rank_with_ties(self, session, F, spark):
        backend_window = _backend_window(session)
        sdf = (
            session.createDataFrame(_RANK_DATA)
            .withColumn(
                "dr",
                F.dense_rank().over(backend_window.partitionBy("p").orderBy("a")),
            )
            .orderBy("p", "a")
        )
        pdf = (
            spark.createDataFrame(_RANK_DATA)
            .withColumn(
                "dr",
                SF.dense_rank().over(SparkWindow.partitionBy("p").orderBy("a")),
            )
            .orderBy("p", "a")
        )
        assert_frame_ordered_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# Aggregate over window
# -----------------------------------------------------------------------------


_AGG_DATA = [
    {"p": "x", "a": 1, "v": 10},
    {"p": "x", "a": 2, "v": 20},
    {"p": "x", "a": 3, "v": 30},
    {"p": "x", "a": 4, "v": 40},
    {"p": "y", "a": 1, "v": 100},
    {"p": "y", "a": 2, "v": 200},
]


class TestAggregateOverWindow:
    def test_sum_partition_only_whole_partition(self, session, F, spark):
        """``sum(col).over(Window.partitionBy("p"))`` reduces the whole partition."""
        backend_window = _backend_window(session)
        sdf = (
            session.createDataFrame(_AGG_DATA)
            .withColumn(
                "ps",
                F.sum("v").over(backend_window.partitionBy("p")),
            )
            .orderBy("p", "a")
        )
        pdf = (
            spark.createDataFrame(_AGG_DATA)
            .withColumn(
                "ps",
                SF.sum("v").over(SparkWindow.partitionBy("p")),
            )
            .orderBy("p", "a")
        )
        assert_frame_ordered_equal(sdf, pdf)

    def test_sum_running_with_order_by(self, session, F, spark):
        """``sum(col).over(partitionBy("p").orderBy("a"))`` produces a running sum."""
        backend_window = _backend_window(session)
        sdf = (
            session.createDataFrame(_AGG_DATA)
            .withColumn(
                "rs",
                F.sum("v").over(backend_window.partitionBy("p").orderBy("a")),
            )
            .orderBy("p", "a")
        )
        pdf = (
            spark.createDataFrame(_AGG_DATA)
            .withColumn(
                "rs",
                SF.sum("v").over(SparkWindow.partitionBy("p").orderBy("a")),
            )
            .orderBy("p", "a")
        )
        assert_frame_ordered_equal(sdf, pdf)

    def test_running_sum_with_rows_between_unbounded_preceding_to_current(
        self, session, F, spark
    ):
        backend_window = _backend_window(session)
        spec = (
            backend_window.partitionBy("p")
            .orderBy("a")
            .rowsBetween(backend_window.unboundedPreceding, backend_window.currentRow)
        )
        spark_spec = (
            SparkWindow.partitionBy("p")
            .orderBy("a")
            .rowsBetween(SparkWindow.unboundedPreceding, SparkWindow.currentRow)
        )
        sdf = (
            session.createDataFrame(_AGG_DATA)
            .withColumn("rs", F.sum("v").over(spec))
            .orderBy("p", "a")
        )
        pdf = (
            spark.createDataFrame(_AGG_DATA)
            .withColumn("rs", SF.sum("v").over(spark_spec))
            .orderBy("p", "a")
        )
        assert_frame_ordered_equal(sdf, pdf)

    def test_sliding_three_row_window(self, session, F, spark):
        """``rowsBetween(-1, 1)`` — three-row sliding window (prev, current, next)."""
        backend_window = _backend_window(session)
        spec = backend_window.partitionBy("p").orderBy("a").rowsBetween(-1, 1)
        spark_spec = SparkWindow.partitionBy("p").orderBy("a").rowsBetween(-1, 1)
        sdf = (
            session.createDataFrame(_AGG_DATA)
            .withColumn("slide", F.sum("v").over(spec))
            .orderBy("p", "a")
        )
        pdf = (
            spark.createDataFrame(_AGG_DATA)
            .withColumn("slide", SF.sum("v").over(spark_spec))
            .orderBy("p", "a")
        )
        assert_frame_ordered_equal(sdf, pdf)

    def test_count_over_partition(self, session, F, spark):
        backend_window = _backend_window(session)
        sdf = (
            session.createDataFrame(_AGG_DATA)
            .withColumn(
                "n",
                F.count("v").over(backend_window.partitionBy("p")),
            )
            .orderBy("p", "a")
        )
        pdf = (
            spark.createDataFrame(_AGG_DATA)
            .withColumn(
                "n",
                SF.count("v").over(SparkWindow.partitionBy("p")),
            )
            .orderBy("p", "a")
        )
        assert_frame_ordered_equal(sdf, pdf)

    def test_min_max_mean_over_partition(self, session, F, spark):
        backend_window = _backend_window(session)
        sdf = (
            session.createDataFrame(_AGG_DATA)
            .withColumn("mn", F.min("v").over(backend_window.partitionBy("p")))
            .withColumn("mx", F.max("v").over(backend_window.partitionBy("p")))
            .withColumn("mean", F.mean("v").over(backend_window.partitionBy("p")))
            .orderBy("p", "a")
        )
        pdf = (
            spark.createDataFrame(_AGG_DATA)
            .withColumn("mn", SF.min("v").over(SparkWindow.partitionBy("p")))
            .withColumn("mx", SF.max("v").over(SparkWindow.partitionBy("p")))
            .withColumn("mean", SF.mean("v").over(SparkWindow.partitionBy("p")))
            .orderBy("p", "a")
        )
        assert_frame_ordered_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# Multi-key partitionBy / orderBy
# -----------------------------------------------------------------------------


_MULTI_DATA = [
    {"p1": "x", "p2": "a", "k1": 1, "k2": 9, "v": 1},
    {"p1": "x", "p2": "a", "k1": 1, "k2": 8, "v": 2},
    {"p1": "x", "p2": "a", "k1": 2, "k2": 7, "v": 3},
    {"p1": "x", "p2": "b", "k1": 1, "k2": 1, "v": 4},
    {"p1": "y", "p2": "a", "k1": 5, "k2": 5, "v": 5},
    {"p1": "y", "p2": "a", "k1": 4, "k2": 4, "v": 6},
]


class TestMultiKeyWindows:
    def test_multi_partition_keys_sum(self, session, F, spark):
        backend_window = _backend_window(session)
        sdf = (
            session.createDataFrame(_MULTI_DATA)
            .withColumn(
                "ps",
                F.sum("v").over(backend_window.partitionBy("p1", "p2")),
            )
            .orderBy("p1", "p2", "k1", "k2")
        )
        pdf = (
            spark.createDataFrame(_MULTI_DATA)
            .withColumn(
                "ps",
                SF.sum("v").over(SparkWindow.partitionBy("p1", "p2")),
            )
            .orderBy("p1", "p2", "k1", "k2")
        )
        assert_frame_ordered_equal(sdf, pdf)

    def test_multi_order_keys_row_number(self, session, F, spark):
        backend_window = _backend_window(session)
        sdf = (
            session.createDataFrame(_MULTI_DATA)
            .withColumn(
                "rn",
                F.row_number().over(
                    backend_window.partitionBy("p1").orderBy("k1", "k2")
                ),
            )
            .orderBy("p1", "k1", "k2")
        )
        pdf = (
            spark.createDataFrame(_MULTI_DATA)
            .withColumn(
                "rn",
                SF.row_number().over(
                    SparkWindow.partitionBy("p1").orderBy("k1", "k2")
                ),
            )
            .orderBy("p1", "k1", "k2")
        )
        assert_frame_ordered_equal(sdf, pdf)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _backend_window(session):
    """Resolve the backend's ``Window`` class from a session instance."""
    module_name = type(session).__module__  # e.g. "sparkleframe.pythondf.session"
    backend_root = module_name.rsplit(".", 1)[0]  # "sparkleframe.pythondf"
    return importlib.import_module(f"{backend_root}.window").Window
