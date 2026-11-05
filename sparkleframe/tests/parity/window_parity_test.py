"""Shared Spark-parity tests for ranking window functions (``row_number`` / ``dense_rank``).

Mirrors the datahub nba pipeline usage: ``Window.partitionBy(...).orderBy(<1..3 sort keys>)`` with
``.desc()`` / ``.desc_nulls_last()`` / ``.asc_nulls_last()``, followed by the ``rn == 1`` pick-one
filter. Spark gives no order guarantee among ties for ``row_number``, so ``row_number`` cases use
sort keys that are unique within each partition, and ties are exercised with ``dense_rank``.

``Window`` is reached through the engine adapter (``engine.Window``), like ``engine.functions``.
"""

import pyspark.sql.functions as SF
import pytest
from pyspark.sql import Window as SparkWindow
from pyspark.sql.types import LongType as SparkLongType
from pyspark.sql.types import StringType as SparkStringType
from pyspark.sql.types import StructField as SparkStructField
from pyspark.sql.types import StructType as SparkStructType

from sparkleframe.tests.parity.oracle import assert_matches_spark

_LONG = SparkLongType()
_STR = SparkStringType()


def _schema(*fields):
    return SparkStructType([SparkStructField(name, dtype, True) for name, dtype in fields])


def _sort_key(functions, name: str, ordering: str):
    """``F.col(name).<ordering>()`` for either side (``ordering`` is a Column sort-method name)."""
    return getattr(functions.col(name), ordering)()


# Unique (ts) per partition so row_number is deterministic; includes a null partition key.
_EVENTS_SCHEMA = _schema(("grp", _STR), ("ts", _LONG), ("val", _STR))
_EVENTS_ROWS = [
    ("a", 1, "a1"),
    ("a", 3, "a3"),
    ("a", 2, "a2"),
    ("b", 10, "b10"),
    ("b", 5, "b5"),
    ("c", 7, "c7"),
    (None, 4, "n4"),
    (None, 8, "n8"),
]


# ----------------------------------------------------------------------------- #
# row_number: partition + order, pick-one, multi-key
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("window.row_number.partition_and_order")
@pytest.mark.parametrize("ordering", ["asc", "desc"])
def test_row_number_partition_and_order(engine, spark, ordering):
    F, W = engine.functions, engine.Window
    actual = engine.build_df(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
        "rn", F.row_number().over(W.partitionBy("grp").orderBy(_sort_key(F, "ts", ordering)))
    )
    expected = spark.createDataFrame(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
        "rn", SF.row_number().over(SparkWindow.partitionBy("grp").orderBy(_sort_key(SF, "ts", ordering)))
    )
    assert_matches_spark(actual, expected, engine)

    # Malformed: ordering by a column that does not exist is an analysis error in Spark.
    with pytest.raises(Exception, match="UNRESOLVED_COLUMN"):
        spark.createDataFrame(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
            "rn", SF.row_number().over(SparkWindow.partitionBy("grp").orderBy(SF.col("missing")))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
                "rn", F.row_number().over(W.partitionBy("grp").orderBy(F.col("missing")))
            )
        )


@pytest.mark.feature("window.row_number.partition_and_order")
def test_row_number_pick_one_latest_per_partition(engine, spark):
    F, W = engine.functions, engine.Window
    actual = (
        engine.build_df(_EVENTS_ROWS, _EVENTS_SCHEMA)
        .withColumn("rn", F.row_number().over(W.partitionBy("grp").orderBy(F.col("ts").desc())))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )
    expected = (
        spark.createDataFrame(_EVENTS_ROWS, _EVENTS_SCHEMA)
        .withColumn("rn", SF.row_number().over(SparkWindow.partitionBy("grp").orderBy(SF.col("ts").desc())))
        .filter(SF.col("rn") == 1)
        .drop("rn")
    )
    assert_matches_spark(actual, expected, engine)

    # Malformed: partitioning by a column that does not exist is an analysis error in Spark.
    with pytest.raises(Exception, match="UNRESOLVED_COLUMN"):
        spark.createDataFrame(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
            "rn", SF.row_number().over(SparkWindow.partitionBy("missing").orderBy(SF.col("ts").desc()))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
                "rn", F.row_number().over(W.partitionBy("missing").orderBy(F.col("ts").desc()))
            )
        )


# Multi-key: ties on k1 broken by k2 (desc), remaining ties broken by k3 -> unique per partition.
_MULTI_SCHEMA = _schema(("p1", _STR), ("p2", _LONG), ("k1", _LONG), ("k2", _LONG), ("k3", _STR))
_MULTI_ROWS = [
    ("x", 1, 1, 5, "a"),
    ("x", 1, 1, 5, "b"),
    ("x", 1, 1, 9, "c"),
    ("x", 1, 2, 1, "d"),
    ("x", 2, 1, 1, "e"),
    ("x", 2, 1, 2, "f"),
    ("y", 1, 3, 3, "g"),
    (None, None, 1, 1, "h"),
    (None, None, 1, 2, "i"),
]


@pytest.mark.feature("window.row_number.partition_and_order")
@pytest.mark.parametrize(
    "orderings",
    [("asc",), ("asc", "desc"), ("asc", "desc", "asc"), ("desc_nulls_last", "asc_nulls_last", "desc")],
    ids=lambda o: "-".join(o),
)
def test_row_number_multi_key_ordering(engine, spark, orderings):
    F, W = engine.functions, engine.Window
    keys = ["k1", "k2", "k3"][: len(orderings)]
    # With fewer than 3 keys some (k1, k2) ties remain; pick-one is only asserted when keys are unique.
    actual_w = W.partitionBy("p1", "p2").orderBy(*[_sort_key(F, k, o) for k, o in zip(keys, orderings)])
    spark_w = SparkWindow.partitionBy("p1", "p2").orderBy(*[_sort_key(SF, k, o) for k, o in zip(keys, orderings)])
    actual = engine.build_df(_MULTI_ROWS, _MULTI_SCHEMA).withColumn("rn", F.row_number().over(actual_w))
    expected = spark.createDataFrame(_MULTI_ROWS, _MULTI_SCHEMA).withColumn("rn", SF.row_number().over(spark_w))
    if len(keys) == 3:
        assert_matches_spark(actual, expected, engine)
        assert_matches_spark(
            actual.filter(F.col("rn") == 1).drop("rn"), expected.filter(SF.col("rn") == 1).drop("rn"), engine
        )
    else:
        # Ties among the sort keys: only the multiset of (partition, rn) is deterministic.
        assert_matches_spark(actual.select("p1", "p2", "rn"), expected.select("p1", "p2", "rn"), engine)

    # Malformed: one of the sort keys does not exist.
    with pytest.raises(Exception, match="UNRESOLVED_COLUMN"):
        spark.createDataFrame(_MULTI_ROWS, _MULTI_SCHEMA).withColumn(
            "rn", SF.row_number().over(SparkWindow.partitionBy("p1").orderBy(SF.col("k1"), SF.col("nope").desc()))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df(_MULTI_ROWS, _MULTI_SCHEMA).withColumn(
                "rn", F.row_number().over(W.partitionBy("p1").orderBy(F.col("k1"), F.col("nope").desc()))
            )
        )


# ----------------------------------------------------------------------------- #
# dense_rank: ties (including tied null sort keys and a null partition)
# ----------------------------------------------------------------------------- #

_TIES_SCHEMA = _schema(("grp", _STR), ("score", _LONG), ("val", _STR))
_TIES_ROWS = [
    ("a", 10, "a1"),
    ("a", 10, "a2"),
    ("a", 7, "a3"),
    ("a", 7, "a4"),
    ("a", 3, "a5"),
    ("b", 5, "b1"),
    ("b", None, "b2"),
    ("b", None, "b3"),
    (None, 1, "n1"),
    (None, 1, "n2"),
    (None, 2, "n3"),
]


@pytest.mark.feature("window.dense_rank.ties")
@pytest.mark.parametrize("ordering", ["asc", "desc", "desc_nulls_last", "asc_nulls_last"])
def test_dense_rank_ties(engine, spark, ordering):
    F, W = engine.functions, engine.Window
    actual = engine.build_df(_TIES_ROWS, _TIES_SCHEMA).withColumn(
        "dr", F.dense_rank().over(W.partitionBy("grp").orderBy(_sort_key(F, "score", ordering)))
    )
    expected = spark.createDataFrame(_TIES_ROWS, _TIES_SCHEMA).withColumn(
        "dr", SF.dense_rank().over(SparkWindow.partitionBy("grp").orderBy(_sort_key(SF, "score", ordering)))
    )
    assert_matches_spark(actual, expected, engine)
    # dense_rank == 1 keeps every row tied for first (deterministic, unlike row_number).
    assert_matches_spark(actual.filter(F.col("dr") == 1), expected.filter(SF.col("dr") == 1), engine)

    # Malformed: dense_rank without ORDER BY is rejected by Spark's analyzer.
    with pytest.raises(Exception, match="requires window to be ordered"):
        spark.createDataFrame(_TIES_ROWS, _TIES_SCHEMA).withColumn(
            "dr", SF.dense_rank().over(SparkWindow.partitionBy("grp"))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df(_TIES_ROWS, _TIES_SCHEMA).withColumn("dr", F.dense_rank().over(W.partitionBy("grp")))
        )


@pytest.mark.feature("window.dense_rank.ties")
def test_dense_rank_ties_multi_key(engine, spark):
    F, W = engine.functions, engine.Window
    actual = engine.build_df(_MULTI_ROWS, _MULTI_SCHEMA).withColumn(
        "dr", F.dense_rank().over(W.partitionBy("p1", "p2").orderBy(F.col("k1"), F.col("k2").desc()))
    )
    expected = spark.createDataFrame(_MULTI_ROWS, _MULTI_SCHEMA).withColumn(
        "dr", SF.dense_rank().over(SparkWindow.partitionBy("p1", "p2").orderBy(SF.col("k1"), SF.col("k2").desc()))
    )
    assert_matches_spark(actual, expected, engine)

    # Malformed: an ORDER BY key that does not exist.
    with pytest.raises(Exception, match="UNRESOLVED_COLUMN"):
        spark.createDataFrame(_MULTI_ROWS, _MULTI_SCHEMA).withColumn(
            "dr", SF.dense_rank().over(SparkWindow.partitionBy("p1").orderBy(SF.col("nope")))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df(_MULTI_ROWS, _MULTI_SCHEMA).withColumn(
                "dr", F.dense_rank().over(W.partitionBy("p1").orderBy(F.col("nope")))
            )
        )


# ----------------------------------------------------------------------------- #
# Null ordering of sort keys
# ----------------------------------------------------------------------------- #

# One null sort key per partition, other keys unique -> row_number is deterministic under every null ordering.
_NULLS_SCHEMA = _schema(("grp", _STR), ("ts", _LONG), ("val", _STR))
_NULLS_ROWS = [
    ("a", 2, "a2"),
    ("a", None, "a_null"),
    ("a", 5, "a5"),
    ("a", 1, "a1"),
    ("b", None, "b_null"),
    ("b", 9, "b9"),
    ("c", None, "c_null"),
    (None, None, "n_null"),
    (None, 3, "n3"),
]

_NULL_ORDERINGS = ["asc", "desc", "asc_nulls_first", "asc_nulls_last", "desc_nulls_first", "desc_nulls_last"]


@pytest.mark.feature("window.order_by.desc_nulls_last")
@pytest.mark.parametrize("ordering", _NULL_ORDERINGS)
def test_order_by_null_ordering_row_number(engine, spark, ordering):
    F, W = engine.functions, engine.Window
    actual = engine.build_df(_NULLS_ROWS, _NULLS_SCHEMA).withColumn(
        "rn", F.row_number().over(W.partitionBy("grp").orderBy(_sort_key(F, "ts", ordering)))
    )
    expected = spark.createDataFrame(_NULLS_ROWS, _NULLS_SCHEMA).withColumn(
        "rn", SF.row_number().over(SparkWindow.partitionBy("grp").orderBy(_sort_key(SF, "ts", ordering)))
    )
    assert_matches_spark(actual, expected, engine)
    # Pick-one: whether the null row wins depends entirely on the null ordering.
    assert_matches_spark(actual.filter(F.col("rn") == 1), expected.filter(SF.col("rn") == 1), engine)

    # Malformed: sorting a missing column with the same null ordering.
    with pytest.raises(Exception, match="UNRESOLVED_COLUMN"):
        spark.createDataFrame(_NULLS_ROWS, _NULLS_SCHEMA).withColumn(
            "rn", SF.row_number().over(SparkWindow.partitionBy("grp").orderBy(_sort_key(SF, "missing", ordering)))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df(_NULLS_ROWS, _NULLS_SCHEMA).withColumn(
                "rn", F.row_number().over(W.partitionBy("grp").orderBy(_sort_key(F, "missing", ordering)))
            )
        )


@pytest.mark.feature("window.order_by.desc_nulls_last")
def test_order_by_desc_nulls_last_then_asc_nulls_last(engine, spark):
    # Two keys, nulls in both: (k1 desc_nulls_last, k2 asc_nulls_last, k3 desc) is unique per partition.
    F, W = engine.functions, engine.Window
    schema = _schema(("grp", _STR), ("k1", _LONG), ("k2", _LONG), ("k3", _STR))
    rows = [
        ("a", None, None, "a"),
        ("a", None, 1, "b"),
        ("a", 2, None, "c"),
        ("a", 2, 1, "d"),
        ("a", 2, 1, "e"),
        ("a", 3, 0, "f"),
        ("b", None, None, "g"),
        ("b", None, None, "h"),
        (None, 1, None, "i"),
        (None, 1, 4, "j"),
    ]
    actual_w = W.partitionBy("grp").orderBy(
        F.col("k1").desc_nulls_last(), F.col("k2").asc_nulls_last(), F.col("k3").desc()
    )
    spark_w = SparkWindow.partitionBy("grp").orderBy(
        SF.col("k1").desc_nulls_last(), SF.col("k2").asc_nulls_last(), SF.col("k3").desc()
    )
    actual = engine.build_df(rows, schema).withColumn("rn", F.row_number().over(actual_w))
    expected = spark.createDataFrame(rows, schema).withColumn("rn", SF.row_number().over(spark_w))
    assert_matches_spark(actual, expected, engine)
    assert_matches_spark(
        actual.filter(F.col("rn") == 1).drop("rn"), expected.filter(SF.col("rn") == 1).drop("rn"), engine
    )

    # Malformed: row_number over a spec with partition only (null ordering never applied) must raise.
    with pytest.raises(Exception, match="requires window to be ordered"):
        spark_df = spark.createDataFrame(rows, schema)
        spark_df.withColumn("rn", SF.row_number().over(SparkWindow.partitionBy("grp"))).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).withColumn("rn", F.row_number().over(W.partitionBy("grp"))))


# ----------------------------------------------------------------------------- #
# Ranking without ORDER BY
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("window.row_number.without_order_raises")
@pytest.mark.parametrize("fn", ["row_number", "dense_rank"])
def test_ranking_without_order_raises(engine, spark, fn):
    F, W = engine.functions, engine.Window

    # Valid: the same function with an ORDER BY (and a global, unpartitioned window) works.
    actual = engine.build_df(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
        "r", getattr(F, fn)().over(W.orderBy(F.col("ts").desc()))
    )
    expected = spark.createDataFrame(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
        "r", getattr(SF, fn)().over(SparkWindow.orderBy(SF.col("ts").desc()))
    )
    assert_matches_spark(actual, expected, engine)

    # Malformed: partitioned window with no ORDER BY.
    with pytest.raises(Exception, match="requires window to be ordered"):
        spark.createDataFrame(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
            "r", getattr(SF, fn)().over(SparkWindow.partitionBy("grp"))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn("r", getattr(F, fn)().over(W.partitionBy("grp")))
        )

    # Malformed: multi-column partition, still no ORDER BY.
    with pytest.raises(Exception, match="requires window to be ordered"):
        spark.createDataFrame(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
            "r", getattr(SF, fn)().over(SparkWindow.partitionBy("grp", "val"))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df(_EVENTS_ROWS, _EVENTS_SCHEMA).withColumn(
                "r", getattr(F, fn)().over(W.partitionBy("grp", "val"))
            )
        )
