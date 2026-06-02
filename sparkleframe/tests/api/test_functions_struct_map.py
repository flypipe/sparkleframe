"""Batch 19 parity tests: struct + map functions in ``F``.

Covered:
    - ``F.struct(*cols)`` — pack columns into a struct (dict). Field naming
      follows Spark's ``CreateStruct``: string args use the (last segment of
      the) name; ``F.col(name)`` keeps the name; ``F.lit(v).alias("k")`` uses
      the alias; unaliased literals become ``col1``, ``col2``, ....
    - ``F.create_map(*cols)`` — alternating key/value args; returns a ``dict``
      per row. polarsdf encodes maps as ``list<struct<key,value>>``, so direct
      comparison diverges; we look values up via ``getItem`` for parity.
    - ``F.map_keys(col)`` — list of keys; non-empty / empty / null map.
    - ``F.map_from_entries(col)`` — array of two-field structs → map.

Map key ordering is non-deterministic in Spark, but the parity helper sorts
JSON keys before comparison, so ordering does not affect equality.
"""
from __future__ import annotations

import pyspark.sql.functions as SF
from pyspark.sql.types import (
    ArrayType as SparkArrayType,
    LongType as SparkLongType,
    MapType as SparkMapType,
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


_ENTRIES_SCHEMA = SparkStructType(
    [
        SparkStructField(
            "entries",
            SparkArrayType(
                SparkStructType(
                    [
                        SparkStructField("key", SparkStringType()),
                        SparkStructField("value", SparkStringType()),
                    ]
                )
            ),
            nullable=True,
        )
    ]
)

_MAP_STR_STR_SCHEMA = SparkStructType(
    [SparkStructField("m", SparkMapType(SparkStringType(), SparkStringType()), nullable=True)]
)


# ---------------------------------------------------------------------------
# F.struct
# ---------------------------------------------------------------------------


class TestStruct:
    def test_struct_from_string_column_names(self, session, F, spark):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        sdf = session.createDataFrame(rows).select(F.struct("a", "b").alias("s"))
        pdf = spark.createDataFrame(rows).select(SF.struct("a", "b").alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_struct_from_plain_col(self, session, F, spark):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        sdf = session.createDataFrame(rows).select(F.struct(F.col("a"), F.col("b")).alias("s"))
        pdf = spark.createDataFrame(rows).select(SF.struct(SF.col("a"), SF.col("b")).alias("s"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_struct_from_aliased_columns(self, session, F, spark):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        sdf = session.createDataFrame(rows).select(
            F.struct(F.col("a").alias("k"), F.col("b").alias("v")).alias("s")
        )
        pdf = spark.createDataFrame(rows).select(
            SF.struct(SF.col("a").alias("k"), SF.col("b").alias("v")).alias("s")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_struct_from_aliased_literals(self, session, F, spark):
        rows = [{"x": 1}, {"x": 2}]
        sdf = session.createDataFrame(rows).select(
            F.struct(
                F.lit("aaa").alias("f1"),
                F.lit("bbb").alias("f2"),
                F.lit("ccc").alias("f3"),
            ).alias("s")
        )
        pdf = spark.createDataFrame(rows).select(
            SF.struct(
                SF.lit("aaa").alias("f1"),
                SF.lit("bbb").alias("f2"),
                SF.lit("ccc").alias("f3"),
            ).alias("s")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_struct_mixed_aliased_lit_and_col(self, session, F, spark):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        sdf = session.createDataFrame(rows).select(
            F.struct(F.col("b"), F.lit("fixed").alias("tag")).alias("s")
        )
        pdf = spark.createDataFrame(rows).select(
            SF.struct(SF.col("b"), SF.lit("fixed").alias("tag")).alias("s")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.create_map
# ---------------------------------------------------------------------------


class TestCreateMap:
    def test_create_map_two_pairs(self, session, F, spark):
        rows = [{"x": "hello", "y": "world"}, {"x": "foo", "y": "bar"}]
        sdf = session.createDataFrame(rows).withColumn(
            "m", F.create_map(F.lit("x_val"), F.col("x"), F.lit("y_val"), F.col("y"))
        ).withColumn("got_x", F.col("m").getItem("x_val")).withColumn(
            "got_y", F.col("m").getItem("y_val")
        ).select("got_x", "got_y")
        pdf = spark.createDataFrame(rows).withColumn(
            "m", SF.create_map(SF.lit("x_val"), SF.col("x"), SF.lit("y_val"), SF.col("y"))
        ).withColumn("got_x", SF.col("m").getItem("x_val")).withColumn(
            "got_y", SF.col("m").getItem("y_val")
        ).select("got_x", "got_y")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_create_map_three_pairs(self, session, F, spark):
        rows = [{"a": "1", "b": "2", "c": "3"}, {"a": "10", "b": "20", "c": "30"}]
        sdf = session.createDataFrame(rows).withColumn(
            "m",
            F.create_map(
                F.lit("k1"), F.col("a"),
                F.lit("k2"), F.col("b"),
                F.lit("k3"), F.col("c"),
            ),
        ).withColumn("v1", F.col("m").getItem("k1")).withColumn(
            "v2", F.col("m").getItem("k2")
        ).withColumn("v3", F.col("m").getItem("k3")).select("v1", "v2", "v3")
        pdf = spark.createDataFrame(rows).withColumn(
            "m",
            SF.create_map(
                SF.lit("k1"), SF.col("a"),
                SF.lit("k2"), SF.col("b"),
                SF.lit("k3"), SF.col("c"),
            ),
        ).withColumn("v1", SF.col("m").getItem("k1")).withColumn(
            "v2", SF.col("m").getItem("k2")
        ).withColumn("v3", SF.col("m").getItem("k3")).select("v1", "v2", "v3")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.map_keys
# ---------------------------------------------------------------------------


class TestMapKeys:
    def test_map_keys_non_empty(self, session, F, spark):
        # Build a map column via create_map so both backends produce their native
        # map encoding; then compare the size of the key list (set-equality on a
        # scalar avoids list-ordering noise from Spark's unordered map type).
        rows = [{"x": "v1", "y": "v2"}]
        sdf = session.createDataFrame(rows).withColumn(
            "m", F.create_map(F.lit("a"), F.col("x"), F.lit("b"), F.col("y"))
        ).withColumn("n", F.size(F.map_keys(F.col("m")))).select("n")
        pdf = spark.createDataFrame(rows).withColumn(
            "m", SF.create_map(SF.lit("a"), SF.col("x"), SF.lit("b"), SF.col("y"))
        ).withColumn("n", SF.size(SF.map_keys(SF.col("m")))).select("n")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_map_keys_empty_map(self, session, F, spark):
        rows = [{"m": {}}]
        sdf = session.createDataFrame(rows).withColumn("n", F.size(F.map_keys(F.col("m")))).select("n")
        pdf = spark.createDataFrame(rows, schema=_MAP_STR_STR_SCHEMA).withColumn(
            "n", SF.size(SF.map_keys(SF.col("m")))
        ).select("n")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_map_keys_null_map(self, session, F, spark):
        rows = [{"m": None}]
        sdf = session.createDataFrame(rows).withColumn("ks", F.map_keys(F.col("m"))).select("ks")
        pdf = spark.createDataFrame(rows, schema=_MAP_STR_STR_SCHEMA).withColumn(
            "ks", SF.map_keys(SF.col("m"))
        ).select("ks")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.map_from_entries
# ---------------------------------------------------------------------------


class TestMapFromEntries:
    def test_map_from_entries_against_spark(self, session, F, spark):
        rows = [
            {"entries": [{"key": "utm_source", "value": "google"}, {"key": "utm_medium", "value": "cpc"}]},
            {"entries": [{"key": "plan", "value": "premium"}]},
        ]
        sdf = session.createDataFrame(rows).withColumn(
            "m", F.map_from_entries(F.col("entries"))
        ).withColumn("src", F.col("m").getItem("utm_source")).select("src")
        pdf = spark.createDataFrame(rows, schema=_ENTRIES_SCHEMA).withColumn(
            "m", SF.map_from_entries(SF.col("entries"))
        ).withColumn("src", SF.col("m").getItem("utm_source")).select("src")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_map_from_entries_empty_array(self, session, F, spark):
        rows = [{"entries": []}]
        sdf = session.createDataFrame(rows).withColumn(
            "m", F.map_from_entries(F.col("entries"))
        ).withColumn("n", F.size(F.map_keys(F.col("m")))).select("n")
        pdf = spark.createDataFrame(rows, schema=_ENTRIES_SCHEMA).withColumn(
            "m", SF.map_from_entries(SF.col("entries"))
        ).withColumn("n", SF.size(SF.map_keys(SF.col("m")))).select("n")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_map_from_entries_null_array(self, session, F, spark):
        rows = [{"entries": None}]
        sdf = session.createDataFrame(rows).withColumn(
            "m", F.map_from_entries(F.col("entries"))
        ).withColumn("ks", F.map_keys(F.col("m"))).select("ks")
        pdf = spark.createDataFrame(rows, schema=_ENTRIES_SCHEMA).withColumn(
            "m", SF.map_from_entries(SF.col("entries"))
        ).withColumn("ks", SF.map_keys(SF.col("m"))).select("ks")
        assert_sparkle_spark_frame_are_equal(sdf, pdf)
