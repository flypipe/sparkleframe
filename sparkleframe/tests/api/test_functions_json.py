"""Batch 20 parity tests: JSON functions in ``F``.

Covered:
    - ``F.get_json_object(col, path)`` — extract via Spark JSONPath subset
      (``$``, ``.name``, ``[N]``, ``['name']``). Returns string of the
      extracted value, null for missing path / malformed JSON / null input.
    - ``F.from_json(col, schema)`` — parse a JSON string column to a
      struct (or array/map per schema). Schema may be a sparkleframe
      ``StructType`` or a DDL string (``"a INT, b STRING"``). Parse error
      → null.
    - ``F.to_json(col, options=None)`` — serialize struct/array/map to JSON
      string. Default ``ignoreNullFields=True`` drops null struct fields,
      matching Spark.
"""
from __future__ import annotations

import json

import pyspark.sql.functions as SF
from pyspark.sql.types import (
    IntegerType as SparkIntegerType,
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


# ---------------------------------------------------------------------------
# F.get_json_object
# ---------------------------------------------------------------------------


class TestGetJsonObject:
    def test_top_level_field(self, session, F, spark):
        rows = [{"j": json.dumps({"a": "b"})}, {"j": json.dumps({"a": "c"})}]
        sdf = session.createDataFrame(rows).select(F.get_json_object("j", "$.a").alias("r"))
        pdf = spark.createDataFrame(rows).select(SF.get_json_object("j", "$.a").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_nested_field(self, session, F, spark):
        rows = [{"j": json.dumps({"a": {"b": "x"}})}, {"j": json.dumps({"a": {"b": "y"}})}]
        sdf = session.createDataFrame(rows).select(F.get_json_object("j", "$.a.b").alias("r"))
        pdf = spark.createDataFrame(rows).select(SF.get_json_object("j", "$.a.b").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_array_index(self, session, F, spark):
        rows = [{"j": json.dumps({"a": [10, 20, 30]})}, {"j": json.dumps({"a": [40, 50, 60]})}]
        sdf = session.createDataFrame(rows).select(F.get_json_object("j", "$.a[0]").alias("r"))
        pdf = spark.createDataFrame(rows).select(SF.get_json_object("j", "$.a[0]").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_nested_array_of_structs(self, session, F, spark):
        rows = [
            {"j": json.dumps({"items": [{"id": 1}, {"id": 2}]})},
            {"j": json.dumps({"items": [{"id": 3}, {"id": 4}]})},
        ]
        sdf = session.createDataFrame(rows).select(
            F.get_json_object("j", "$.items[1].id").alias("r")
        )
        pdf = spark.createDataFrame(rows).select(
            SF.get_json_object("j", "$.items[1].id").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_missing_key_returns_null(self, session, F, spark):
        rows = [{"j": json.dumps({"a": 1})}]
        sdf = session.createDataFrame(rows).select(F.get_json_object("j", "$.missing").alias("r"))
        pdf = spark.createDataFrame(rows).select(SF.get_json_object("j", "$.missing").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_malformed_json_returns_null(self, session, F, spark):
        rows = [{"j": "not-a-json"}]
        sdf = session.createDataFrame(rows).select(F.get_json_object("j", "$.a").alias("r"))
        pdf = spark.createDataFrame(rows).select(SF.get_json_object("j", "$.a").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_null_input_returns_null(self, session, F, spark):
        rows = [{"j": None}]
        sdf = session.createDataFrame(rows).select(F.get_json_object("j", "$.a").alias("r"))
        spark_schema = SparkStructType([SparkStructField("j", SparkStringType(), nullable=True)])
        pdf = spark.createDataFrame(rows, schema=spark_schema).select(
            SF.get_json_object("j", "$.a").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.from_json
# ---------------------------------------------------------------------------


_SPARK_STRUCT_SCHEMA = SparkStructType(
    [
        SparkStructField("field1", SparkStringType()),
        SparkStructField("field2", SparkIntegerType()),
    ]
)


class TestFromJson:
    def test_struct_schema_typed(self, session, F, T, spark):
        payload = json.dumps({"field1": "hello", "field2": 999})
        rows = [{"j": payload}]
        backend_schema = T.StructType(
            [
                T.StructField("field1", T.StringType()),
                T.StructField("field2", T.IntegerType()),
            ]
        )
        sdf = session.createDataFrame(rows).select(
            F.from_json(F.col("j"), backend_schema).alias("parsed")
        )
        pdf = spark.createDataFrame(rows).select(
            SF.from_json(SF.col("j"), _SPARK_STRUCT_SCHEMA).alias("parsed")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_ddl_string_schema(self, session, F, spark):
        payload = json.dumps({"a": 7, "b": "hi"})
        rows = [{"j": payload}]
        sdf = session.createDataFrame(rows).select(
            F.from_json(F.col("j"), "a INT, b STRING").alias("parsed")
        )
        pdf = spark.createDataFrame(rows).select(
            SF.from_json(SF.col("j"), "a INT, b STRING").alias("parsed")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_malformed_returns_null(self, session, F, T, spark):
        rows = [{"j": "not-json"}]
        backend_schema = T.StructType(
            [
                T.StructField("field1", T.StringType()),
                T.StructField("field2", T.IntegerType()),
            ]
        )
        sdf = session.createDataFrame(rows).select(
            F.from_json(F.col("j"), backend_schema).alias("parsed")
        )
        # PySpark wraps malformed parses in an all-null struct rather than a
        # single null scalar; compare the .field1 / .field2 components, both
        # of which are null in either case.
        sdf_extracted = sdf.select(
            F.col("parsed").getItem("field1").alias("f1"),
            F.col("parsed").getItem("field2").alias("f2"),
        )
        pdf = spark.createDataFrame(rows).select(
            SF.from_json(SF.col("j"), _SPARK_STRUCT_SCHEMA).alias("parsed")
        )
        pdf_extracted = pdf.select(
            SF.col("parsed").getItem("field1").alias("f1"),
            SF.col("parsed").getItem("field2").alias("f2"),
        )
        assert_sparkle_spark_frame_are_equal(sdf_extracted, pdf_extracted)


# ---------------------------------------------------------------------------
# F.to_json
# ---------------------------------------------------------------------------


class TestToJson:
    def test_struct_to_json(self, session, F, spark):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        sdf = session.createDataFrame(rows).select(
            F.to_json(F.struct("a", "b")).alias("j")
        )
        pdf = spark.createDataFrame(rows).select(
            SF.to_json(SF.struct("a", "b")).alias("j")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_array_to_json(self, session, F, spark):
        rows = [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
        sdf = session.createDataFrame(rows).select(
            F.to_json(F.array("a", "b")).alias("j")
        )
        pdf = spark.createDataFrame(rows).select(
            SF.to_json(SF.array("a", "b")).alias("j")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_struct_with_null_field_drops_key_by_default(self, session, F, spark):
        # Default ignoreNullFields=True: null struct fields are dropped.
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": None}]
        sdf = session.createDataFrame(rows).select(
            F.to_json(F.struct("a", "b")).alias("j")
        )
        pdf = spark.createDataFrame(rows).select(
            SF.to_json(SF.struct("a", "b")).alias("j")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)
