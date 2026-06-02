"""Batch 18 parity tests: array functions in ``F``.

Covered:
    - ``F.array(*cols)`` — pack columns into a list per row.
    - ``F.array_contains(col, value)`` — null array → null; null value → null.
    - ``F.size(col)`` — length of array/map; ``null`` for null input (Spark 3+).
    - ``F.element_at(col, index)`` / ``F.try_element_at`` — 1-indexed, negatives
      from end; ``element_at`` raises out-of-bounds (Spark 4 ANSI),
      ``try_element_at`` returns null.
    - ``F.sort_array(col, asc=True)`` — ascending puts nulls first; descending
      puts them last.
    - ``F.explode(col)`` — row-multiplying. pythondf and polarsdf both match
      ``F.explode_outer`` (one null row per empty / null source array), so the
      parity tests compare against ``SF.explode_outer``.
    - ``F.transform(col, func)`` — higher-order map over array elements.
    - ``F.filter(col, func)`` — higher-order filter over array elements.

All tests run against both backends via the ``(session, F, spark)`` fixtures
and assert parity with native PySpark.
"""
from __future__ import annotations

import pytest

import pyspark.sql.functions as SF
from pyspark.sql.types import (
    ArrayType as SparkArrayType,
    IntegerType as SparkIntegerType,
    LongType as SparkLongType,
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


_INT_ARRAY_SCHEMA = SparkStructType(
    [SparkStructField("a", SparkArrayType(SparkLongType(), True), nullable=True)]
)
_STR_ARRAY_SCHEMA = SparkStructType(
    [SparkStructField("a", SparkArrayType(SparkStringType(), True), nullable=True)]
)


# ---------------------------------------------------------------------------
# F.array
# ---------------------------------------------------------------------------


class TestArray:
    def test_array_two_columns(self, session, F, spark):
        rows = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        sdf = session.createDataFrame(rows).select(F.array("a", "b").alias("arr"))
        pdf = spark.createDataFrame(rows).select(SF.array("a", "b").alias("arr"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_array_three_columns(self, session, F, spark):
        rows = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
        sdf = session.createDataFrame(rows).select(F.array("a", "b", "c").alias("arr"))
        pdf = spark.createDataFrame(rows).select(SF.array("a", "b", "c").alias("arr"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_array_with_null_element(self, session, F, spark):
        rows = [{"a": 1, "b": None}, {"a": None, "b": 5}]
        schema = SparkStructType(
            [
                SparkStructField("a", SparkLongType(), True),
                SparkStructField("b", SparkLongType(), True),
            ]
        )
        sdf = session.createDataFrame(rows).select(F.array("a", "b").alias("arr"))
        pdf = spark.createDataFrame(rows, schema=schema).select(SF.array("a", "b").alias("arr"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.array_contains
# ---------------------------------------------------------------------------


class TestArrayContains:
    def test_present(self, session, F, spark):
        rows = [{"a": [1, 2, 3]}, {"a": [10]}]
        sdf = session.createDataFrame(rows).select(F.array_contains("a", 2).alias("c"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.array_contains("a", 2).alias("c")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_absent(self, session, F, spark):
        rows = [{"a": [1, 2, 3]}, {"a": [10]}]
        sdf = session.createDataFrame(rows).select(F.array_contains("a", 99).alias("c"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.array_contains("a", 99).alias("c")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_null_array_returns_null(self, session, F, spark):
        rows = [{"a": [1, 2]}, {"a": None}]
        sdf = session.createDataFrame(rows).select(F.array_contains("a", 1).alias("c"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.array_contains("a", 1).alias("c")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.size
# ---------------------------------------------------------------------------


class TestSize:
    def test_non_empty(self, session, F, spark):
        rows = [{"a": [1, 2, 3]}, {"a": [10]}]
        sdf = session.createDataFrame(rows).select(F.size("a").alias("sz"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(SF.size("a").alias("sz"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_array(self, session, F, spark):
        rows = [{"a": [1]}, {"a": []}]
        sdf = session.createDataFrame(rows).select(F.size("a").alias("sz"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(SF.size("a").alias("sz"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_null_array(self, session, F, spark):
        rows = [{"a": [1]}, {"a": None}]
        sdf = session.createDataFrame(rows).select(F.size("a").alias("sz"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(SF.size("a").alias("sz"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.element_at / F.try_element_at
# ---------------------------------------------------------------------------


class TestElementAt:
    def test_first_index(self, session, F, spark):
        rows = [{"a": ["x", "y", "z"]}, {"a": ["m", "n"]}]
        sdf = session.createDataFrame(rows).select(F.element_at("a", 1).alias("v"))
        pdf = spark.createDataFrame(rows, schema=_STR_ARRAY_SCHEMA).select(
            SF.element_at("a", 1).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_last_index_via_negative_one(self, session, F, spark):
        rows = [{"a": ["x", "y", "z"]}, {"a": ["m", "n"]}]
        sdf = session.createDataFrame(rows).select(F.element_at("a", -1).alias("v"))
        pdf = spark.createDataFrame(rows, schema=_STR_ARRAY_SCHEMA).select(
            SF.element_at("a", -1).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_out_of_bounds_raises(self, session, F, spark):
        rows = [{"a": ["x"]}]
        sdf = session.createDataFrame(rows)
        with pytest.raises(Exception):
            sdf.select(F.element_at("a", 5).alias("v")).collect()
        pdf = spark.createDataFrame(rows, schema=_STR_ARRAY_SCHEMA)
        with pytest.raises(Exception):
            pdf.select(SF.element_at("a", 5).alias("v")).collect()


class TestTryElementAt:
    def test_out_of_bounds_returns_null(self, session, F, spark):
        rows = [{"a": ["x", "y"]}, {"a": ["m"]}]
        sdf = session.createDataFrame(rows).select(F.try_element_at("a", 5).alias("v"))
        pdf = spark.createDataFrame(rows, schema=_STR_ARRAY_SCHEMA).select(
            SF.try_element_at("a", SF.lit(5)).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_negative_index(self, session, F, spark):
        rows = [{"a": ["x", "y", "z"]}, {"a": ["m", "n"]}]
        sdf = session.createDataFrame(rows).select(F.try_element_at("a", -2).alias("v"))
        pdf = spark.createDataFrame(rows, schema=_STR_ARRAY_SCHEMA).select(
            SF.try_element_at("a", SF.lit(-2)).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.sort_array
# ---------------------------------------------------------------------------


class TestSortArray:
    def test_ascending_default(self, session, F, spark):
        rows = [{"a": [3, 1, 2]}, {"a": [6, 4, 5]}]
        sdf = session.createDataFrame(rows).select(F.sort_array("a").alias("s"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.sort_array("a").alias("s")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_descending(self, session, F, spark):
        rows = [{"a": [3, 1, 2]}, {"a": [6, 4, 5]}]
        sdf = session.createDataFrame(rows).select(F.sort_array("a", asc=False).alias("s"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.sort_array("a", asc=False).alias("s")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_with_nulls_asc_puts_them_first(self, session, F, spark):
        rows = [{"a": [3, None, 1, 2]}]
        sdf = session.createDataFrame(rows).select(F.sort_array("a").alias("s"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.sort_array("a").alias("s")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_with_nulls_desc_puts_them_last(self, session, F, spark):
        rows = [{"a": [3, None, 1, 2]}]
        sdf = session.createDataFrame(rows).select(F.sort_array("a", asc=False).alias("s"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.sort_array("a", asc=False).alias("s")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.explode (matches PySpark's explode_outer — see backend docstrings)
# ---------------------------------------------------------------------------


class TestExplode:
    def test_three_element_array_expands_to_three_rows(self, session, F, spark):
        rows = [{"a": [1, 2, 3]}]
        sdf = session.createDataFrame(rows).select(F.explode("a").alias("e"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.explode_outer("a").alias("e")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_empty_array_yields_null_row(self, session, F, spark):
        rows = [{"a": [1, 2]}, {"a": []}]
        sdf = session.createDataFrame(rows).select(F.explode("a").alias("e"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.explode_outer("a").alias("e")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_null_array_yields_null_row(self, session, F, spark):
        rows = [{"a": [1, 2]}, {"a": None}]
        sdf = session.createDataFrame(rows).select(F.explode("a").alias("e"))
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.explode_outer("a").alias("e")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.transform (higher-order)
# ---------------------------------------------------------------------------


class TestTransform:
    def test_increment_each_element(self, session, F, spark):
        rows = [{"a": [1, 2, 3]}, {"a": [10]}]
        sdf = session.createDataFrame(rows).select(
            F.transform("a", lambda x: x + 1).alias("t")
        )
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.transform("a", lambda x: x + 1).alias("t")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.filter (higher-order on arrays — not DataFrame.filter)
# ---------------------------------------------------------------------------


class TestFilter:
    def test_keep_elements_greater_than_two(self, session, F, spark):
        rows = [{"a": [1, 2, 3, 4]}, {"a": [10, 0, 5]}]
        sdf = session.createDataFrame(rows).select(
            F.filter("a", lambda x: x > 2).alias("f")
        )
        pdf = spark.createDataFrame(rows, schema=_INT_ARRAY_SCHEMA).select(
            SF.filter("a", lambda x: x > 2).alias("f")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)
