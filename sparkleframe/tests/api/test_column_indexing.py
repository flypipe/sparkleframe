"""Batch 7 parity tests: Column.getItem / Column.getField / Column.__getitem__.

Covers struct field access (string key), array element access (int index),
map value access (string key), and nested chains. Tests are written so that
each operation is wrapped in an explicit ``.alias(...)`` — this avoids
backend-vs-PySpark divergence on default column naming (PySpark renders
``s.a`` for struct fields and ``arr[0]`` for arrays).

The PySpark side needs explicit schemas for Map/Array/Struct construction
(otherwise inference picks wrong dtypes for nested null elements). The
sparkleframe backends infer schema from raw Python rows.
"""
from __future__ import annotations

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import (
    ArrayType as SparkArrayType,
    IntegerType as SparkIntegerType,
    MapType as SparkMapType,
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


# -----------------------------------------------------------------------------
# Struct field access
# -----------------------------------------------------------------------------


class TestStructFieldAccess:
    """col.getItem('field') / col.getField('field') / col['field'] on structs."""

    @staticmethod
    def _struct_schema():
        return SparkStructType([
            SparkStructField(
                "s",
                SparkStructType([
                    SparkStructField("a", SparkIntegerType(), nullable=True),
                    SparkStructField("b", SparkIntegerType(), nullable=True),
                ]),
                nullable=True,
            )
        ])

    def test_get_item_struct_field(self, session, F, spark):
        rows = [{"s": {"a": 1, "b": 10}}, {"s": {"a": 2, "b": 20}}]
        df_sp = spark.createDataFrame(rows, schema=self._struct_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("s").getItem("a").alias("a"))
        result = df.select(F.col("s").getItem("a").alias("a"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_get_field_struct(self, session, F, spark):
        rows = [{"s": {"a": 1, "b": 10}}, {"s": {"a": 2, "b": 20}}]
        df_sp = spark.createDataFrame(rows, schema=self._struct_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("s").getField("a").alias("a"))
        result = df.select(F.col("s").getField("a").alias("a"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_getitem_sugar_struct(self, session, F, spark):
        rows = [{"s": {"a": 1, "b": 10}}, {"s": {"a": 2, "b": 20}}]
        df_sp = spark.createDataFrame(rows, schema=self._struct_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("s")["a"].alias("a"))
        result = df.select(F.col("s")["a"].alias("a"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_struct_missing_field_returns_null(self, session, F, backend):
        # PySpark raises an AnalysisException at plan time for unknown fields,
        # so this isn't a parity test. Backend behavior: pythondf returns None;
        # polarsdf casts to a null typed expression that also reads as None.
        if backend == "polarsdf":
            pytest.skip("polarsdf's missing-field returns a typed-null column requiring schema setup")
        rows = [{"s": {"a": 1}}]
        df = session.createDataFrame(rows)
        out = df.select(F.col("s").getItem("nope").alias("v")).collect()
        assert out == [{"v": None}]

    def test_struct_null_input(self, session, F, spark):
        rows = [{"s": {"a": 1}}, {"s": None}]
        df_sp = spark.createDataFrame(rows, schema=self._struct_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("s").getItem("a").alias("a"))
        result = df.select(F.col("s").getItem("a").alias("a"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)


# -----------------------------------------------------------------------------
# Array element access
# -----------------------------------------------------------------------------


class TestArrayIndexAccess:
    """col.getItem(int) / col[int] on ArrayType."""

    @staticmethod
    def _array_schema():
        return SparkStructType([
            SparkStructField("arr", SparkArrayType(SparkIntegerType()), nullable=True)
        ])

    def test_get_item_array_first(self, session, F, spark):
        rows = [{"arr": [10, 20, 30]}, {"arr": [40, 50, 60]}]
        df_sp = spark.createDataFrame(rows, schema=self._array_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("arr").getItem(0).alias("v"))
        result = df.select(F.col("arr").getItem(0).alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_get_item_array_middle(self, session, F, spark):
        rows = [{"arr": [10, 20, 30]}, {"arr": [40, 50, 60]}]
        df_sp = spark.createDataFrame(rows, schema=self._array_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("arr").getItem(1).alias("v"))
        result = df.select(F.col("arr").getItem(1).alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_getitem_sugar_array(self, session, F, spark):
        rows = [{"arr": [10, 20, 30]}, {"arr": [40, 50, 60]}]
        df_sp = spark.createDataFrame(rows, schema=self._array_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("arr")[0].alias("v"))
        result = df.select(F.col("arr")[0].alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_array_out_of_bounds_returns_null(self, session, F):
        # In Spark 4 with ANSI mode (the default), getItem(out_of_bounds)
        # raises SparkArrayIndexOutOfBoundsException. Both sparkleframe
        # backends choose the safer null-return behavior; assert that here
        # without comparing to PySpark.
        rows = [{"arr": [10, 20]}]
        df = session.createDataFrame(rows)
        out = df.select(F.col("arr").getItem(5).alias("v")).collect()
        assert out == [{"v": None}]

    def test_array_null_input(self, session, F, spark):
        rows = [{"arr": [1, 2]}, {"arr": None}]
        df_sp = spark.createDataFrame(rows, schema=self._array_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("arr").getItem(0).alias("v"))
        result = df.select(F.col("arr").getItem(0).alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)


# -----------------------------------------------------------------------------
# Map value access
# -----------------------------------------------------------------------------


class TestMapValueAccess:
    """col.getItem('key') / col['key'] on MapType."""

    @staticmethod
    def _map_schema():
        return SparkStructType([
            SparkStructField(
                "m", SparkMapType(SparkStringType(), SparkIntegerType()), nullable=True
            )
        ])

    def test_get_item_map_value(self, session, F, spark):
        rows = [{"m": {"k": 1, "j": 2}}, {"m": {"k": 3, "j": 4}}]
        df_sp = spark.createDataFrame(rows, schema=self._map_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("m").getItem("k").alias("v"))
        result = df.select(F.col("m").getItem("k").alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_getitem_sugar_map(self, session, F, spark):
        rows = [{"m": {"k": 1}}, {"m": {"k": 2}}]
        df_sp = spark.createDataFrame(rows, schema=self._map_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("m")["k"].alias("v"))
        result = df.select(F.col("m")["k"].alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_map_missing_key_returns_null(self, session, F, spark):
        rows = [{"m": {"k": 1}}, {"m": {"k": 2}}]
        df_sp = spark.createDataFrame(rows, schema=self._map_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("m").getItem("missing").alias("v"))
        result = df.select(F.col("m").getItem("missing").alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_map_null_input(self, session, F, spark):
        rows = [{"m": {"k": 1}}, {"m": None}]
        df_sp = spark.createDataFrame(rows, schema=self._map_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("m").getItem("k").alias("v"))
        result = df.select(F.col("m").getItem("k").alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)


# -----------------------------------------------------------------------------
# Nested chains
# -----------------------------------------------------------------------------


class TestNestedChains:
    """Multi-step indexing across struct/array boundaries."""

    @staticmethod
    def _struct_with_array_schema():
        return SparkStructType([
            SparkStructField(
                "nested",
                SparkStructType([
                    SparkStructField(
                        "outer",
                        SparkArrayType(SparkIntegerType()),
                        nullable=True,
                    ),
                ]),
                nullable=True,
            )
        ])

    @staticmethod
    def _struct_of_struct_schema():
        return SparkStructType([
            SparkStructField(
                "nested",
                SparkStructType([
                    SparkStructField(
                        "outer",
                        SparkStructType([
                            SparkStructField("inner", SparkIntegerType(), nullable=True),
                        ]),
                        nullable=True,
                    ),
                ]),
                nullable=True,
            )
        ])

    def test_nested_struct_struct_chained_sugar(self, session, F, spark):
        rows = [{"nested": {"outer": {"inner": 7}}}, {"nested": {"outer": {"inner": 9}}}]
        df_sp = spark.createDataFrame(rows, schema=self._struct_of_struct_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("nested")["outer"]["inner"].alias("v"))
        result = df.select(F.col("nested")["outer"]["inner"].alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)

    def test_nested_struct_array_chained_getitem(self, session, F, spark):
        rows = [{"nested": {"outer": [100, 200]}}, {"nested": {"outer": [300, 400]}}]
        df_sp = spark.createDataFrame(rows, schema=self._struct_with_array_schema())
        df = session.createDataFrame(rows)

        result_sp = df_sp.select(SF.col("nested").getItem("outer").getItem(0).alias("v"))
        result = df.select(F.col("nested").getItem("outer").getItem(0).alias("v"))
        assert_sparkle_spark_frame_are_equal(result, result_sp)


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------


class TestInputValidation:
    """Type guards on getItem keys."""

    def test_get_item_rejects_invalid_key_type(self, session, F):
        rows = [{"s": {"a": 1}}]
        df = session.createDataFrame(rows)
        with pytest.raises(TypeError):
            df.select(F.col("s").getItem(1.5))
