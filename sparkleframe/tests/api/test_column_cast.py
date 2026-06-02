"""Batch 5 parity tests: Column.cast and Column.try_cast.

Spark 4 has ANSI mode enabled by default, so ``cast`` raises
``CAST_INVALID_INPUT`` on malformed input while ``try_cast`` returns NULL.
Tests run against both backends (``polarsdf``, ``pythondf``) and assert parity
against native PySpark via ``assert_sparkle_spark_frame_are_equal`` for valid
paths, plus direct ``pytest.raises`` for ANSI-strict paths.

The ``T`` fixture resolves to the backend's ``types`` module so each backend
receives its own type instances (polarsdf's ``cast`` rejects pythondf types
via ``isinstance``).
"""
from __future__ import annotations

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import (
    BooleanType as SparkBooleanType,
    DateType as SparkDateType,
    DecimalType as SparkDecimalType,
    DoubleType as SparkDoubleType,
    FloatType as SparkFloatType,
    IntegerType as SparkIntegerType,
    LongType as SparkLongType,
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
    TimestampType as SparkTimestampType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


def _schema(spark_type, *names):
    return SparkStructType([SparkStructField(n, spark_type, nullable=True) for n in names])


_NUMERIC_PAIRS = [
    ("IntegerType", SparkIntegerType()),
    ("LongType", SparkLongType()),
    ("DoubleType", SparkDoubleType()),
    ("FloatType", SparkFloatType()),
]
_NUMERIC_INVALID_PAIRS = [
    ("IntegerType", SparkIntegerType()),
    ("LongType", SparkLongType()),
    ("DoubleType", SparkDoubleType()),
]
_FLOATING_PAIRS = [
    ("DoubleType", SparkDoubleType()),
    ("FloatType", SparkFloatType()),
]


# -----------------------------------------------------------------------------
# Numeric ↔ numeric casts
# -----------------------------------------------------------------------------


class TestNumericCasts:
    def test_int_to_long(self, session, F, T, spark):
        rows_tuples = [(1,), (-100,), (None,), (2**30,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkIntegerType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkLongType()).alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").cast(T.LongType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_long_to_int_in_range(self, session, F, T, spark):
        rows_tuples = [(1,), (-100,), (None,), (2**30,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkLongType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkIntegerType()).alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").cast(T.IntegerType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_long_to_int_overflow_try_returns_null(self, session, F, T, spark):
        # 2**40 doesn't fit in int32; try_cast returns null
        rows_tuples = [(2**40,), (5,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkLongType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").try_cast(SparkIntegerType()).alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").try_cast(T.IntegerType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_long_to_int_overflow_strict_raises(self, session, F, T):
        rows = [{"s": 2**40}]
        with pytest.raises(Exception):
            session.createDataFrame(rows).select(
                F.col("s").cast(T.IntegerType()).alias("v")
            ).collect()

    def test_float_to_double(self, session, F, T, spark):
        rows_tuples = [(1.5,), (-3.25,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkFloatType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkDoubleType()).alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").cast(T.DoubleType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_double_to_float(self, session, F, T, spark):
        rows_tuples = [(1.5,), (-3.25,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkDoubleType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkFloatType()).alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").cast(T.FloatType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_double_to_int_truncates(self, session, F, T, spark):
        # Spark truncates toward zero (not rounds) for double->int.
        rows_tuples = [(1.7,), (-1.7,), (0.5,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkDoubleType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkIntegerType()).alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").cast(T.IntegerType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# String → numeric casts
# -----------------------------------------------------------------------------


class TestStringToNumeric:
    @pytest.mark.parametrize("sparkle_type_name, spark_dtype", _NUMERIC_PAIRS)
    def test_string_to_numeric_valid(self, session, F, T, spark, sparkle_type_name, spark_dtype):
        # Valid for all numeric targets: integer-looking strings + whitespace + sign.
        rows_tuples = [("123",), ("-7",), (" 10 ",), ("+5",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(spark_dtype).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(getattr(T, sparkle_type_name)()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    @pytest.mark.parametrize("sparkle_type_name, spark_dtype", _FLOATING_PAIRS)
    def test_string_with_decimals_to_floating_valid(
        self, session, F, T, spark, sparkle_type_name, spark_dtype
    ):
        # Decimal + exponent forms only valid for floating types.
        rows_tuples = [("1.7",), ("1e2",), ("-0.5",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(spark_dtype).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(getattr(T, sparkle_type_name)()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    @pytest.mark.parametrize("sparkle_type_name, spark_dtype", _NUMERIC_INVALID_PAIRS)
    def test_string_to_numeric_invalid_strict_raises(
        self, session, F, T, sparkle_type_name, spark_dtype
    ):
        rows = [{"s": "abc"}]
        with pytest.raises(Exception):
            session.createDataFrame(rows).select(
                F.col("s").cast(getattr(T, sparkle_type_name)()).alias("v")
            ).collect()

    @pytest.mark.parametrize("sparkle_type_name, spark_dtype", _NUMERIC_INVALID_PAIRS)
    def test_string_to_numeric_invalid_try_returns_null(
        self, session, F, T, spark, sparkle_type_name, spark_dtype
    ):
        rows_tuples = [("123",), ("abc",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").try_cast(spark_dtype).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").try_cast(getattr(T, sparkle_type_name)()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_string_with_decimal_to_int_strict_raises(self, session, F, T):
        """Spark ANSI: '1.7' cast to int RAISES (no implicit truncation from string)."""
        rows = [{"s": "1.7"}]
        with pytest.raises(Exception):
            session.createDataFrame(rows).select(
                F.col("s").cast(T.IntegerType()).alias("v")
            ).collect()

    def test_string_with_decimal_to_int_try_returns_null(self, session, F, T, spark):
        rows_tuples = [("1.7",), ("1e2",), ("10",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").try_cast(SparkIntegerType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").try_cast(T.IntegerType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# Boolean casts
# -----------------------------------------------------------------------------


class TestBooleanCasts:
    def test_string_to_bool_valid(self, session, F, T, spark):
        # Spark accepts: true/false/t/f/y/n/yes/no/1/0 (case-insensitive).
        rows_tuples = [
            ("true",), ("TRUE",), ("True",), ("t",), ("T",), ("yes",), ("y",), ("Y",), ("1",),
            ("false",), ("FALSE",), ("False",), ("f",), ("F",), ("no",), ("n",), ("N",), ("0",),
            (None,),
        ]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkBooleanType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(T.BooleanType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_string_to_bool_invalid_strict_raises(self, session, F, T):
        rows = [{"s": "maybe"}]
        with pytest.raises(Exception):
            session.createDataFrame(rows).select(
                F.col("s").cast(T.BooleanType()).alias("v")
            ).collect()

    def test_string_to_bool_invalid_try_returns_null(self, session, F, T, spark):
        rows_tuples = [("true",), ("maybe",), ("",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").try_cast(SparkBooleanType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").try_cast(T.BooleanType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_int_to_bool(self, session, F, T, spark):
        rows_tuples = [(1,), (0,), (-3,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkIntegerType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkBooleanType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(T.BooleanType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_double_to_bool(self, session, F, T, spark):
        rows_tuples = [(1.5,), (0.0,), (-3.2,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkDoubleType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkBooleanType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(T.BooleanType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# To-string casts
# -----------------------------------------------------------------------------


class TestToStringCasts:
    def test_int_to_string(self, session, F, T, spark):
        rows_tuples = [(1,), (0,), (-3,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkIntegerType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkStringType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(T.StringType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_double_to_string(self, session, F, T, spark):
        # Spark formats 1.5 -> '1.5', 0.0 -> '0.0'.
        rows_tuples = [(1.5,), (0.0,), (-3.25,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkDoubleType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkStringType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(T.StringType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_bool_to_string(self, session, F, T, spark):
        # Spark: True -> 'true', False -> 'false'.
        rows_tuples = [(True,), (False,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkBooleanType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkStringType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(T.StringType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# Date / timestamp casts
# -----------------------------------------------------------------------------


class TestDateCasts:
    def test_string_to_date_valid(self, session, F, T, spark):
        rows_tuples = [("2024-01-15",), ("2024-1-5",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkDateType()).alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").cast(T.DateType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_string_to_date_invalid_strict_raises(self, session, F, T):
        rows = [{"s": "not-a-date"}]
        with pytest.raises(Exception):
            session.createDataFrame(rows).select(
                F.col("s").cast(T.DateType()).alias("v")
            ).collect()

    def test_string_to_date_invalid_try_returns_null(self, session, F, T, spark):
        rows_tuples = [("2024-01-15",), ("not-a-date",), ("2024/01/15",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").try_cast(SparkDateType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").try_cast(T.DateType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)


class TestTimestampCasts:
    def test_string_to_timestamp_via_string_roundtrip(self, session, F, T, spark):
        """Compare via cast back to string to avoid JVM-vs-driver timezone differences.

        Spark stores TS as UTC and renders with ``spark.sql.session.timeZone`` (=UTC).
        ``pythondf`` stores naive datetimes; we round-trip both through string to align.
        """
        rows_tuples = [
            ("2024-01-15 10:30:45",),
            ("2024-01-15 10:30:45.123456",),
            ("2024-01-15T10:30:45",),
            (None,),
        ]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkTimestampType()).cast(SparkStringType()).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(T.TimestampType()).cast(T.StringType()).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_string_to_timestamp_invalid_strict_raises(self, session, F, T):
        rows = [{"s": "not-a-timestamp"}]
        with pytest.raises(Exception):
            session.createDataFrame(rows).select(
                F.col("s").cast(T.TimestampType()).alias("v")
            ).collect()


# -----------------------------------------------------------------------------
# Decimal casts
# -----------------------------------------------------------------------------


class TestDecimalCasts:
    def test_int_to_decimal(self, session, F, T, spark):
        rows_tuples = [(5,), (-3,), (0,), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkIntegerType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkDecimalType(10, 2)).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(T.DecimalType(10, 2)).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_string_to_decimal_valid(self, session, F, T, spark):
        rows_tuples = [("5.25",), ("-1.50",), ("0",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast(SparkDecimalType(10, 2)).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").cast(T.DecimalType(10, 2)).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_string_to_decimal_invalid_strict_raises(self, session, F, T):
        rows = [{"s": "abc"}]
        with pytest.raises(Exception):
            session.createDataFrame(rows).select(
                F.col("s").cast(T.DecimalType(10, 2)).alias("v")
            ).collect()

    def test_string_to_decimal_invalid_try_returns_null(self, session, F, T, spark):
        rows_tuples = [("5.25",), ("abc",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").try_cast(SparkDecimalType(10, 2)).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").try_cast(T.DecimalType(10, 2)).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_decimal_precision_overflow_try_returns_null(self, session, F, T, spark):
        rows_tuples = [("12345.67",), ("5.25",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").try_cast(SparkDecimalType(4, 2)).alias("v")
        )
        df = session.createDataFrame(rows).select(
            F.col("s").try_cast(T.DecimalType(4, 2)).alias("v")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_decimal_precision_overflow_strict_raises(self, session, F, T):
        rows = [{"s": "12345.67"}]
        with pytest.raises(Exception):
            session.createDataFrame(rows).select(
                F.col("s").cast(T.DecimalType(4, 2)).alias("v")
            ).collect()


# -----------------------------------------------------------------------------
# cast() accepts both DataType instances AND string type names
# -----------------------------------------------------------------------------


class TestCastAcceptsStringTypeName:
    def test_cast_accepts_string_type_name(self, session, F, spark):
        rows_tuples = [("123",), ("-7",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast("int").alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").cast("int").alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_try_cast_accepts_string_type_name(self, session, F, spark):
        rows_tuples = [("1.5",), ("abc",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").try_cast("double").alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").try_cast("double").alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_cast_accepts_decimal_string_type_name(self, session, F, spark):
        rows_tuples = [("5.25",), (None,)]
        rows = [{"s": t[0]} for t in rows_tuples]
        schema = _schema(SparkStringType(), "s")
        spark_df = spark.createDataFrame(rows_tuples, schema=schema).select(
            SF.col("s").cast("decimal(10,2)").alias("v")
        )
        df = session.createDataFrame(rows).select(F.col("s").cast("decimal(10,2)").alias("v"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)
