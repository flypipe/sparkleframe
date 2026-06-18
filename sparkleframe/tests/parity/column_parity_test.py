"""Shared Spark-parity tests for ``Column`` arithmetic, comparison and cast.

Lifted from ``polarsdf/column_test.py`` and converted to the engine-parametrized
harness: each test builds the actual frame via ``engine.build_df`` and the
expected frame via real Spark, then compares with ``assert_matches_spark``. Every
expression is built from ``engine.functions`` so it runs on the active engine.

Known per-parametrization Polars gaps (documented in ``docs/known_gaps.md``) are
preserved via inline ``pytest.xfail`` guarded on the polars engine, so they keep
biting Polars without pre-judging the Python engine.
"""

import itertools
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import BinaryType as SparkBinaryType
from pyspark.sql.types import BooleanType as SparkBooleanType
from pyspark.sql.types import ByteType as SparkByteType
from pyspark.sql.types import DateType as SparkDateType
from pyspark.sql.types import DecimalType as SparkDecimalType
from pyspark.sql.types import DoubleType as SparkDoubleType
from pyspark.sql.types import FloatType as SparkFloatType
from pyspark.sql.types import IntegerType as SparkIntegerType
from pyspark.sql.types import LongType as SparkLongType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import ShortType as SparkShortType
from pyspark.sql.types import StringType as SparkStringType
from pyspark.sql.types import StructField as SparkStructField
from pyspark.sql.types import StructType as SparkStructType
from pyspark.sql.types import TimestampType as SparkTimestampType

from sparkleframe.tests.parity.oracle import assert_matches_spark

_ARITHMETIC_TYPE_FIXTURES = [
    ("byte", SparkByteType(), [2, 5, 10], [1, 3, 4]),
    ("short", SparkShortType(), [2, 5, 10], [1, 3, 4]),
    ("int", SparkIntegerType(), [2, 5, 10], [1, 3, 4]),
    ("long", SparkLongType(), [2, 5, 10], [1, 3, 4]),
    ("float", SparkFloatType(), [1.5, 2.5, 3.5], [0.5, 1.0, 2.0]),
    ("double", SparkDoubleType(), [1.5, 2.5, 3.5], [0.5, 1.0, 2.0]),
    ("string", SparkStringType(), ["1.5", "2.5", "3.5"], ["0.5", "1.0", "2.0"]),
    ("boolean", SparkBooleanType(), [True, False, True], [False, True, False]),
    (
        "date",
        SparkDateType(),
        [date(2024, 1, 1), date(2024, 6, 15), date(2025, 1, 1)],
        [date(2023, 1, 1), date(2024, 3, 1), date(2024, 12, 31)],
    ),
    (
        "timestamp",
        SparkTimestampType(),
        [datetime(2024, 1, 1), datetime(2024, 6, 15, 12, 0), datetime(2025, 1, 1)],
        [datetime(2023, 1, 1), datetime(2024, 3, 1, 8, 0), datetime(2024, 12, 31)],
    ),
    (
        "decimal",
        SparkDecimalType(10, 2),
        [Decimal("1.50"), Decimal("2.50"), Decimal("3.50")],
        [Decimal("0.50"), Decimal("1.00"), Decimal("2.00")],
    ),
    ("binary", SparkBinaryType(), [b"\x01\x02", b"\x03\x04", b"\x05\x06"], [b"\x07\x08", b"\x09\x0a", b"\x0b\x0c"]),
    ("array_int", SparkArrayType(SparkIntegerType()), [[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]),
    (
        "map_str_int",
        SparkMapType(SparkStringType(), SparkIntegerType()),
        [{"a": 1}, {"b": 2}, {"c": 3}],
        [{"d": 4}, {"e": 5}, {"f": 6}],
    ),
    (
        "struct",
        SparkStructType([SparkStructField("x", SparkIntegerType()), SparkStructField("y", SparkIntegerType())]),
        [(1, 2), (3, 4), (5, 6)],
        [(7, 8), (9, 10), (11, 12)],
    ),
]

_ARITHMETIC_OPS = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
    ("/", lambda a, b: a / b),
    ("**", lambda a, b: a**b),
]

_COMPARISON_OPS = [
    ("==", lambda a, b: a == b),
    ("!=", lambda a, b: a != b),
    ("<", lambda a, b: a < b),
    ("<=", lambda a, b: a <= b),
    (">", lambda a, b: a > b),
    (">=", lambda a, b: a >= b),
]

_MIXED_TYPE_PAIRS = [
    (f"{left[0]}_x_{right[0]}", left[1], left[2], right[1], right[2])
    for left, right in itertools.combinations(_ARITHMETIC_TYPE_FIXTURES, 2)
]


# Known parity gaps — see docs/known_gaps.md for full details and fix paths.

_MIXED_ARITH_STRING_COERCION_PAIRS = {
    "float_x_string",
    "double_x_string",
    "string_x_decimal",
}
_MIXED_ARITH_STRING_OPS = {"+", "-", "*", "/"}

_MIXED_ARITH_DATE_INT_PAIRS = {
    "byte_x_date",
    "short_x_date",
    "int_x_date",
}


def _mixed_pair_xfail_reason(pair_label: str, op_name: str) -> Optional[str]:
    """Return an xfail reason for known Polars gaps, or ``None`` if the test should run.

    See ``docs/known_gaps.md`` for the full catalogue of gaps and possible fix paths.
    """
    left, _, right = pair_label.partition("_x_")
    nested_ordering_ops = {"<", "<=", ">", ">="}
    nested_labels = {"array_int", "struct"}
    if op_name in nested_ordering_ops and (left in nested_labels or right in nested_labels):
        return "Polars does not support <, <=, >, >= on List / Struct dtypes"
    if "map_str_int" in (left, right):
        return "Polars stores maps as List(Struct) -- indistinguishable from arrays"
    if pair_label in _MIXED_ARITH_STRING_COERCION_PAIRS and op_name in _MIXED_ARITH_STRING_OPS:
        return "dtype unknown at build time -- see docs/known_gaps.md 'Mixed-type column arithmetic'"
    if pair_label in _MIXED_ARITH_DATE_INT_PAIRS and op_name == "+":
        return "dtype unknown at build time -- see docs/known_gaps.md 'Mixed-type column arithmetic'"
    return None


_MAP_COMPARISON_GAPS = {
    # Polars stores maps as List(Struct([key, value])) -- indistinguishable from arrays at dtype level.
    # Spark rejects all comparisons on maps, but sparkleframe can't detect map vs array.
    ("map_str_int", "=="),
    ("map_str_int", "!="),
    ("map_str_int", "<"),
    ("map_str_int", "<="),
    ("map_str_int", ">"),
    ("map_str_int", ">="),
}


_COMPLEX_ORDERING_GAPS = {
    # Polars doesn't implement <, <=, >, >= on List or Struct dtypes (lexicographic
    # comparison is not built in), so we can't match Spark without re-implementing it
    # manually via explode + element-wise compare.
    ("array_int", "<"),
    ("array_int", "<="),
    ("array_int", ">"),
    ("array_int", ">="),
    ("struct", "<"),
    ("struct", "<="),
    ("struct", ">"),
    ("struct", ">="),
}


def _assert_op_matches_or_both_raise(engine, spark, schema, rows, op_func):
    """Shared driver: when Spark raises, the engine must raise on evaluation too;
    otherwise the engine result must equal Spark's."""
    F = engine.functions

    spark_raised = False
    try:
        spark_result = spark.createDataFrame(rows, schema).select(op_func(SF.col("a"), SF.col("b")).alias("result"))
        spark_result.collect()
    except Exception:
        spark_raised = True

    if spark_raised:
        with pytest.raises(Exception):
            engine.to_records(engine.build_df(rows, schema).select(op_func(F.col("a"), F.col("b")).alias("result")))
    else:
        actual = engine.build_df(rows, schema).select(op_func(F.col("a"), F.col("b")).alias("result"))
        assert_matches_spark(actual, spark_result, engine)


# ----------------------------------------------------------------------------- #
# Arithmetic / comparison parity
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("column.arithmetic.col_col")
@pytest.mark.parametrize("dtype_label, spark_type, values_a, values_b", _ARITHMETIC_TYPE_FIXTURES)
@pytest.mark.parametrize("op_name, op_func", _ARITHMETIC_OPS)
def test_arithmetic_col_col(engine, spark, dtype_label, spark_type, values_a, values_b, op_name, op_func):
    schema = SparkStructType([SparkStructField("a", spark_type), SparkStructField("b", spark_type)])
    rows = list(zip(values_a, values_b))
    _assert_op_matches_or_both_raise(engine, spark, schema, rows, op_func)


@pytest.mark.feature("column.comparison.col_col")
@pytest.mark.parametrize("dtype_label, spark_type, values_a, values_b", _ARITHMETIC_TYPE_FIXTURES)
@pytest.mark.parametrize("op_name, op_func", _COMPARISON_OPS)
def test_comparison_col_col(engine, spark, dtype_label, spark_type, values_a, values_b, op_name, op_func):
    if engine.name == "polars" and (dtype_label, op_name) in _MAP_COMPARISON_GAPS:
        pytest.xfail("Polars stores maps as List(Struct) -- indistinguishable from arrays")
    if engine.name == "polars" and (dtype_label, op_name) in _COMPLEX_ORDERING_GAPS:
        pytest.xfail("Polars does not support <, <=, >, >= on List / Struct dtypes")
    schema = SparkStructType([SparkStructField("a", spark_type), SparkStructField("b", spark_type)])
    rows = list(zip(values_a, values_b))
    _assert_op_matches_or_both_raise(engine, spark, schema, rows, op_func)


@pytest.mark.feature("column.arithmetic.with_nulls")
@pytest.mark.parametrize(
    "dtype_label, spark_type, values_a, values_b",
    [
        ("int", SparkIntegerType(), [1, 2, 3, None], [4, None, 6, None]),
        ("long", SparkLongType(), [1, 2, 3, None], [4, None, 6, None]),
        ("double", SparkDoubleType(), [1.0, 2.0, None, None], [None, 5.0, 6.0, None]),
        ("float", SparkFloatType(), [1.0, 2.0, None, None], [None, 5.0, 6.0, None]),
    ],
)
@pytest.mark.parametrize("op_name, op_func", _ARITHMETIC_OPS)
def test_arithmetic_with_nulls(engine, spark, dtype_label, spark_type, values_a, values_b, op_name, op_func):
    F = engine.functions
    schema = SparkStructType(
        [
            SparkStructField("a", spark_type, nullable=True),
            SparkStructField("b", spark_type, nullable=True),
        ]
    )
    rows = list(zip(values_a, values_b))

    expected = spark.createDataFrame(rows, schema).select(op_func(SF.col("a"), SF.col("b")).alias("result"))
    actual = engine.build_df(rows, schema).select(op_func(F.col("a"), F.col("b")).alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.arithmetic.literal_col")
@pytest.mark.parametrize(
    "dtype_label, spark_type, values",
    [
        ("int", SparkIntegerType(), [2, 5, 10]),
        ("long", SparkLongType(), [2, 5, 10]),
        ("float", SparkFloatType(), [1.5, 2.5, 3.5]),
        ("double", SparkDoubleType(), [1.5, 2.5, 3.5]),
    ],
)
@pytest.mark.parametrize(
    "op_name, op_func",
    [
        ("radd", lambda v: 10 + v),
        ("rsub", lambda v: 10 - v),
        ("rmul", lambda v: 10 * v),
        ("rtruediv", lambda v: 10 / v),
    ],
)
def test_arithmetic_literal_col(engine, spark, dtype_label, spark_type, values, op_name, op_func):
    F = engine.functions
    schema = SparkStructType([SparkStructField("a", spark_type)])
    rows = [(v,) for v in values]

    expected = spark.createDataFrame(rows, schema).select(op_func(SF.col("a")).alias("result"))
    actual = engine.build_df(rows, schema).select(op_func(F.col("a")).alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.arithmetic.col_col_mixed_types")
@pytest.mark.parametrize(
    "pair_label, left_spark_type, left_values, right_spark_type, right_values",
    _MIXED_TYPE_PAIRS,
    ids=[p[0] for p in _MIXED_TYPE_PAIRS],
)
@pytest.mark.parametrize("op_name, op_func", _ARITHMETIC_OPS)
def test_arithmetic_col_col_mixed_types(
    engine, spark, pair_label, left_spark_type, left_values, right_spark_type, right_values, op_name, op_func
):
    reason = _mixed_pair_xfail_reason(pair_label, op_name)
    if engine.name == "polars" and reason:
        pytest.xfail(reason)
    schema = SparkStructType([SparkStructField("a", left_spark_type), SparkStructField("b", right_spark_type)])
    rows = list(zip(left_values, right_values))
    _assert_op_matches_or_both_raise(engine, spark, schema, rows, op_func)


@pytest.mark.feature("column.comparison.col_col_mixed_types")
@pytest.mark.parametrize(
    "pair_label, left_spark_type, left_values, right_spark_type, right_values",
    _MIXED_TYPE_PAIRS,
    ids=[p[0] for p in _MIXED_TYPE_PAIRS],
)
@pytest.mark.parametrize("op_name, op_func", _COMPARISON_OPS)
def test_comparison_col_col_mixed_types(
    engine, spark, pair_label, left_spark_type, left_values, right_spark_type, right_values, op_name, op_func
):
    reason = _mixed_pair_xfail_reason(pair_label, op_name)
    if engine.name == "polars" and reason:
        pytest.xfail(reason)
    schema = SparkStructType([SparkStructField("a", left_spark_type), SparkStructField("b", right_spark_type)])
    rows = list(zip(left_values, right_values))
    _assert_op_matches_or_both_raise(engine, spark, schema, rows, op_func)


# ----------------------------------------------------------------------------- #
# Cast / try_cast parity
#
# Spark 4 enables ``spark.sql.ansi.enabled`` by default, so ``cast`` of a malformed
# value raises ``CAST_INVALID_INPUT`` while ``try_cast`` returns ``NULL``; the engine
# must match both. Strict ``cast`` targets use ``engine.dtype(<spark type>)`` to stay
# engine-agnostic (the engine's own DataType); ``try_cast`` accepts type-name strings.
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("column.cast.string_to_numeric_valid")
@pytest.mark.parametrize(
    "spark_dtype",
    [SparkIntegerType(), SparkLongType(), SparkDoubleType(), SparkFloatType()],
)
def test_cast_string_to_numeric_valid(engine, spark, spark_dtype):
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkStringType())])
    rows = [("123",), ("-7",), (None,)]
    expected = spark.createDataFrame(rows, schema).select(SF.col("s").cast(spark_dtype).alias("v"))
    actual = engine.build_df(rows, schema).select(F.col("s").cast(engine.dtype(spark_dtype)).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.try_cast.string_to_numeric_invalid_null")
@pytest.mark.parametrize(
    "spark_dtype, type_name",
    [(SparkIntegerType(), "int"), (SparkLongType(), "long"), (SparkDoubleType(), "double")],
)
def test_try_cast_string_to_numeric_invalid_returns_null(engine, spark, spark_dtype, type_name):
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkStringType())])
    rows = [("123",), ("Bob",), (None,)]
    expected = spark.createDataFrame(rows, schema).select(SF.col("s").try_cast(spark_dtype).alias("v"))
    actual = engine.build_df(rows, schema).select(F.col("s").try_cast(type_name).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.cast.string_to_numeric_invalid_raises")
@pytest.mark.parametrize(
    "spark_dtype",
    [SparkIntegerType(), SparkLongType(), SparkDoubleType()],
)
def test_cast_string_to_numeric_invalid_raises_like_spark_ansi(engine, spark, spark_dtype):
    """Both Spark 4 (ANSI default) and the engine raise when an invalid string can't be cast."""
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkStringType())])
    rows = [("123",), ("Bob",), (None,)]
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.col("s").cast(spark_dtype).alias("v")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.col("s").cast(engine.dtype(spark_dtype)).alias("v")))


@pytest.mark.feature("column.cast.string_to_boolean_invalid_raises")
def test_cast_string_to_boolean_invalid_raises_like_spark_ansi(engine, spark):
    """Same parity for boolean casts: Spark and the engine both raise on 'maybe'."""
    F = engine.functions
    bool_type = SparkBooleanType()
    schema = SparkStructType([SparkStructField("s", SparkStringType())])
    rows = [("true",), ("maybe",), (None,)]
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.col("s").cast(bool_type).alias("v")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.col("s").cast(engine.dtype(bool_type)).alias("v")))


@pytest.mark.feature("column.cast.int_to_boolean")
def test_cast_int_to_boolean_parity(engine, spark):
    """Spark casts nonzero int -> True, 0 -> False; no raise. The engine must match."""
    F = engine.functions
    bool_type = SparkBooleanType()
    schema = SparkStructType([SparkStructField("s", SparkIntegerType())])
    rows = [(1,), (2,), (0,), (-1,), (None,)]
    expected = spark.createDataFrame(rows, schema).select(SF.col("s").cast(bool_type).alias("v"))
    actual = engine.build_df(rows, schema).select(F.col("s").cast(engine.dtype(bool_type)).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.cast.double_to_boolean")
def test_cast_double_to_boolean_parity(engine, spark):
    F = engine.functions
    bool_type = SparkBooleanType()
    schema = SparkStructType([SparkStructField("s", SparkDoubleType())])
    rows = [(1.5,), (0.0,), (-3.2,), (None,)]
    expected = spark.createDataFrame(rows, schema).select(SF.col("s").cast(bool_type).alias("v"))
    actual = engine.build_df(rows, schema).select(F.col("s").cast(engine.dtype(bool_type)).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.try_cast.string_to_boolean")
def test_try_cast_string_to_boolean_parity(engine, spark):
    F = engine.functions
    # Only test the literals Spark actually accepts for string -> boolean.
    schema = SparkStructType([SparkStructField("s", SparkStringType())])
    rows = [("true",), ("TRUE",), ("false",), ("FALSE",), ("t",), ("f",), ("1",), ("0",), ("y",), ("n",), (None,)]
    expected = spark.createDataFrame(rows, schema).select(SF.col("s").try_cast(SparkBooleanType()).alias("v"))
    actual = engine.build_df(rows, schema).select(F.col("s").try_cast("boolean").alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.cast.string_to_boolean_valid")
def test_cast_string_to_boolean_valid(engine, spark):
    F = engine.functions
    bool_type = SparkBooleanType()
    # Valid literals only -- avoids the ANSI raise divergence above.
    schema = SparkStructType([SparkStructField("s", SparkStringType())])
    rows = [("true",), ("false",), ("t",), ("f",), ("1",), ("0",), (None,)]
    expected = spark.createDataFrame(rows, schema).select(SF.col("s").cast(bool_type).alias("v"))
    actual = engine.build_df(rows, schema).select(F.col("s").cast(engine.dtype(bool_type)).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.cast.int_to_other")
@pytest.mark.parametrize(
    "spark_dtype",
    [SparkStringType(), SparkDoubleType(), SparkLongType()],
)
def test_cast_int_to_other_parity(engine, spark, spark_dtype):
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkIntegerType())])
    rows = [(1,), (-5,), (0,), (None,)]
    expected = spark.createDataFrame(rows, schema).select(SF.col("s").cast(spark_dtype).alias("v"))
    actual = engine.build_df(rows, schema).select(F.col("s").cast(engine.dtype(spark_dtype)).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.try_cast.inside_when_then")
def test_try_cast_inside_when_then_against_spark(engine, spark):
    """Polars evaluates all when/then branches eagerly, so a strict cast fails on
    non-matching rows. Use try_cast inside when/then as a workaround; the result
    matches PySpark's short-circuit cast behavior."""
    F = engine.functions
    schema = SparkStructType([SparkStructField("v", SparkStringType())])
    rows = [("1",), ("2",), ("N/A",)]

    expected = spark.createDataFrame(rows, schema).withColumn(
        "num",
        SF.when(SF.col("v").rlike(r"^\d+$"), SF.col("v").cast(SparkIntegerType())).otherwise(SF.lit(-1)),
    )
    actual = engine.build_df(rows, schema).withColumn(
        "num",
        F.when(F.col("v").rlike(r"^\d+$"), F.col("v").try_cast("int")).otherwise(F.lit(-1)),
    )
    assert_matches_spark(actual, expected, engine)
