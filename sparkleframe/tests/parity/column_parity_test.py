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


def _all_comparisons(left, right):
    return [op_func(left, right).alias(f"r_{i}") for i, (_, op_func) in enumerate(_COMPARISON_OPS)]


@pytest.mark.feature("column.comparison.with_nulls")
@pytest.mark.parametrize(
    "spark_type, values_a, values_b",
    [
        (SparkIntegerType(), [1, None, None, 3], [None, 2, None, 3]),
        (SparkDoubleType(), [1.5, None, None], [None, 2.5, None]),
        (SparkStringType(), ["a", None, None], [None, "b", None]),
    ],
    ids=["int", "double", "string"],
)
def test_comparison_with_nulls(engine, spark, spark_type, values_a, values_b):
    # A null operand makes every comparison null — including ``null == null``.
    F = engine.functions
    schema = SparkStructType([SparkStructField("a", spark_type), SparkStructField("b", spark_type)])
    rows = list(zip(values_a, values_b))
    actual = engine.build_df(rows, schema).select(
        *_all_comparisons(F.col("a"), F.col("b")), (F.col("a") == F.lit(None)).alias("vs_null_lit")
    )
    expected = spark.createDataFrame(rows, schema).select(
        *_all_comparisons(SF.col("a"), SF.col("b")), (SF.col("a") == SF.lit(None)).alias("vs_null_lit")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.comparison.nan_semantics")
@pytest.mark.parametrize("spark_type", [SparkDoubleType(), SparkFloatType()], ids=["double", "float"])
def test_comparison_nan_semantics(engine, spark, spark_type):
    # Spark: NaN = NaN is true and NaN sorts above every other value (even +inf); 0.0 = -0.0.
    F = engine.functions
    nan, inf = float("nan"), float("inf")
    schema = SparkStructType([SparkStructField("a", spark_type), SparkStructField("b", spark_type)])
    rows = [(nan, nan), (nan, 1.0), (1.0, nan), (inf, nan), (nan, -inf), (0.0, -0.0), (2.0, 1.0)]
    actual = engine.build_df(rows, schema).select(*_all_comparisons(F.col("a"), F.col("b")))
    expected = spark.createDataFrame(rows, schema).select(*_all_comparisons(SF.col("a"), SF.col("b")))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.comparison.nested_nulls_and_nan")
@pytest.mark.parametrize(
    "spark_type, values_a, values_b",
    [
        (
            SparkArrayType(SparkIntegerType()),
            [[1, None], [None], [1, 2], [], [None, 5]],
            [[1, None], [1], [1], [None], [None, 4]],
        ),
        (SparkArrayType(SparkDoubleType()), [[float("nan")], [float("nan")], [1.0]], [[float("nan")], [1.0], [2.0]]),
        (
            SparkStructType([SparkStructField("x", SparkIntegerType()), SparkStructField("y", SparkIntegerType())]),
            [(1, None), (None, 1), (2, 1)],
            [{"x": 1, "y": None}, {"x": 1, "y": 1}, {"x": 1, "y": 9}],  # dict-shaped struct cells
        ),
    ],
    ids=["array_int", "array_double", "struct"],
)
def test_comparison_nested_nulls_and_nan(engine, spark, spark_type, values_a, values_b):
    # Inside arrays/structs a null element sorts first and equals another null; NaN equals NaN.
    F = engine.functions
    schema = SparkStructType([SparkStructField("a", spark_type), SparkStructField("b", spark_type)])
    rows = list(zip(values_a, values_b))
    actual = engine.build_df(rows, schema).select(*_all_comparisons(F.col("a"), F.col("b")))
    expected = spark.createDataFrame(rows, schema).select(*_all_comparisons(SF.col("a"), SF.col("b")))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.comparison.literals")
def test_comparison_with_literals(engine, spark):
    # Python literals are typed (int, float, str, None, date, datetime, Decimal) then coerced like columns.
    F = engine.functions
    schema = SparkStructType(
        [
            SparkStructField("i", SparkIntegerType()),
            SparkStructField("s", SparkStringType()),
            SparkStructField("d", SparkDateType()),
            SparkStructField("ts", SparkTimestampType()),
            SparkStructField("dec", SparkDecimalType(10, 2)),
        ]
    )
    rows = [
        (1, "a", date(2024, 1, 1), datetime(2024, 1, 1, 12, 0), Decimal("1.50")),
        (2, "b", date(2024, 6, 15), datetime(2024, 6, 15, 8, 30), Decimal("2.25")),
        (3, "c", date(2025, 1, 1), datetime(2025, 1, 1, 0, 0), Decimal("3.00")),
    ]

    def exprs(M):
        return [
            (M.col("i") == 2).alias("int_eq"),
            (M.col("i") < 2.5).alias("int_lt_float"),
            (M.col("i") >= "2").alias("int_ge_str"),
            (M.col("i") == M.lit(None)).alias("int_eq_null"),
            (5 > M.col("i")).alias("reflected"),
            (M.col("s") > "a").alias("str_gt"),
            (M.col("d") >= M.lit(date(2024, 6, 15))).alias("date_ge"),
            (M.col("d") < "2024-06-15").alias("date_lt_str"),
            (M.col("ts") > M.lit(datetime(2024, 6, 15, 8, 0))).alias("ts_gt"),
            (M.col("ts") >= M.lit(date(2024, 6, 15))).alias("ts_ge_date"),
            (M.col("dec") <= M.lit(Decimal("2.25"))).alias("dec_le"),
            (M.col("dec") == 3).alias("dec_eq_int"),
        ]

    actual = engine.build_df(rows, schema).select(*exprs(F))
    expected = spark.createDataFrame(rows, schema).select(*exprs(SF))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.comparison.string_implicit_cast_valid")
@pytest.mark.parametrize(
    "other_type, strings, others",
    [
        (SparkLongType(), [" 7 ", "+5", "-3", "\t9\n"], [7, 5, 2, 10]),
        (
            SparkDoubleType(),
            ["1e2", "Infinity", "NaN", ".5", "1d", "-inf", "0x1p3"],
            [100.0, 1.0, float("nan"), 0.5, 2.0, 0.0, 8.0],
        ),
        (SparkBooleanType(), ["yes", " F ", "1", "n", "TRUE"], [True, False, False, False, True]),
        (
            SparkDateType(),
            ["2024-01-15", "2024-1-5", "2024", "2024-01-15T10:00", "2024-01-15 10:00:00", "+2024-03"],
            [
                date(2024, 1, 15),
                date(2024, 1, 5),
                date(2024, 1, 1),
                date(2024, 1, 16),
                date(2024, 1, 14),
                date(2024, 3, 1),
            ],
        ),
        (
            SparkTimestampType(),
            [
                "2024-01-15 10:30:45",
                "2024-01-15T10:30:45",
                "2024-01-15",
                "2024-01",
                "2024-01-15 10:30:45.1234567",
                "2024-01-15 10",
            ],
            [
                datetime(2024, 1, 15, 10, 30, 45),
                datetime(2024, 1, 15, 10, 30, 46),
                datetime(2024, 1, 15),
                datetime(2023, 12, 31),
                datetime(2024, 1, 15, 10, 30, 45, 123456),
                datetime(2024, 1, 15, 10),
            ],
        ),
        (SparkBinaryType(), ["ab", "é", ""], [b"ab", b"a", b""]),
    ],
    ids=["long", "double", "boolean", "date", "timestamp", "binary"],
)
def test_comparison_string_implicit_cast_valid(engine, spark, other_type, strings, others):
    # Spark casts the string side to the other operand's type (bigint for integrals) with its own
    # parsing rules: whitespace trimming, special double literals, partial dates, 'T' separators...
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkStringType()), SparkStructField("o", other_type)])
    rows = list(zip(strings, others))
    actual = engine.build_df(rows, schema).select(*_all_comparisons(F.col("s"), F.col("o")))
    expected = spark.createDataFrame(rows, schema).select(*_all_comparisons(SF.col("s"), SF.col("o")))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.comparison.string_implicit_cast_malformed_raises")
@pytest.mark.parametrize(
    "other_type, other_value, bad_string",
    [
        (SparkLongType(), 1, "1.0"),
        (SparkIntegerType(), 1, "1_0"),
        (SparkLongType(), 1, "9223372036854775808"),
        (SparkDoubleType(), 1.0, "1_0"),
        (SparkBooleanType(), True, "maybe"),
        (SparkDateType(), date(2024, 1, 1), "2024-02-30"),
        (SparkDateType(), date(2024, 1, 1), "2024/01/15"),
        (SparkTimestampType(), datetime(2024, 1, 1), "2024-01-15 25:00:00"),
        (SparkTimestampType(), datetime(2024, 1, 1), "2024-01-15abc"),
    ],
    ids=[
        "long_decimal",
        "int_underscore",
        "long_overflow",
        "double",
        "boolean",
        "date",
        "date_slashes",
        "ts",
        "ts_junk",
    ],
)
def test_comparison_string_implicit_cast_malformed_raises(engine, spark, other_type, other_value, bad_string):
    # ANSI: the implicit cast is strict, so a malformed string fails the query instead of comparing null.
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkStringType()), SparkStructField("o", other_type)])
    rows = [(bad_string, other_value)]
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        spark.createDataFrame(rows, schema).select(SF.col("s") == SF.col("o")).collect()
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        engine.to_records(engine.build_df(rows, schema).select((F.col("s") == F.col("o")).alias("r")))


# ----------------------------------------------------------------------------- #
# Logical (and / or / not) parity
# ----------------------------------------------------------------------------- #

_BOOLEAN_PAIRS = list(itertools.product([True, False, None], repeat=2))


@pytest.mark.feature("column.logical.three_valued")
def test_logical_three_valued_truth_table(engine, spark):
    # SQL three-valued logic: null AND false is false, null OR true is true, NOT null is null.
    F = engine.functions
    schema = SparkStructType([SparkStructField("a", SparkBooleanType()), SparkStructField("b", SparkBooleanType())])

    def exprs(M):
        return [
            M.col("a"),
            M.col("b"),
            (M.col("a") & M.col("b")).alias("and_"),
            (M.col("a") | M.col("b")).alias("or_"),
            (~M.col("a")).alias("not_"),
        ]

    actual = engine.build_df(_BOOLEAN_PAIRS, schema).select(*exprs(F))
    expected = spark.createDataFrame(_BOOLEAN_PAIRS, schema).select(*exprs(SF))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.logical.string_and_null_operands")
def test_logical_string_and_null_operands(engine, spark):
    # Spark casts string operands to boolean and types a null literal as boolean.
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkStringType()), SparkStructField("b", SparkBooleanType())])
    rows = [("true", True), ("no", True), (" Y ", False), ("0", None), (None, True)]

    def exprs(M):
        return [
            (M.col("s") & M.col("b")).alias("str_and"),
            (M.col("b") | M.col("s")).alias("str_or"),
            (~M.col("s")).alias("not_str"),
            (M.col("b") & M.lit(None)).alias("and_null"),
            (M.col("b") | M.lit(None)).alias("or_null"),
            (M.col("b") | M.lit("yes")).alias("or_str_lit"),
            (M.col("b") & True).alias("and_true"),
        ]

    actual = engine.build_df(rows, schema).select(*exprs(F))
    expected = spark.createDataFrame(rows, schema).select(*exprs(SF))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.logical.malformed_string_raises")
@pytest.mark.parametrize("build", [lambda M: M.col("b") & M.col("s"), lambda M: ~M.col("s")], ids=["and", "not"])
def test_logical_malformed_string_raises(engine, spark, build):
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkStringType()), SparkStructField("b", SparkBooleanType())])
    rows = [("true", True), ("maybe", True)]
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        spark.createDataFrame(rows, schema).select(build(SF)).collect()
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        engine.to_records(engine.build_df(rows, schema).select(build(F).alias("r")))


@pytest.mark.feature("column.logical.non_boolean_operand_raises")
@pytest.mark.parametrize(
    "build",
    [
        lambda M: M.col("i") & M.col("b"),
        lambda M: M.col("b") | M.col("d"),
        lambda M: M.col("i") | M.col("i"),
        lambda M: ~M.col("i"),
    ],
    ids=["int_and_bool", "bool_or_double", "int_or_int", "not_int"],
)
def test_logical_non_boolean_operand_raises(engine, spark, build):
    # Numeric operands are an analysis error in Spark (DATATYPE_MISMATCH), not a cast.
    F = engine.functions
    schema = SparkStructType(
        [
            SparkStructField("i", SparkIntegerType()),
            SparkStructField("d", SparkDoubleType()),
            SparkStructField("b", SparkBooleanType()),
        ]
    )
    rows = [(1, 1.0, True)]
    with pytest.raises(Exception, match="DATATYPE_MISMATCH"):
        spark.createDataFrame(rows, schema).select(build(SF)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(build(F).alias("r")))


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


# ----------------------------------------------------------------------------- #
# startswith parity
# ----------------------------------------------------------------------------- #

_STARTSWITH_SCHEMA = SparkStructType([SparkStructField("text", SparkStringType())])


@pytest.mark.feature("column.startswith.literal_prefix")
@pytest.mark.parametrize(
    "values, prefix",
    [
        (["apple", "banana", "apricot", None], "ap"),  # basic prefix
        (["Spark", "spark", "SPARK", None], "spark"),  # case sensitivity
        (["ab", "a", "", None], "ab"),  # prefix longer than value, empty value
        (["abc", "", None], ""),  # empty prefix matches every non-null value
        (["a.b", "ab", "a*b", None], "a."),  # literal special char (not regex)
        (["xab", "ab", None], "b"),  # substring that is not a prefix
    ],
)
def test_startswith_literal_prefix(engine, spark, values, prefix):
    F = engine.functions
    rows = [(v,) for v in values]
    expected = spark.createDataFrame(rows, _STARTSWITH_SCHEMA).select(SF.col("text").startswith(prefix).alias("v"))
    actual = engine.build_df(rows, _STARTSWITH_SCHEMA).select(F.col("text").startswith(prefix).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.startswith.column_prefix")
def test_startswith_column_prefix(engine, spark):
    F = engine.functions
    schema = SparkStructType(
        [SparkStructField("text", SparkStringType()), SparkStructField("prefix", SparkStringType())]
    )
    rows = [("apple", "ap"), ("apple", "pl"), ("apple", None), (None, "ap"), ("", "")]
    expected = spark.createDataFrame(rows, schema).select(SF.col("text").startswith(SF.col("prefix")).alias("v"))
    actual = engine.build_df(rows, schema).select(F.col("text").startswith(F.col("prefix")).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.startswith.null_prefix_returns_null")
def test_startswith_null_prefix_returns_null(engine, spark):
    """A ``None`` prefix is a null literal in Spark: every row is null, nothing raises."""
    F = engine.functions
    rows = [("apple",), ("",), (None,)]
    expected = spark.createDataFrame(rows, _STARTSWITH_SCHEMA).select(SF.col("text").startswith(None).alias("v"))
    actual = engine.build_df(rows, _STARTSWITH_SCHEMA).select(F.col("text").startswith(None).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.startswith.non_string_prefix_raises")
@pytest.mark.parametrize("prefix", [1, ["a"]])
def test_startswith_non_string_prefix_raises(engine, spark, prefix):
    """Spark rejects a non-string, non-Column prefix when the expression is built."""
    F = engine.functions
    with pytest.raises(Exception):
        spark.createDataFrame([("apple",)], _STARTSWITH_SCHEMA).select(SF.col("text").startswith(prefix))
    with pytest.raises(Exception):
        engine.build_df([("apple",)], _STARTSWITH_SCHEMA).select(F.col("text").startswith(prefix))
