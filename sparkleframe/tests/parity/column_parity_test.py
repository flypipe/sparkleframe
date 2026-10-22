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


# ----------------------------------------------------------------------------- #
# Decimal result types, decimal / timestamp casts
#
# The oracle compares decimals as floats, so wherever the decimal *scale* or the
# resolved result type is the point, these tests also compare the resolved
# ``dtypes`` with Spark's and the values rendered through a string cast.
# ----------------------------------------------------------------------------- #


def _assert_strict_raises(engine, spark, rows, schema, build, spark_match: Optional[str] = None):
    """Spark (ANSI) fails ``build`` on ``rows``; the engine must fail too once evaluated."""
    F = engine.functions
    with pytest.raises(Exception, match=spark_match):
        spark.createDataFrame(rows, schema).select(build(SF, SparkStringType())).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(build(F, engine.dtype(SparkStringType()))))


def _assert_matches_with_dtypes(engine, spark, rows, schema, build):
    """Compare values, resolved dtypes, and the string rendering of every output column."""
    F = engine.functions
    actual = engine.build_df(rows, schema).select(*build(F, engine.dtype(SparkStringType())))
    expected = spark.createDataFrame(rows, schema).select(*build(SF, SparkStringType()))
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)
    str_type = engine.dtype(SparkStringType())
    actual_str = actual.select(*[F.col(name).cast(str_type).alias(name) for name in actual.columns])
    expected_str = expected.select(*[SF.col(name).cast(SparkStringType()).alias(name) for name in expected.columns])
    assert_matches_spark(actual_str, expected_str, engine)


_DECIMAL_OPERANDS_SCHEMA = SparkStructType(
    [
        SparkStructField("a", SparkDecimalType(10, 2)),
        SparkStructField("b", SparkDecimalType(5, 3)),
        SparkStructField("i", SparkIntegerType()),
        SparkStructField("l", SparkLongType()),
    ]
)
_DECIMAL_OPERANDS_ROWS = [
    (Decimal("12345678.91"), Decimal("12.345"), 7, 3),
    (Decimal("-1.50"), Decimal("0.007"), -3, -9),
    (Decimal("0.05"), Decimal("99.999"), 0, 1),
    (None, Decimal("1.000"), None, None),
]


@pytest.mark.feature("column.arithmetic.decimal_result_type")
def test_arithmetic_decimal_result_type(engine, spark):
    # decimal(10,2) op decimal(5,3): + / - -> decimal(12,3), * -> decimal(16,5); an int operand is
    # promoted to decimal(10,0) (+ -> decimal(13,2)) and a bigint to decimal(20,0) (* -> decimal(31,2)).
    def build(M, _str_type):
        return [
            (M.col("a") + M.col("b")).alias("add"),
            (M.col("a") - M.col("b")).alias("sub"),
            (M.col("a") * M.col("b")).alias("mul"),
            (M.col("a") + M.col("i")).alias("add_int"),
            (M.col("i") - M.col("b")).alias("int_sub"),
            (M.col("a") * M.col("l")).alias("mul_long"),
        ]

    _assert_matches_with_dtypes(engine, spark, _DECIMAL_OPERANDS_ROWS, _DECIMAL_OPERANDS_SCHEMA, build)

    # Malformed: decimal(38,0) + decimal(38,0) stays decimal(38,0) and overflows -> ANSI raises.
    overflow_schema = SparkStructType([SparkStructField("x", SparkDecimalType(38, 0))])
    overflow_rows = [(Decimal(1),), (Decimal(10**38 - 1),)]
    _assert_strict_raises(
        engine,
        spark,
        overflow_rows,
        overflow_schema,
        lambda M, _t: (M.col("x") + M.col("x")).alias("r"),
        spark_match="NUMERIC_VALUE_OUT_OF_RANGE",
    )


@pytest.mark.feature("column.cast.decimal_to_string_keeps_scale")
def test_cast_decimal_to_string_keeps_scale(engine, spark):
    # Spark renders the full declared scale (``1.50``, ``0.00``, ``100.00``) in plain notation, never
    # scientific (``0E-10`` -> ``0.0000000000``).
    schema = SparkStructType(
        [
            SparkStructField("d2", SparkDecimalType(10, 2)),
            SparkStructField("d0", SparkDecimalType(5, 0)),
            SparkStructField("d10", SparkDecimalType(20, 10)),
        ]
    )
    rows = [
        (Decimal("1.50"), Decimal("7"), Decimal("0E-10")),
        (Decimal("0.00"), Decimal("-12345"), Decimal("1E-10")),
        (Decimal("-3.10"), Decimal("0"), Decimal("12.5")),
        (Decimal("100"), None, Decimal("-0.0000001")),
        (None, Decimal("1"), None),
    ]

    def build(M, str_type):
        return [M.col(c).cast(str_type).alias(c) for c in ("d2", "d0", "d10")]

    F = engine.functions
    actual = engine.build_df(rows, schema).select(*build(F, engine.dtype(SparkStringType())))
    expected = spark.createDataFrame(rows, schema).select(*build(SF, SparkStringType()))
    assert_matches_spark(actual, expected, engine)

    # Round trip: a valid string cast to decimal(10,2) and back keeps the declared scale.
    str_schema = SparkStructType([SparkStructField("s", SparkStringType())])
    str_rows = [("1.5",), (" 2 ",), ("-0.1",), (None,)]

    def round_trip(M, str_type, dec_type):
        return M.col("s").cast(dec_type).cast(str_type).alias("r")

    actual = engine.build_df(str_rows, str_schema).select(
        round_trip(F, engine.dtype(SparkStringType()), engine.dtype(SparkDecimalType(10, 2)))
    )
    expected = spark.createDataFrame(str_rows, str_schema).select(
        round_trip(SF, SparkStringType(), SparkDecimalType(10, 2))
    )
    assert_matches_spark(actual, expected, engine)

    # Malformed: a non-numeric string cannot be cast to decimal under ANSI.
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        spark.createDataFrame([("1.5x",)], str_schema).select(
            round_trip(SF, SparkStringType(), SparkDecimalType(10, 2))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df([("1.5x",)], str_schema).select(
                round_trip(F, engine.dtype(SparkStringType()), engine.dtype(SparkDecimalType(10, 2)))
            )
        )


@pytest.mark.feature("column.cast.timestamp_to_string")
def test_cast_timestamp_to_string(engine, spark):
    # Spark prints ``yyyy-MM-dd HH:mm:ss`` and only the significant fraction digits (``.1``, ``.123456``).
    F = engine.functions
    schema = SparkStructType([SparkStructField("ts", SparkTimestampType())])
    rows = [
        (datetime(2024, 1, 15, 10, 30, 45),),
        (datetime(2024, 1, 15, 10, 30, 45, 100000),),
        (datetime(2024, 1, 15, 10, 30, 45, 123000),),
        (datetime(2024, 1, 15, 10, 30, 45, 123456),),
        (datetime(2024, 1, 15, 0, 0, 0, 10),),
        (datetime(1969, 12, 31, 23, 59, 59, 999999),),
        (datetime(1, 1, 1, 0, 0),),
        (None,),
    ]
    actual = engine.build_df(rows, schema).select(F.col("ts").cast(engine.dtype(SparkStringType())).alias("s"))
    expected = spark.createDataFrame(rows, schema).select(SF.col("ts").cast(SparkStringType()).alias("s"))
    assert_matches_spark(actual, expected, engine)

    # Malformed: the reverse cast of a string that is not a timestamp raises under ANSI.
    str_schema = SparkStructType([SparkStructField("s", SparkStringType())])
    for bad in ["2024-13-01 00:00:00", "2024-01-15 10:30:45abc"]:
        with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
            spark.createDataFrame([(bad,)], str_schema).select(SF.col("s").cast(SparkTimestampType())).collect()
        with pytest.raises(Exception):
            engine.to_records(
                engine.build_df([(bad,)], str_schema).select(
                    F.col("s").cast(engine.dtype(SparkTimestampType())).alias("r")
                )
            )


_TO_DECIMAL_5_2_SOURCES = [
    (SparkDecimalType(10, 2), [Decimal("123.45"), Decimal("-999.99"), None], Decimal("1234.56")),
    (SparkIntegerType(), [999, -5, None], 1000),
    (SparkLongType(), [0, 12, None], -100000),
    (SparkDoubleType(), [999.99, -0.5, None], 1000.0),
    (SparkStringType(), ["999.99", " -1.5 ", None], "1000"),
]


@pytest.mark.feature("column.cast.to_decimal_overflow_raises")
@pytest.mark.parametrize(
    "source_type, valid, overflow",
    _TO_DECIMAL_5_2_SOURCES,
    ids=["decimal", "int", "long", "double", "string"],
)
def test_cast_to_decimal_overflow_raises(engine, spark, source_type, valid, overflow):
    # decimal(5,2) holds at most 999.99: a value that needs more integral digits raises under ANSI.
    F = engine.functions
    schema = SparkStructType([SparkStructField("x", source_type)])
    target = SparkDecimalType(5, 2)
    str_type = SparkStringType()

    rows = [(v,) for v in valid]
    actual = engine.build_df(rows, schema).select(
        F.col("x").cast(engine.dtype(target)).cast(engine.dtype(str_type)).alias("r")
    )
    expected = spark.createDataFrame(rows, schema).select(SF.col("x").cast(target).cast(str_type).alias("r"))
    assert_matches_spark(actual, expected, engine)

    bad_rows = rows + [(overflow,)]
    with pytest.raises(Exception, match="NUMERIC_VALUE_OUT_OF_RANGE"):
        spark.createDataFrame(bad_rows, schema).select(SF.col("x").cast(target)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(bad_rows, schema).select(F.col("x").cast(engine.dtype(target)).alias("r")))


@pytest.mark.feature("column.cast.to_decimal_rounds_half_up")
def test_cast_to_decimal_rounds_half_up(engine, spark):
    # Spark rounds HALF_UP (away from zero on a tie) when the target scale is smaller -- not banker's rounding.
    F = engine.functions
    schema = SparkStructType(
        [SparkStructField("s", SparkStringType()), SparkStructField("d", SparkDecimalType(10, 3))]
    )
    rows = [
        ("1.005", Decimal("1.005")),
        ("1.015", Decimal("-2.675")),
        ("-1.005", Decimal("0.125")),
        ("2.5", Decimal("2.500")),
        ("-2.5", Decimal("-0.005")),
        ("0.125", Decimal("1.004")),
        ("1.0049", Decimal("3.500")),
        (None, None),
    ]

    def build(M, str_type, dec2, dec0):
        return [
            M.col("s").cast(dec2).cast(str_type).alias("s_2"),
            M.col("s").cast(dec0).cast(str_type).alias("s_0"),
            M.col("d").cast(dec2).cast(str_type).alias("d_2"),
            M.col("d").cast(dec0).cast(str_type).alias("d_0"),
        ]

    actual = engine.build_df(rows, schema).select(
        *build(
            F,
            engine.dtype(SparkStringType()),
            engine.dtype(SparkDecimalType(10, 2)),
            engine.dtype(SparkDecimalType(10, 0)),
        )
    )
    expected = spark.createDataFrame(rows, schema).select(
        *build(SF, SparkStringType(), SparkDecimalType(10, 2), SparkDecimalType(10, 0))
    )
    assert_matches_spark(actual, expected, engine)

    # Malformed: an unparseable string raises, and rounding up past the precision overflows.
    str_schema = SparkStructType([SparkStructField("s", SparkStringType())])
    for bad, target, error in [
        ("1.0.5", SparkDecimalType(10, 2), "CAST_INVALID_INPUT"),
        ("999.995", SparkDecimalType(5, 2), "NUMERIC_VALUE_OUT_OF_RANGE"),
    ]:
        with pytest.raises(Exception, match=error):
            spark.createDataFrame([(bad,)], str_schema).select(SF.col("s").cast(target)).collect()
        with pytest.raises(Exception):
            engine.to_records(
                engine.build_df([(bad,)], str_schema).select(F.col("s").cast(engine.dtype(target)).alias("r"))
            )


# ----------------------------------------------------------------------------- #
# Division
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("column.divide.by_zero_raises")
@pytest.mark.parametrize(
    "spark_type, a, b",
    [
        (SparkIntegerType(), 5, 0),
        (SparkLongType(), -5, 0),
        (SparkDoubleType(), 1.0, 0.0),
        (SparkDoubleType(), -2.5, -0.0),
        (SparkDecimalType(10, 2), Decimal("1.50"), Decimal("0.00")),
    ],
    ids=["int", "long", "double", "double_neg_zero", "decimal"],
)
def test_divide_by_zero_raises(engine, spark, spark_type, a, b):
    # ANSI: x / 0 raises DIVIDE_BY_ZERO for every numeric type (doubles too). A null dividend or
    # divisor short-circuits to null, even when the divisor is zero.
    F = engine.functions
    schema = SparkStructType([SparkStructField("a", spark_type), SparkStructField("b", spark_type)])
    one = type(a)(1)
    valid_rows = [(a, a), (a, None), (None, b), (None, None), (b, one)]
    actual = engine.build_df(valid_rows, schema).select((F.col("a") / F.col("b")).alias("r"))
    expected = spark.createDataFrame(valid_rows, schema).select((SF.col("a") / SF.col("b")).alias("r"))
    assert_matches_spark(actual, expected, engine)

    bad_rows = valid_rows + [(a, b)]
    with pytest.raises(Exception, match="DIVIDE_BY_ZERO"):
        spark.createDataFrame(bad_rows, schema).select(SF.col("a") / SF.col("b")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(bad_rows, schema).select((F.col("a") / F.col("b")).alias("r")))
    with pytest.raises(Exception, match="DIVIDE_BY_ZERO"):
        spark.createDataFrame(valid_rows, schema).select(SF.col("a") / SF.lit(0)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(valid_rows, schema).select((F.col("a") / F.lit(0)).alias("r")))


@pytest.mark.feature("column.divide.decimal_result_type")
def test_divide_decimal_result_type(engine, spark):
    # decimal(p1,s1) / decimal(p2,s2): scale = max(6, s1 + p2 + 1), precision = p1 - s1 + s2 + scale.
    # So decimal(10,2) / decimal(5,3) -> decimal(19,8); an int divisor is decimal(10,0) -> decimal(21,13).
    def build(M, _str_type):
        return [
            (M.col("a") / M.col("b")).alias("dec_dec"),
            (M.col("a") / M.col("l")).alias("dec_long"),
            (M.col("i") / M.col("b")).alias("int_dec"),
            (M.col("b") / M.col("b")).alias("self"),
        ]

    rows = [row for row in _DECIMAL_OPERANDS_ROWS if row[2] != 0]
    _assert_matches_with_dtypes(engine, spark, rows, _DECIMAL_OPERANDS_SCHEMA, build)

    # Malformed: a zero decimal divisor raises under ANSI.
    bad_rows = rows + [(Decimal("1.00"), Decimal("0.000"), 1, 1)]
    _assert_strict_raises(
        engine,
        spark,
        bad_rows,
        _DECIMAL_OPERANDS_SCHEMA,
        lambda M, _t: (M.col("a") / M.col("b")).alias("r"),
        spark_match="DIVIDE_BY_ZERO",
    )


@pytest.mark.feature("column.divide.integral_returns_double")
def test_divide_integral_returns_double(engine, spark):
    # ``/`` is true division: integral operands of any width yield a double, never a truncated integer.
    schema = SparkStructType(
        [
            SparkStructField("i", SparkIntegerType()),
            SparkStructField("j", SparkIntegerType()),
            SparkStructField("l", SparkLongType()),
            SparkStructField("s", SparkShortType()),
            SparkStructField("b", SparkByteType()),
        ]
    )
    rows = [(7, 2, 3, 9, 4), (1, 3, -2, -7, 2), (-7, 2, 9223372036854775807, 1, -1), (None, 1, 1, None, 3)]

    def build(M, _str_type):
        return [
            (M.col("i") / M.col("j")).alias("int_int"),
            (M.col("l") / M.col("i")).alias("long_int"),
            (M.col("s") / M.col("b")).alias("short_byte"),
            (M.col("i") / 2).alias("int_lit"),
            (10 / M.col("b")).alias("lit_byte"),
        ]

    F = engine.functions
    actual = engine.build_df(rows, schema).select(*build(F, None))
    expected = spark.createDataFrame(rows, schema).select(*build(SF, None))
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)

    # Malformed: an integral zero divisor raises (no silent null / inf).
    _assert_strict_raises(
        engine,
        spark,
        rows + [(1, 0, 1, 1, 1)],
        schema,
        lambda M, _t: (M.col("i") / M.col("j")).alias("r"),
        spark_match="DIVIDE_BY_ZERO",
    )


# ----------------------------------------------------------------------------- #
# Null-safe equality
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("column.eq_null_safe.nulls")
@pytest.mark.parametrize(
    "spark_type, x, y",
    [(SparkIntegerType(), 1, 2), (SparkStringType(), "a", "b"), (SparkDoubleType(), float("nan"), 1.0)],
    ids=["int", "string", "double_nan"],
)
def test_eq_null_safe_nulls(engine, spark, spark_type, x, y):
    # ``<=>`` never returns null: null <=> null is true, null <=> value is false.
    F = engine.functions
    schema = SparkStructType([SparkStructField("a", spark_type), SparkStructField("b", spark_type)])
    rows = [(x, x), (x, y), (x, None), (None, x), (None, None)]

    def build(M):
        return [
            M.col("a").eqNullSafe(M.col("b")).alias("col_col"),
            M.col("a").eqNullSafe(None).alias("vs_none"),
            M.col("a").eqNullSafe(M.lit(None)).alias("vs_null_lit"),
            M.col("a").eqNullSafe(x).alias("vs_value"),
        ]

    actual = engine.build_df(rows, schema).select(*build(F))
    expected = spark.createDataFrame(rows, schema).select(*build(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: incomparable operand types are an analysis error.
    arr_schema = SparkStructType([SparkStructField("arr", SparkArrayType(SparkIntegerType()))])
    with pytest.raises(Exception, match="DATATYPE_MISMATCH"):
        spark.createDataFrame([([1],)], arr_schema).select(SF.col("arr").eqNullSafe(1)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df([([1],)], arr_schema).select(F.col("arr").eqNullSafe(1).alias("r")))


@pytest.mark.feature("column.eq_null_safe.mixed_types")
def test_eq_null_safe_mixed_types(engine, spark):
    # Operands are coerced like ``==`` (int vs string -> bigint, string vs date -> date, decimal vs
    # double -> double) and nulls still compare null-safely.
    F = engine.functions
    schema = SparkStructType(
        [
            SparkStructField("i", SparkIntegerType()),
            SparkStructField("l", SparkLongType()),
            SparkStructField("d", SparkDoubleType()),
            SparkStructField("s", SparkStringType()),
            SparkStructField("dt", SparkDateType()),
            SparkStructField("sd", SparkStringType()),
            SparkStructField("dec", SparkDecimalType(10, 2)),
        ]
    )
    rows = [
        (1, 1, 1.0, "1", date(2024, 1, 1), "2024-01-01", Decimal("1.00")),
        (2, None, 2.5, " 2 ", None, None, Decimal("2.50")),
        (None, None, None, None, None, None, None),
        (3, 4, None, "+3", date(2024, 1, 2), "2024-01-03", None),
        (None, 5, 5.0, None, date(2024, 1, 5), "2024-1-5", Decimal("5.00")),
    ]

    def build(M):
        return [
            M.col("i").eqNullSafe(M.col("l")).alias("int_long"),
            M.col("i").eqNullSafe(M.col("d")).alias("int_double"),
            M.col("i").eqNullSafe(M.col("s")).alias("int_string"),
            M.col("d").eqNullSafe(M.col("dec")).alias("double_decimal"),
            M.col("i").eqNullSafe(M.col("dec")).alias("int_decimal"),
            M.col("dt").eqNullSafe(M.col("sd")).alias("date_string"),
            M.col("l").eqNullSafe(2.5).alias("long_float_lit"),
            M.col("s").eqNullSafe(1).alias("string_int_lit"),
        ]

    actual = engine.build_df(rows, schema).select(*build(F))
    expected = spark.createDataFrame(rows, schema).select(*build(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: the implicit string -> bigint cast is ANSI-strict.
    bad_rows = rows + [(1, 1, 1.0, "abc", None, None, None)]
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        spark.createDataFrame(bad_rows, schema).select(SF.col("i").eqNullSafe(SF.col("s"))).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(bad_rows, schema).select(F.col("i").eqNullSafe(F.col("s")).alias("r")))


# ----------------------------------------------------------------------------- #
# Struct / array / map extraction
# ----------------------------------------------------------------------------- #

_NESTED_STRUCT_TYPE = SparkStructType(
    [
        SparkStructField("x", SparkIntegerType()),
        SparkStructField("y", SparkStringType()),
        SparkStructField("inner", SparkStructType([SparkStructField("z", SparkDoubleType())])),
    ]
)


@pytest.mark.feature("column.get_field.struct")
def test_get_field_struct(engine, spark):
    # A null struct (or a null nested struct) yields null fields, never an error.
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", _NESTED_STRUCT_TYPE), SparkStructField("i", SparkIntegerType())])
    rows = [((1, "a", (1.5,)), 1), ((None, "b", None), 2), (None, 3), ((4, None, (None,)), 4)]

    def build(M):
        return [
            M.col("s").getField("x").alias("x"),
            M.col("s").getField("y").alias("y"),
            M.col("s").getField("inner").getField("z").alias("z"),
            M.col("s")["x"].alias("x_bracket"),
            M.col("s.inner.z").alias("z_dotted"),
        ]

    actual = engine.build_df(rows, schema).select(*build(F))
    expected = spark.createDataFrame(rows, schema).select(*build(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: an unknown field, or extracting from a non-struct column, fails analysis.
    for build_bad, error in [
        (lambda M: M.col("s").getField("missing"), "FIELD_NOT_FOUND"),
        (lambda M: M.col("i").getField("x"), "INVALID_EXTRACT_BASE_FIELD_TYPE"),
    ]:
        with pytest.raises(Exception, match=error):
            spark.createDataFrame(rows, schema).select(build_bad(SF)).collect()
        with pytest.raises(Exception):
            engine.to_records(engine.build_df(rows, schema).select(build_bad(F).alias("r")))


@pytest.mark.feature("column.getitem.array_out_of_range_null")
def test_getitem_array_out_of_range(engine, spark):
    # Spark 4 (ANSI): ``arr[i]`` on a null array or a null element is null, but an index outside the
    # array -- past the end or negative -- raises INVALID_ARRAY_INDEX (only ``get()`` / ``try_element_at``
    # tolerate it with null).
    F = engine.functions
    schema = SparkStructType([SparkStructField("a", SparkArrayType(SparkIntegerType()))])
    rows = [([1, 2, 3],), ([None, 5, 6],), (None,)]

    def build(M):
        return [M.col("a").getItem(0).alias("first"), M.col("a")[2].alias("last"), M.col("a").getItem(1).alias("mid")]

    actual = engine.build_df(rows, schema).select(*build(F))
    expected = spark.createDataFrame(rows, schema).select(*build(SF))
    assert_matches_spark(actual, expected, engine)

    for bad_rows, index in [(rows, 3), (rows, -1), (rows + [([],)], 0)]:
        with pytest.raises(Exception, match="INVALID_ARRAY_INDEX"):
            spark.createDataFrame(bad_rows, schema).select(SF.col("a").getItem(index)).collect()
        with pytest.raises(Exception):
            engine.to_records(engine.build_df(bad_rows, schema).select(F.col("a").getItem(index).alias("r")))


@pytest.mark.feature("column.getitem.map_missing_key_null")
def test_getitem_map_missing_key_null(engine, spark):
    # A missing key, an empty map or a null map all yield null (``[]`` on maps is lenient even under ANSI).
    F = engine.functions
    schema = SparkStructType(
        [
            SparkStructField("m", SparkMapType(SparkStringType(), SparkIntegerType())),
            SparkStructField("mi", SparkMapType(SparkIntegerType(), SparkStringType())),
        ]
    )
    rows = [({"a": 1, "b": None}, {1: "one", 2: "two"}), ({}, {}), (None, None), ({"z": 26}, {3: None})]

    def build(M):
        return [
            M.col("m").getItem("a").alias("present"),
            M.col("m").getItem("b").alias("null_value"),
            M.col("m").getItem("missing").alias("missing"),
            M.col("m")["z"].alias("bracket"),
            M.col("mi").getItem(2).alias("int_key"),
            M.col("mi").getItem(99).alias("int_missing"),
            M.col("mi").getItem("1").alias("str_key_coerced"),
        ]

    actual = engine.build_df(rows, schema).select(*build(F))
    expected = spark.createDataFrame(rows, schema).select(*build(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: a key that cannot be cast to the map's key type raises (ANSI implicit cast).
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        spark.createDataFrame(rows, schema).select(SF.col("mi").getItem("x")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.col("mi").getItem("x").alias("r")))


# ----------------------------------------------------------------------------- #
# isin / contains / rlike
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("column.isin.null_semantics")
def test_isin_null_semantics(engine, spark):
    # SQL IN: a match is true; no match is false, or null if the list holds a null; a null probe is null;
    # an empty list is always false.
    F = engine.functions
    schema = SparkStructType([SparkStructField("i", SparkIntegerType()), SparkStructField("s", SparkStringType())])
    rows = [(1, "a"), (2, "b"), (None, None), (3, "")]

    def build(M):
        return [
            M.col("i").isin(1, 3).alias("plain"),
            M.col("i").isin(1, None).alias("with_null"),
            M.col("i").isin([None]).alias("only_null"),
            M.col("i").isin().alias("empty"),
            M.col("s").isin(["a", ""]).alias("str_list"),
            M.col("s").isin("z", None).alias("str_with_null"),
        ]

    actual = engine.build_df(rows, schema).select(*build(F))
    expected = spark.createDataFrame(rows, schema).select(*build(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: incomparable candidate types fail analysis.
    arr_schema = SparkStructType([SparkStructField("arr", SparkArrayType(SparkIntegerType()))])
    with pytest.raises(Exception, match="DATATYPE_MISMATCH"):
        spark.createDataFrame([([1],)], arr_schema).select(SF.col("arr").isin(1)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df([([1],)], arr_schema).select(F.col("arr").isin(1).alias("r")))


@pytest.mark.feature("column.isin.coerces_candidates")
def test_isin_coerces_candidates(engine, spark):
    # Probe and candidates widen to one common type: string candidates against an int probe (and int
    # candidates against a string probe) compare as bigint; a float candidate widens the comparison to double.
    F = engine.functions
    schema = SparkStructType(
        [
            SparkStructField("i", SparkIntegerType()),
            SparkStructField("s", SparkStringType()),
            SparkStructField("d", SparkDoubleType()),
            SparkStructField("dec", SparkDecimalType(10, 2)),
        ]
    )
    rows = [
        (1, "1", 1.0, Decimal("1.00")),
        (2, " 2 ", 2.5, Decimal("2.50")),
        (None, "07", None, None),
        (3, "+3", 3.0, Decimal("3.10")),
    ]

    def build(M):
        return [
            M.col("i").isin("1", "3").alias("int_vs_str"),
            M.col("i").isin("1", 2.0).alias("int_vs_mixed"),
            M.col("i").isin(2.5).alias("int_vs_float"),
            M.col("s").isin(1, 7).alias("str_vs_int"),
            M.col("d").isin(1, "2.5").alias("double_vs_mixed"),
            M.col("dec").isin(1, 2.5).alias("decimal_vs_num"),
            M.col("i").isin(M.col("d"), 99).alias("int_vs_column"),
        ]

    actual = engine.build_df(rows, schema).select(*build(F))
    expected = spark.createDataFrame(rows, schema).select(*build(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: a candidate (or probe) string that can't be cast to the common numeric type raises.
    for build_bad, bad_rows in [
        (lambda M: M.col("i").isin("abc"), rows),
        (lambda M: M.col("s").isin(1, 2), rows + [(4, "abc", 4.0, None)]),
    ]:
        with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
            spark.createDataFrame(bad_rows, schema).select(build_bad(SF)).collect()
        with pytest.raises(Exception):
            engine.to_records(engine.build_df(bad_rows, schema).select(build_bad(F).alias("r")))


@pytest.mark.feature("column.contains.literal_and_null")
def test_contains_literal_and_null(engine, spark):
    # Substring match is literal (``.`` is not a regex); an empty needle matches every non-null value;
    # a null on either side gives null.
    F = engine.functions
    schema = SparkStructType([SparkStructField("t", SparkStringType()), SparkStructField("sub", SparkStringType())])
    rows = [("banana", "an"), ("banana", "xy"), ("", ""), (None, "a"), ("abc", None), ("a.c", "."), ("ABC", "b")]

    def build(M):
        return [
            M.col("t").contains("an").alias("lit"),
            M.col("t").contains("").alias("empty"),
            M.col("t").contains(".").alias("dot"),
            M.col("t").contains(None).alias("null_lit"),
            M.col("t").contains(M.col("sub")).alias("col"),
        ]

    actual = engine.build_df(rows, schema).select(*build(F))
    expected = spark.createDataFrame(rows, schema).select(*build(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: a non-string (array) haystack fails analysis.
    arr_schema = SparkStructType([SparkStructField("arr", SparkArrayType(SparkIntegerType()))])
    with pytest.raises(Exception, match="DATATYPE_MISMATCH"):
        spark.createDataFrame([([1],)], arr_schema).select(SF.col("arr").contains("a")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df([([1],)], arr_schema).select(F.col("arr").contains("a").alias("r")))


@pytest.mark.feature("column.like")
def test_like_against_spark(engine, spark):
    """``_`` is one character and ``%`` any run (newlines included); ``\\`` escapes them; the whole string must match."""
    F = engine.functions
    schema = SparkStructType([SparkStructField("t", SparkStringType()), SparkStructField("n", SparkIntegerType())])
    rows = [("abc", 123), ("a_c", 10), ("a%c", None), ("ab\nc", 1), ("", 0), ("ABC", 12), (None, 5)]
    patterns = ["a%", "a_c", "%c", "a\\_c", "a\\%c", "a%c", "%", "", "abc", "a.c", "[a]%", "a\\\\%"]

    def build(M):
        cols = [M.col("t").like(p).alias(f"p{i}") for i, p in enumerate(patterns)]
        return cols + [M.col("n").like("1%").alias("int_input")]

    actual = engine.build_df(rows, schema).select("t", *build(F))
    expected = spark.createDataFrame(rows, schema).select("t", *build(SF))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("column.like")
@pytest.mark.parametrize("pattern", ["a\\", "a\\bc"], ids=["escape_at_end", "escape_before_other"])
def test_like_invalid_escape_raises_like_spark(engine, spark, pattern):
    F = engine.functions
    schema = SparkStructType([SparkStructField("t", SparkStringType())])
    rows = [("abc",)]
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.col("t").like(pattern)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.col("t").like(pattern).alias("r")))

    # Malformed input type: an array is not implicitly cast to string.
    arr_schema = SparkStructType([SparkStructField("arr", SparkArrayType(SparkIntegerType()))])
    with pytest.raises(Exception):
        spark.createDataFrame([([1],)], arr_schema).select(SF.col("arr").like("a")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df([([1],)], arr_schema).select(F.col("arr").like("a").alias("r")))


@pytest.mark.feature("column.rlike.invalid_regex_raises")
@pytest.mark.parametrize(
    "pattern", ["[", "a(b", "*a", "a{2,1}"], ids=["open_class", "open_group", "dangling", "range"]
)
def test_rlike_invalid_regex_raises(engine, spark, pattern):
    F = engine.functions
    schema = SparkStructType([SparkStructField("t", SparkStringType())])
    rows = [("abc",), ("",), (None,)]

    # Valid: a well-formed pattern is a partial (find) match.
    actual = engine.build_df(rows, schema).select(F.col("t").rlike("b").alias("r"))
    expected = spark.createDataFrame(rows, schema).select(SF.col("t").rlike("b").alias("r"))
    assert_matches_spark(actual, expected, engine)

    # Malformed: Java's regex compiler rejects the pattern, failing the query.
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.col("t").rlike(pattern)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.col("t").rlike(pattern).alias("r")))


# The rlike patterns the datahub nba pipeline (model/nba and the std/rpt tables it reads) applies.
_NBA_RLIKE_PATTERNS = [
    "check|(^|-)ck($|-)",
    "money-market",
    "checking|money market",
    "cert|(^|-)cd($|-)",
    "sav|check|hysa|high-yield",
    "^[0-9]+$",
    "^[0-9A-HJKMNP-TV-Z]{26}$",
    r"^\d\.\d{4,}$",
    r"^\d{4}-\d{2}-\d{2}$",
    r"^\d{2}/\d{4}$",
    r"^\d{2}/\d{2}$",
    r"^\d{2}/\d{2}/\d{4}$",
    r"^\d{4}$",
    r"^-?\d+\.?\d*$",
    r"^\d+Y$",
    r"^\d+M$",
    r"(?i)iPad|Tablet|Silk|Kindle|PlayBook",
    r"(?i)Mobi|iPhone|iPod|Android.*Mobile|Windows Phone|BlackBerry",
    r"(?i)axios|node-fetch|python-requests|curl|Go-http-client",
    r"(\bautos?\b|vehicle|\bcar\b|\brv\b|motorcycle|boat)",
    r"(checking|savings|certificate|money market|\bcds?\b|dao|deposit|account opening)",
]

_NBA_RLIKE_INPUTS = [
    "checking",
    "share-ck",
    "ck-plus",
    "deck",
    "money-market",
    "Money Market Checking",
    "cd",
    "jumbo-cd-12",
    "cds",
    "hysa",
    "00012345",
    "12a45",
    "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "01ARZ3NDEKTSV4RRFFQ69G5FAU",
    "0.12345",
    "0.123",
    "2024-01-15",
    "2024-1-15",
    "01/2024",
    "01/24",
    "01/24/2024",
    "2024",
    "-12.",
    "12.5.1",
    "-",
    "10Y",
    "36M",
    "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X)",
    "Mozilla/5.0 (Linux; Android 14) Mobile Safari",
    "axios/1.6.0",
    "CURL/8.0",
    "New Auto Loan",
    "carpet deals",
    "RV refinance",
    "١٢٣",  # Arabic-Indic digits: Java's \d and [0-9] are ASCII-only
    "123\n",
    "",
    None,
]


@pytest.mark.feature("column.rlike.nba_patterns")
def test_rlike_nba_patterns(engine, spark):
    # Java regex semantics: partial (find) match, inline (?i), \b word boundaries, ASCII-only \d.
    F = engine.functions
    schema = SparkStructType([SparkStructField("t", SparkStringType())])
    rows = [(v,) for v in _NBA_RLIKE_INPUTS]

    def build(M):
        return [M.col("t").rlike(p).alias(f"p{i}") for i, p in enumerate(_NBA_RLIKE_PATTERNS)]

    actual = engine.build_df(rows, schema).select(F.col("t"), *build(F))
    expected = spark.createDataFrame(rows, schema).select(SF.col("t"), *build(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: a broken variant of an nba pattern fails the query.
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.col("t").rlike(r"^\d{4-\d{2}$")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.col("t").rlike(r"^\d{4-\d{2}$").alias("r")))


# ----------------------------------------------------------------------------- #
# Sort ordering
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("column.sort.nulls_ordering")
@pytest.mark.parametrize(
    "build",
    [
        lambda M: [M.col("v").asc(), M.col("k")],
        lambda M: [M.col("v").desc(), M.col("k")],
        lambda M: [M.col("v").asc_nulls_first(), M.col("k")],
        lambda M: [M.col("v").asc_nulls_last(), M.col("k")],
        lambda M: [M.col("v").desc_nulls_first(), M.col("k")],
        lambda M: [M.col("v").desc_nulls_last(), M.col("k")],
        lambda M: [M.col("s").desc_nulls_first(), M.col("v").asc_nulls_last(), M.col("k")],
    ],
    ids=["asc", "desc", "asc_nulls_first", "asc_nulls_last", "desc_nulls_first", "desc_nulls_last", "multi"],
)
def test_sort_nulls_ordering(engine, spark, build):
    # Spark: ascending puts nulls first and descending puts them last by default; NaN sorts above
    # every number (and +inf), and the explicit *_nulls_* variants override the null placement.
    F = engine.functions
    schema = SparkStructType(
        [
            SparkStructField("k", SparkIntegerType()),
            SparkStructField("v", SparkDoubleType()),
            SparkStructField("s", SparkStringType()),
        ]
    )
    rows = [
        (1, 2.0, "b"),
        (2, None, "a"),
        (3, float("nan"), None),
        (4, -1.0, "b"),
        (5, None, None),
        (6, float("inf"), "a"),
        (7, 2.0, "c"),
    ]
    actual = engine.build_df(rows, schema).orderBy(*build(F))
    expected = spark.createDataFrame(rows, schema).orderBy(*build(SF))
    assert_matches_spark(actual, expected, engine, check_row_order=True)

    # Malformed: maps are not orderable.
    map_schema = SparkStructType([SparkStructField("m", SparkMapType(SparkStringType(), SparkIntegerType()))])
    map_rows = [({"a": 1},), (None,)]
    with pytest.raises(Exception, match="INVALID_ORDERING_TYPE"):
        spark.createDataFrame(map_rows, map_schema).orderBy(SF.col("m").asc()).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(map_rows, map_schema).orderBy(F.col("m").asc()))


# ----------------------------------------------------------------------------- #
# try_cast: DDL targets, malformed matrix, decimal overflow
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("column.try_cast.ddl_string_target")
@pytest.mark.parametrize(
    "ddl, spark_type",
    [
        ("decimal(5,2)", SparkDecimalType(5, 2)),
        ("DECIMAL(10, 0)", SparkDecimalType(10, 0)),
        ("bigint", SparkLongType()),
        ("smallint", SparkShortType()),
        ("tinyint", SparkByteType()),
        ("integer", SparkIntegerType()),
        ("float", SparkFloatType()),
        ("date", SparkDateType()),
        ("timestamp", SparkTimestampType()),
        ("boolean", SparkBooleanType()),
    ],
)
def test_try_cast_ddl_string_target(engine, spark, ddl, spark_type):
    # A DDL type string names the same target as the DataType; malformed values become null.
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkStringType())])
    rows = [("1",), ("12.345",), ("-300",), ("2024-01-15",), ("true",), ("abc",), ("",), (None,)]
    actual = engine.build_df(rows, schema).select(F.col("s").try_cast(ddl).alias("v"))
    expected = spark.createDataFrame(rows, schema).select(SF.col("s").try_cast(ddl).alias("v"))
    assert expected.schema["v"].dataType == spark_type
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)
    str_type = engine.dtype(SparkStringType())
    assert_matches_spark(
        actual.select(F.col("v").cast(str_type).alias("v")),
        expected.select(SF.col("v").cast(SparkStringType()).alias("v")),
        engine,
    )

    # Malformed DDL: an unparseable or unknown type string ("bool" is not a Spark type name) fails even
    # the lenient cast.
    for bad_ddl in ["decimal(5", "notatype", "bool"]:
        with pytest.raises(Exception):
            spark.createDataFrame(rows, schema).select(SF.col("s").try_cast(bad_ddl)).collect()
        with pytest.raises(Exception):
            engine.to_records(engine.build_df(rows, schema).select(F.col("s").try_cast(bad_ddl).alias("v")))


_TRY_CAST_MALFORMED_STRINGS = [
    " 12 ",
    "+7",
    "-0",
    "1.5",
    "1e3",
    "1_000",
    "0x10",
    "9999999999",
    "99999999999999999999",
    "NaN",
    "-Infinity",
    "abc",
    "",
    " ",
    "t",
    "yes",
    "2024-01-15",
    "2024-02-30",
    "2024-01-15 25:00:00",
    "2024-01-15T10:30:45",
    "15/01/2024",
    None,
]


@pytest.mark.feature("column.try_cast.malformed_string_matrix")
@pytest.mark.parametrize(
    "ddl, spark_type",
    [
        ("int", SparkIntegerType()),
        ("bigint", SparkLongType()),
        ("smallint", SparkShortType()),
        ("double", SparkDoubleType()),
        ("float", SparkFloatType()),
        ("decimal(10,2)", SparkDecimalType(10, 2)),
        ("boolean", SparkBooleanType()),
        ("date", SparkDateType()),
        ("timestamp", SparkTimestampType()),
    ],
)
def test_try_cast_malformed_string_matrix(engine, spark, ddl, spark_type):
    # Every string Spark's strict cast would reject becomes null under try_cast; the rest parse exactly
    # like ``cast`` (whitespace trimmed, sign, special doubles, ISO 'T' separator...).
    F = engine.functions
    schema = SparkStructType([SparkStructField("s", SparkStringType())])
    rows = [(v,) for v in _TRY_CAST_MALFORMED_STRINGS]
    actual = engine.build_df(rows, schema).select(F.col("s"), F.col("s").try_cast(ddl).alias("v"))
    expected = spark.createDataFrame(rows, schema).select(SF.col("s"), SF.col("s").try_cast(spark_type).alias("v"))
    assert_matches_spark(actual, expected, engine)

    # The strict cast of the same matrix raises (every target has at least one malformed string).
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.col("s").cast(spark_type)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.col("s").cast(engine.dtype(spark_type)).alias("v")))


@pytest.mark.feature("column.try_cast.to_decimal_overflow_null")
@pytest.mark.parametrize(
    "source_type, valid, overflow",
    _TO_DECIMAL_5_2_SOURCES,
    ids=["decimal", "int", "long", "double", "string"],
)
def test_try_cast_to_decimal_overflow_null(engine, spark, source_type, valid, overflow):
    # The lenient sibling of ``column.cast.to_decimal_overflow_raises``: overflow becomes null.
    F = engine.functions
    schema = SparkStructType([SparkStructField("x", source_type)])
    target = SparkDecimalType(5, 2)
    rows = [(v,) for v in valid] + [(overflow,)]
    actual = engine.build_df(rows, schema).select(
        F.col("x").try_cast("decimal(5,2)").cast(engine.dtype(SparkStringType())).alias("r")
    )
    expected = spark.createDataFrame(rows, schema).select(
        SF.col("x").try_cast(target).cast(SparkStringType()).alias("r")
    )
    assert_matches_spark(actual, expected, engine)

    # The strict cast of the same rows raises.
    with pytest.raises(Exception, match="NUMERIC_VALUE_OUT_OF_RANGE"):
        spark.createDataFrame(rows, schema).select(SF.col("x").cast(target)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.col("x").cast(engine.dtype(target)).alias("r")))
