"""Shared Spark-parity tests for ``functions`` (col/lit/when/cast/...).

Lifted from ``polarsdf/functions_test.py`` and converted to the engine-parametrized
harness: each test builds the actual frame via ``engine.build_df`` and the expected
frame via real Spark, then compares with ``assert_matches_spark``. Every expression
is built from ``engine.functions`` so it runs on the active engine.

Tests that don't fit this shape stay co-located in the source file:

- Pandas round-trip assertions (``test_lit_against_spark``, ``test_round_against_spark``).
- Shape-only / non-Spark assertions (``test_uuid_yields_version4_string_format``,
  the ``TestRand`` checks, ``test_broadcast_returns_same_dataframe``).
- ``Object``-dtype list cells (``TestSizeObjectList``) — Polars-specific layout.
- Arrow-compare frames (``test_struct_nested_with_array_and_map``).
- Unit tests of private helpers / API guards (``_struct_child_field_name``,
  ``test_concat_without_inputs_raises``, ``test_struct_requires_at_least_one_column``,
  ``test_options_argument_rejected``).
- ``test_ranks_with_subcategory`` — uses ``polarsdf.Window`` (not on the adapter yet).
- ``TestNow``/``current_timestamp`` — assert wall-clock proximity, not value equality.
- ``test_array_zero_index_returns_null`` (TryElementAt) — Spark side via pandas roundtrip.
- ``test_from_json_malformed_returns_null`` — polars returns ``None``, Spark returns a
  Row of nulls; oracle normalization diverges. Valid case is lifted below.
"""

import json
import math
from datetime import date, datetime
from decimal import Decimal

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

_STR = SparkStringType()
_LONG = SparkLongType()
_INT = SparkIntegerType()
_DOUBLE = SparkDoubleType()
_BIN = SparkBinaryType()


def _schema(*fields):
    return SparkStructType([SparkStructField(name, dtype, True) for name, dtype in fields])


# ============================================================================ #
# TestFunctions (mixed)
# ============================================================================ #


@pytest.mark.feature("functions.when.basic")
def test_when(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG), ("c", _LONG))
    rows = [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

    actual = engine.build_df(rows, schema).withColumn("result", F.when(F.col("a") > 2, "yes").otherwise("no"))
    expected = spark.createDataFrame(rows, schema).withColumn(
        "result", SF.when(SF.col("a") > 2, "yes").otherwise("no")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.when.chained_boolean_output")
def test_chained_when_boolean_output(engine, spark):
    F = engine.functions
    schema = _schema(("b", _STR), ("c", _STR))
    rows = [("A", "b"), ("B", "e"), ("C", "g"), ("D", "z")]

    sf_expr = (
        F.when((F.col("b") == "A") & (F.col("c").isin("A", "b", "c")), True)
        .when((F.col("b") == "B") & (F.col("c").isin("d", "e")), True)
        .when((F.col("b") == "C") & (F.col("c").isin("f", "g", "h", "i")), True)
        .otherwise(False)
    )
    sp_expr = (
        SF.when((SF.col("b") == "A") & (SF.col("c").isin("A", "b", "c")), True)
        .when((SF.col("b") == "B") & (SF.col("c").isin("d", "e")), True)
        .when((SF.col("b") == "C") & (SF.col("c").isin("f", "g", "h", "i")), True)
        .otherwise(False)
    )

    actual = engine.build_df(rows, schema).withColumn("result", sf_expr)
    expected = spark.createDataFrame(rows, schema).withColumn("result", sp_expr)
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.get_json_object")
@pytest.mark.parametrize(
    "json_data, path",
    [
        ([json.dumps({"a": 1}), json.dumps({"a": 2})], "$.a"),
        ([json.dumps({"a": {"b": 3}}), json.dumps({"a": {"b": 4}})], "$.a.b"),
        ([json.dumps({"arr": [10, 20]}), json.dumps({"arr": [30, 40]})], "$.arr[1]"),
        ([json.dumps({"a": {"b": [5, 6]}}), json.dumps({"a": {"b": [7, 8]}})], "$.a.b[0]"),
        (
            [json.dumps({"items": [{"id": 1}, {"id": 2}]}), json.dumps({"items": [{"id": 3}, {"id": 4}]})],
            "$.items[1].id",
        ),
    ],
)
def test_get_json_object(engine, spark, json_data, path):
    F = engine.functions
    schema = _schema(("json_col", _STR))
    rows = [(j,) for j in json_data]
    actual = engine.build_df(rows, schema).select(F.get_json_object("json_col", path).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.get_json_object("json_col", path).alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.coalesce")
@pytest.mark.parametrize(
    "a_vals, b_vals",
    [
        ([None, 2, None], [1, None, 3]),
        ([None, None, None], [None, None, None]),
        ([None, 5, 6], ["x", "y", None]),
        (["", None, "z"], ["a", "b", None]),
    ],
)
def test_coalesce_against_spark(engine, spark, a_vals, b_vals):
    F = engine.functions
    # Both columns string-typed (Spark 4 ANSI requires a common type; string covers int/str mixes).
    schema = _schema(("a", _STR), ("b", _STR))
    rows = [(None if a is None else str(a), None if b is None else str(b)) for a, b in zip(a_vals, b_vals)]

    actual = engine.build_df(rows, schema).select(F.coalesce(F.col("a"), F.col("b")).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.coalesce(SF.col("a"), SF.col("b")).alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.asc")
@pytest.mark.parametrize(
    "values, spark_type",
    [
        ([3, 1, 2], _LONG),
        (["b", "c", "a"], _STR),
        ([3.3, 1.1, 2.2], _DOUBLE),
    ],
)
def test_asc_against_spark(engine, spark, values, spark_type):
    F = engine.functions
    schema = _schema(("x", spark_type))
    rows = [(v,) for v in values]
    actual = engine.build_df(rows, schema).orderBy(F.asc(F.col("x")))
    expected = spark.createDataFrame(rows, schema).orderBy(SF.asc("x"))
    assert_matches_spark(actual, expected, engine, check_row_order=True)


@pytest.mark.feature("functions.desc")
@pytest.mark.parametrize(
    "values, spark_type",
    [
        ([3, 1, 2], _LONG),
        (["b", "c", "a"], _STR),
        ([3.3, 1.1, 2.2], _DOUBLE),
    ],
)
def test_desc_against_spark(engine, spark, values, spark_type):
    F = engine.functions
    schema = _schema(("x", spark_type))
    rows = [(v,) for v in values]
    actual = engine.build_df(rows, schema).orderBy(F.desc(F.col("x")))
    expected = spark.createDataFrame(rows, schema).orderBy(SF.desc("x"))
    assert_matches_spark(actual, expected, engine, check_row_order=True)


@pytest.mark.feature("functions.regexp_replace.str_vs_column")
@pytest.mark.parametrize("col_input_kind", ["name", "col"])
@pytest.mark.parametrize(
    "input_values, pattern, replacement",
    [
        (["abc123", "xyz456"], r"\d+", ""),
        (["hello world", "world hello"], "world", "earth"),
        (["aaa", "aba", "aca"], "a", "x"),
        (["test123", "123test"], r"^\d+", "NUM"),
        (["test123", "123test"], r"\d+$", "END"),
    ],
)
def test_regexp_replace_str_vs_column(engine, spark, col_input_kind, input_values, pattern, replacement):
    F = engine.functions
    schema = _schema(("txt", _STR))
    rows = [(v,) for v in input_values]
    col_input = "txt" if col_input_kind == "name" else F.col("txt")

    actual = engine.build_df(rows, schema).select(F.regexp_replace(col_input, pattern, replacement).alias("replaced"))
    expected = spark.createDataFrame(rows, schema).select(
        SF.regexp_replace("txt", pattern, replacement).alias("replaced")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.length.str_vs_column")
@pytest.mark.parametrize("col_input_kind", ["name", "col"])
@pytest.mark.parametrize(
    "input_values",
    [
        ["abc", "de", ""],
        ["你好", "世界", ""],
        [None, "x", "longer string"],
        ["😊", "👍🏽", "💯"],
    ],
)
def test_length_str_vs_column(engine, spark, col_input_kind, input_values):
    F = engine.functions
    schema = _schema(("txt", _STR))
    rows = [(v,) for v in input_values]
    col_input = "txt" if col_input_kind == "name" else F.col("txt")

    actual = engine.build_df(rows, schema).select(F.length(col_input).cast(engine.dtype(_INT)).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.length("txt").alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_timestamp.with_format")
@pytest.mark.parametrize("col_input_kind", ["name", "col"])
@pytest.mark.parametrize(
    "datetime_strs, fmt",
    [
        (["2023-01-01 12:34:56", "2024-02-02 23:45:01"], "yyyy-MM-dd HH:mm:ss"),
        (["01-03-2023 09:15:00", "31-12-2022 23:59:59"], "dd-MM-yyyy HH:mm:ss"),
        (["20230101 120000", "20240101 130000"], "yyyyMMdd HHmmss"),
        (["2024-05-31 20:14:19.993", "2023-12-12 11:11:11.123"], "yyyy-MM-dd HH:mm:ss.SSS"),
        (["2024-05-31 23:58:32.880000", "2023-12-12 11:11:11.123456"], "yyyy-MM-dd HH:mm:ss.SSSSSS"),
        (["2024-05-31 20:14:19.9", "2023-12-12 11:11:11.1"], "yyyy-MM-dd HH:mm:ss.S"),
        (["2024-05-31 20:14:19.99", "2023-12-12 11:11:11.12"], "yyyy-MM-dd HH:mm:ss.SS"),
        (["2024-05-31 20:14:19.12345", "2023-12-12 11:11:11.99999"], "yyyy-MM-dd HH:mm:ss.SSSSS"),
    ],
)
def test_to_timestamp_against_spark(engine, spark, col_input_kind, datetime_strs, fmt):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [(v,) for v in datetime_strs]
    col_input = "ts" if col_input_kind == "name" else F.col("ts")

    actual = engine.build_df(rows, schema).select(F.to_timestamp(col_input, fmt).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.to_timestamp("ts", fmt).alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_timestamp.no_format")
@pytest.mark.parametrize(
    "datetime_strs",
    [
        ["2023-01-01 12:34:56", "2024-02-02 23:45:01"],
        ["2024-05-31T20:14:19", "2023-12-12T11:11:11"],
    ],
)
def test_to_timestamp_no_format_against_spark(engine, spark, datetime_strs):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [(v,) for v in datetime_strs]

    actual = engine.build_df(rows, schema).select(F.to_timestamp("ts").alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.to_timestamp("ts").alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_timestamp.no_format_iso8601_z")
def test_to_timestamp_no_format_iso8601_z_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("createdOn", _STR))
    rows = [("2026-04-26T00:00:00Z",)]
    expected = spark.createDataFrame(rows, schema).select(SF.to_timestamp("createdOn").alias("t"))
    for fn_name in ("to_timestamp", "try_to_timestamp"):
        actual = engine.build_df(rows, schema).select(getattr(F, fn_name)("createdOn").alias("t"))
        assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_timestamp.no_format_malformed_raises")
def test_to_timestamp_no_format_malformed_raises(engine, spark):
    """Spark 4 to_timestamp(col) without format is ANSI-strict and raises on malformed input."""
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [("2023-01-01 12:34:56",), ("not-a-date",)]
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.to_timestamp("ts").alias("result")))
    with pytest.raises(Exception):
        spark.createDataFrame([("not-a-date",)], schema).select(SF.to_timestamp("ts").alias("result")).collect()


@pytest.mark.feature("functions.to_timestamp.with_format_malformed_raises")
@pytest.mark.parametrize(
    "bad_value, fmt",
    [
        ("not-a-date", "yyyy-MM-dd HH:mm:ss"),
        ("01-03-2023 09:15:00", "yyyy-MM-dd HH:mm:ss"),
    ],
)
def test_to_timestamp_with_format_malformed_raises(engine, spark, bad_value, fmt):
    """Spark 4 to_timestamp(col, fmt) is ANSI-strict and raises on malformed or pattern-mismatched input."""
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [(bad_value,)]
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.to_timestamp("ts", fmt).alias("result")))
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.to_timestamp("ts", fmt).alias("result")).collect()


@pytest.mark.feature("functions.abs.parametrized")
@pytest.mark.parametrize(
    "values, spark_type",
    [
        ([-5, -1, 0, 1, 5], _LONG),
        ([-3.5, -0.1, 0.0, 0.1, 3.5], _DOUBLE),
        ([None, -2, 2, None], _LONG),
    ],
)
def test_abs_against_spark_parametrized(engine, spark, values, spark_type):
    F = engine.functions
    schema = _schema(("x", spark_type))
    rows = [(v,) for v in values]
    actual = engine.build_df(rows, schema).select(F.abs(F.col("x")).alias("abs_x"))
    expected = spark.createDataFrame(rows, schema).select(SF.abs("x").alias("abs_x"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.lower.str_vs_column")
@pytest.mark.parametrize("col_input_kind", ["name", "col"])
@pytest.mark.parametrize(
    "input_values",
    [
        ["ABC", "abc", "AbC"],
        ["", None, "Already lower"],
        ["MiXeD 123!@$", "CamelCase", "UPPER lower"],
    ],
)
def test_lower_str_vs_column(engine, spark, col_input_kind, input_values):
    F = engine.functions
    schema = _schema(("txt", _STR))
    rows = [(v,) for v in input_values]
    col_input = "txt" if col_input_kind == "name" else F.col("txt")

    actual = engine.build_df(rows, schema).select(F.lower(col_input).alias("lowered"))
    expected = spark.createDataFrame(rows, schema).select(SF.lower("txt").alias("lowered"))
    assert_matches_spark(actual, expected, engine)


_STRUCT_ABC_SCHEMA = _schema(("a", _LONG), ("b", _LONG), ("c", _LONG))
_STRUCT_ABC_ROWS = [(1, 4, 7), (2, 5, 8), (3, 6, 9)]


@pytest.mark.feature("functions.struct.oracle")
def test_struct_oracle(engine, spark):
    """PySpark parity: field names follow CreateStruct (plain cols vs colN for lit / expr)."""
    F = engine.functions
    sf_exprs = [
        F.struct("a", "b").alias("s1"),
        F.struct(F.col("a"), F.col("b")).alias("s2"),
        F.struct([F.col("a"), F.col("b")]).alias("s3"),
        F.struct(F.col("a"), F.lit(1)).alias("s4"),
        F.struct(F.lit(1), F.lit(2)).alias("s5"),
        F.struct(F.col("a") + 1, F.col("b")).alias("s6"),
        F.struct(F.col("a"), F.struct(F.col("b"), F.lit(3))).alias("s7"),
        F.struct(F.col("a").alias("z")).alias("s8"),
    ]
    sp_exprs = [
        SF.struct("a", "b").alias("s1"),
        SF.struct(SF.col("a"), SF.col("b")).alias("s2"),
        SF.struct([SF.col("a"), SF.col("b")]).alias("s3"),
        SF.struct(SF.col("a"), SF.lit(1)).alias("s4"),
        SF.struct(SF.lit(1), SF.lit(2)).alias("s5"),
        SF.struct(SF.col("a") + 1, SF.col("b")).alias("s6"),
        SF.struct(SF.col("a"), SF.struct(SF.col("b"), SF.lit(3))).alias("s7"),
        SF.struct(SF.col("a").alias("z")).alias("s8"),
    ]
    actual = engine.build_df(_STRUCT_ABC_ROWS, _STRUCT_ABC_SCHEMA).select(*sf_exprs)
    expected = spark.createDataFrame(_STRUCT_ABC_ROWS, _STRUCT_ABC_SCHEMA).select(*sp_exprs)
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.struct.nested_aliased_inner")
def test_struct_nested_aliased_inner_structs_match_spark(engine, spark):
    """Outer struct must use .alias() names for nested struct children (not col1/col2)."""
    F = engine.functions
    sf = F.struct(
        F.struct(F.col("a").alias("field_a")).alias("nested_x"),
        F.struct(F.col("b").alias("field_b")).alias("nested_y"),
    ).alias("composite")
    sp = SF.struct(
        SF.struct(SF.col("a").alias("field_a")).alias("nested_x"),
        SF.struct(SF.col("b").alias("field_b")).alias("nested_y"),
    ).alias("composite")
    actual = engine.build_df(_STRUCT_ABC_ROWS, _STRUCT_ABC_SCHEMA).select(sf)
    expected = spark.createDataFrame(_STRUCT_ABC_ROWS, _STRUCT_ABC_SCHEMA).select(sp)
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestConcat
# ============================================================================ #


@pytest.mark.feature("functions.concat.single_column_identity")
def test_concat_single_column_is_identity_on_strings(engine, spark):
    F = engine.functions
    schema = _schema(("token", _STR))
    rows = [("zig",), (None,), ("",)]
    actual = engine.build_df(rows, schema).select(F.concat("token").alias("out"))
    expected = spark.createDataFrame(rows, schema).select(SF.concat(SF.col("token")).alias("out"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.concat.two_parts_any_null_yields_null")
def test_concat_two_parts_any_null_yields_null(engine, spark):
    F = engine.functions
    schema = _schema(("prefix", _STR), ("suffix", _STR))
    rows = [("aa", "bb"), (None, "bb"), ("cc", None)]
    actual = engine.build_df(rows, schema).select(F.concat(F.col("prefix"), F.col("suffix")).alias("out"))
    expected = spark.createDataFrame(rows, schema).select(SF.concat(SF.col("prefix"), SF.col("suffix")).alias("out"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.concat.accepts_string_name_or_column")
def test_concat_accepts_string_name_or_column_object(engine, spark):
    F = engine.functions
    schema = _schema(("segment", _STR))
    rows = [("north",), ("south",)]
    sparkle_df = engine.build_df(rows, schema)
    spark_df = spark.createDataFrame(rows, schema)
    by_name = sparkle_df.select(F.concat("segment", F.lit(":"), F.col("segment")).alias("out"))
    by_col = sparkle_df.select(F.concat(F.col("segment"), F.lit(":"), "segment").alias("out"))
    expected = spark_df.select(SF.concat(SF.col("segment"), SF.lit(":"), SF.col("segment")).alias("out"))
    assert_matches_spark(by_name, expected, engine)
    assert_matches_spark(by_col, expected, engine)


@pytest.mark.feature("functions.concat.coerces_integers")
def test_concat_coerces_integer_columns_like_strings(engine, spark):
    F = engine.functions
    schema = _schema(("lane", _LONG), ("slot", _LONG))
    rows = [(7, 13), (0, 42)]
    actual = engine.build_df(rows, schema).select(F.concat(F.col("lane"), F.lit("-"), F.col("slot")).alias("merged"))
    expected = spark.createDataFrame(rows, schema).select(
        SF.concat(SF.col("lane"), SF.lit("-"), SF.col("slot")).alias("merged")
    )
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestInitcap / TestMd5 / TestTrimAndSplit / TestMonotonicallyIncreasingId
# ============================================================================ #


@pytest.mark.feature("functions.initcap")
@pytest.mark.parametrize(
    "values",
    [
        ["hello world", "FOO BAR", "already Title"],
        [None, "", "café latte"],
        ["UPPER", "lower", "mIxEd CaSe"],
    ],
)
def test_initcap_against_spark(engine, spark, values):
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [(v,) for v in values]
    actual = engine.build_df(rows, schema).select(F.initcap("s").alias("out"))
    expected = spark.createDataFrame(rows, schema).select(SF.initcap(SF.col("s")).alias("out"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.md5.string")
def test_md5_string_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [("abc",), ("",), (None,), ("café",)]
    actual = engine.build_df(rows, schema).select(F.md5("s").alias("h"))
    expected = spark.createDataFrame(rows, schema).select(SF.md5(SF.col("s")).alias("h"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.md5.binary")
def test_md5_binary_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("b", _BIN))
    rows = [(b"abc",), (None,), (b"",), (b"\x00\xff",)]
    actual = engine.build_df(rows, schema).select(F.md5("b").alias("h"))
    expected = spark.createDataFrame(rows, schema).select(SF.md5(SF.col("b")).alias("h"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.trim")
@pytest.mark.parametrize(
    "values",
    [
        ["  a  ", "b\t", " c \n"],
        [None, "  x  ", ""],
    ],
)
def test_trim_against_spark(engine, spark, values):
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [(v,) for v in values]
    actual = engine.build_df(rows, schema).select(F.trim("s").alias("out"))
    expected = spark.createDataFrame(rows, schema).select(SF.trim(SF.col("s")).alias("out"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.split")
@pytest.mark.parametrize(
    "values, pattern, limit",
    [
        (["a-b-c", "x-y-z", None], r"-", -1),
        (["a1b1c", "nope"], r"\d", -1),
        (["a-b-c-d", "p.q"], r"-", 2),
    ],
)
def test_split_against_spark(engine, spark, values, pattern, limit):
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [(v,) for v in values]
    actual = engine.build_df(rows, schema).select(F.split("s", pattern, limit).alias("parts"))
    expected = spark.createDataFrame(rows, schema).select(SF.split(SF.col("s"), SF.lit(pattern), limit).alias("parts"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.monotonically_increasing_id")
def test_monotonically_increasing_id_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("k", _STR))
    rows = [("a",), ("b",), ("c",), ("d",)]
    actual = engine.build_df(rows, schema).select(F.monotonically_increasing_id().alias("id"))
    # The id encodes the partition index; the engines are single-partition, so pin Spark to one partition
    # (otherwise the ids depend on the session's default parallelism).
    expected = spark.createDataFrame(rows, schema).coalesce(1).select(SF.monotonically_increasing_id().alias("id"))
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestToDate
# ============================================================================ #


@pytest.mark.feature("functions.to_date.with_format")
@pytest.mark.parametrize(
    "date_strs, fmt",
    [
        (["1997-02-28", "2024-12-31"], "yyyy-MM-dd"),
        (["28-02-1997", "31-12-2024"], "dd-MM-yyyy"),
    ],
)
def test_to_date_against_spark(engine, spark, date_strs, fmt):
    F = engine.functions
    schema = _schema(("d", _STR))
    rows = [(v,) for v in date_strs]
    actual = engine.build_df(rows, schema).select(F.to_date("d", fmt).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.to_date("d", fmt).alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_date.malformed_raises")
@pytest.mark.parametrize(
    "bad_value, fmt",
    [
        ("bad", "yyyy-MM-dd"),
        ("1997-02-28", "dd-MM-yyyy"),
    ],
)
def test_to_date_malformed_raises(engine, spark, bad_value, fmt):
    """Spark 4 to_date is ANSI-strict and raises on malformed or pattern-mismatched input."""
    F = engine.functions
    schema = _schema(("d", _STR))
    rows = [(bad_value,)]
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.to_date("d", fmt).alias("result")))
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.to_date("d", fmt).alias("result")).collect()


# ============================================================================ #
# TestTryToTimestamp / TestTryToDate / element_at + try_element_at
# ============================================================================ #


@pytest.mark.feature("functions.try_to_timestamp.valid")
@pytest.mark.parametrize(
    "datetime_strs, fmt",
    [
        (["2023-01-01 12:34:56", "2024-02-02 23:45:01"], "yyyy-MM-dd HH:mm:ss"),
        (["01-03-2023 09:15:00", "31-12-2022 23:59:59"], "dd-MM-yyyy HH:mm:ss"),
        (["2024-05-31 20:14:19.993", "2023-12-12 11:11:11.123"], "yyyy-MM-dd HH:mm:ss.SSS"),
    ],
)
def test_try_to_timestamp_valid_matches_to_timestamp(engine, spark, datetime_strs, fmt):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [(v,) for v in datetime_strs]

    actual = engine.build_df(rows, schema).select(F.try_to_timestamp("ts", fmt).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(
        SF.try_to_timestamp(SF.col("ts"), SF.lit(fmt)).alias("result")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_to_timestamp.default_malformed_returns_null")
def test_try_to_timestamp_malformed_returns_null(engine, spark):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [("2023-01-01 12:34:56",), ("not-a-date",), (None,)]
    actual = engine.build_df(rows, schema).select(F.try_to_timestamp("ts").alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.try_to_timestamp("ts").alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_to_timestamp.with_format_malformed_returns_null")
@pytest.mark.parametrize(
    "datetime_strs, fmt",
    [
        (["2023-01-01 12:34:56", "not-a-date", None], "yyyy-MM-dd HH:mm:ss"),
        (["2023-01-01 12:34:56", "01-03-2023 09:15:00", None], "yyyy-MM-dd HH:mm:ss"),
    ],
)
def test_try_to_timestamp_with_format_malformed_returns_null(engine, spark, datetime_strs, fmt):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [(v,) for v in datetime_strs]
    actual = engine.build_df(rows, schema).select(F.try_to_timestamp("ts", fmt).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(
        SF.try_to_timestamp(SF.col("ts"), SF.lit(fmt)).alias("result")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_to_timestamp.iso_t_separator_returns_null")
def test_try_to_timestamp_format_iso_t_separator_returns_null(engine, spark):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [("2024-03-15T10:20:30",), ("2024-03-15 10:20:30",)]
    fmt = "yyyy-MM-dd HH:mm:ss"
    actual = engine.build_df(rows, schema).select(F.try_to_timestamp("ts", fmt).alias("t"))
    expected = spark.createDataFrame(rows, schema).selectExpr(f"try_to_timestamp(ts, '{fmt}') as t")
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_timestamp.iso_t_separator_raises")
def test_to_timestamp_with_format_iso_t_separator_raises(engine, spark):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [("2024-03-15T10:20:30",)]
    fmt = "yyyy-MM-dd HH:mm:ss"
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.to_timestamp("ts", fmt).alias("t")))
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).selectExpr(f"to_timestamp(ts, '{fmt}') as t").collect()


@pytest.mark.feature("functions.try_to_timestamp.column_input")
def test_try_to_timestamp_accepts_column_input(engine, spark):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [("2023-01-01 12:34:56",)]
    actual = engine.build_df(rows, schema).select(F.try_to_timestamp(F.col("ts")).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.try_to_timestamp("ts").alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_to_date.valid_matches_to_date")
@pytest.mark.parametrize(
    "date_strs, fmt",
    [
        (["1997-02-28", "2024-12-31"], "yyyy-MM-dd"),
        (["28-02-1997", "31-12-2024"], "dd-MM-yyyy"),
    ],
)
def test_try_to_date_valid_matches_to_date(engine, spark, date_strs, fmt):
    F = engine.functions
    schema = _schema(("d", _STR))
    rows = [(v,) for v in date_strs]
    actual = engine.build_df(rows, schema).select(F.try_to_date("d", fmt).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.try_to_date("d", fmt).alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_to_date.default_format_malformed")
def test_try_to_date_default_format_malformed_returns_null(engine, spark):
    F = engine.functions
    schema = _schema(("d", _STR))
    rows = [("1997-02-28",), ("2024-12-31",), ("bad",), (None,)]
    actual = engine.build_df(rows, schema).select(F.try_to_date("d").alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.try_to_date("d").alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_to_date.custom_format")
def test_try_to_date_custom_format_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("d", _STR))
    rows = [("28-02-1997",), ("31-12-2024",)]
    actual = engine.build_df(rows, schema).select(F.try_to_date("d", "dd-MM-yyyy").alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.try_to_date("d", "dd-MM-yyyy").alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_to_date.custom_format_malformed")
def test_try_to_date_custom_format_malformed_returns_null(engine, spark):
    F = engine.functions
    schema = _schema(("d", _STR))
    rows = [("28-02-1997",), ("31-12-2024",), ("not-a-date",), ("1997-02-28",), (None,)]
    actual = engine.build_df(rows, schema).select(F.try_to_date("d", "dd-MM-yyyy").alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.try_to_date("d", "dd-MM-yyyy").alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_to_date.column_input")
def test_try_to_date_accepts_column_input(engine, spark):
    F = engine.functions
    schema = _schema(("d", _STR))
    rows = [("2024-01-01",)]
    actual = engine.build_df(rows, schema).select(F.try_to_date(F.col("d")).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(SF.try_to_date("d").alias("result"))
    assert_matches_spark(actual, expected, engine)


# ----- try_element_at / element_at parity ----- #

_STR_ARRAY_SCHEMA = _schema(("arr", SparkArrayType(_STR)))
_STR_ARRAY_ROWS = [(["a", "b", "c"],)]
_MAP_STR_DBL_SCHEMA = _schema(("m", SparkMapType(_STR, _DOUBLE)))
_MAP_STR_DBL_ROWS = [({"a": 1.0, "b": 2.0},)]


@pytest.mark.feature("functions.try_element_at.array_positive_index")
def test_try_element_at_array_positive_index(engine, spark):
    F = engine.functions
    actual = engine.build_df(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(F.try_element_at("arr", 1).alias("v"))
    expected = spark.createDataFrame(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(
        SF.try_element_at(SF.col("arr"), SF.lit(1)).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_element_at.array_last_element")
def test_try_element_at_array_last_element(engine, spark):
    F = engine.functions
    actual = engine.build_df(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(F.try_element_at("arr", 3).alias("v"))
    expected = spark.createDataFrame(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(
        SF.try_element_at(SF.col("arr"), SF.lit(3)).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_element_at.array_negative_index")
def test_try_element_at_array_negative_index(engine, spark):
    F = engine.functions
    actual = engine.build_df(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(F.try_element_at("arr", -1).alias("v"))
    expected = spark.createDataFrame(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(
        SF.try_element_at(SF.col("arr"), SF.lit(-1)).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_element_at.array_oob_returns_null")
def test_try_element_at_array_oob_returns_null(engine, spark):
    F = engine.functions
    actual = engine.build_df(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(F.try_element_at("arr", 4).alias("v"))
    expected = spark.createDataFrame(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(
        SF.try_element_at(SF.col("arr"), SF.lit(4)).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_element_at.map_literal_key_via_lit")
def test_try_element_at_map_literal_key_via_lit_column(engine, spark):
    """Literal map keys use ``lit`` in PySpark; ``try_element_at`` does not treat bare strings as literals."""
    F = engine.functions
    actual = engine.build_df(_MAP_STR_DBL_ROWS, _MAP_STR_DBL_SCHEMA).select(
        F.try_element_at("m", F.lit("a")).alias("v")
    )
    expected = spark.createDataFrame(_MAP_STR_DBL_ROWS, _MAP_STR_DBL_SCHEMA).select(
        SF.try_element_at(SF.col("m"), SF.lit("a")).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_element_at.map_string_extraction_is_column_name")
def test_try_element_at_map_string_extraction_is_column_name(engine, spark):
    """``try_element_at(map, 'col')`` uses column ``col`` as the key (SPARK-48766), not literal ``'col'``."""
    F = engine.functions
    schema = _schema(("m", SparkMapType(_STR, _DOUBLE)), ("lookup", _STR))
    rows = [({"a": 1.0, "b": 2.0}, "a")]
    actual = engine.build_df(rows, schema).select(F.try_element_at("m", "lookup").alias("v"))
    expected = spark.createDataFrame(rows, schema).select(SF.try_element_at(SF.col("m"), SF.col("lookup")).alias("v"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_element_at.accepts_column")
def test_try_element_at_accepts_column_input(engine, spark):
    F = engine.functions
    rows = [(["x", "y"],)]
    actual = engine.build_df(rows, _STR_ARRAY_SCHEMA).select(F.try_element_at(F.col("arr"), 1).alias("v"))
    expected = spark.createDataFrame(rows, _STR_ARRAY_SCHEMA).select(
        SF.try_element_at(SF.col("arr"), SF.lit(1)).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.element_at.array_valid")
def test_element_at_array_valid_index_matches_spark(engine, spark):
    F = engine.functions
    actual = engine.build_df(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(F.element_at("arr", 2).alias("v"))
    expected = spark.createDataFrame(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(
        SF.element_at(SF.col("arr"), SF.lit(2)).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.element_at.array_oob_raises")
def test_element_at_array_oob_raises_like_spark(engine, spark):
    """Spark 4 ANSI: out-of-bounds ``element_at`` raises; ``try_element_at`` returns null."""
    F = engine.functions
    with pytest.raises(Exception):
        spark.createDataFrame(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(
            SF.element_at(SF.col("arr"), SF.lit(4))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(F.element_at("arr", 4)))
    # Lenient counterpart must not raise.
    engine.to_records(
        engine.build_df(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(F.try_element_at("arr", 4).alias("v"))
    )


@pytest.mark.feature("functions.element_at.array_zero_raises")
def test_element_at_array_zero_index_raises_like_spark(engine, spark):
    F = engine.functions
    with pytest.raises(Exception):
        spark.createDataFrame(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(
            SF.element_at(SF.col("arr"), SF.lit(0))
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(_STR_ARRAY_ROWS, _STR_ARRAY_SCHEMA).select(F.element_at("arr", 0)))


@pytest.mark.feature("functions.element_at.map_literal_string_key")
def test_element_at_map_literal_string_key_matches_spark(engine, spark):
    """``element_at(map, 'key')`` uses a literal key (unlike ``try_element_at`` string = column name)."""
    F = engine.functions
    actual = engine.build_df(_MAP_STR_DBL_ROWS, _MAP_STR_DBL_SCHEMA).select(F.element_at("m", "a").alias("v"))
    expected = spark.createDataFrame(_MAP_STR_DBL_ROWS, _MAP_STR_DBL_SCHEMA).select(
        SF.element_at(SF.col("m"), SF.lit("a")).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.element_at.map_missing_literal_key_returns_null")
def test_element_at_map_missing_literal_key_returns_null(engine, spark):
    F = engine.functions
    actual = engine.build_df(_MAP_STR_DBL_ROWS, _MAP_STR_DBL_SCHEMA).select(F.element_at("m", "c").alias("v"))
    expected = spark.createDataFrame(_MAP_STR_DBL_ROWS, _MAP_STR_DBL_SCHEMA).select(
        SF.element_at(SF.col("m"), SF.lit("c")).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.element_at.map_computed_key_keeps_value_type")
def test_element_at_map_computed_key_keeps_value_type(engine, spark):
    """A computed map key returns the map's value type, so string functions (``lower``) apply to it.
    Covers a matching key, no matching key (null key) and a null map."""
    F = engine.functions
    schema = _schema(("m", SparkMapType(_STR, _STR)))
    rows = [({"referral_src": "FaceBook", "gclid": "x"},), ({"a": "b"},), (None,)]
    actual = engine.build_df(rows, schema).select(
        F.lower(
            F.element_at(
                F.col("m"),
                F.try_element_at(
                    F.filter(F.map_keys(F.col("m")), lambda x: F.lower(x).contains("referral")), F.lit(1)
                ),
            )
        ).alias("v")
    )
    expected = spark.createDataFrame(rows, schema).select(
        SF.lower(
            SF.element_at(
                SF.col("m"),
                SF.try_element_at(
                    SF.filter(SF.map_keys(SF.col("m")), lambda x: SF.lower(x).contains("referral")), SF.lit(1)
                ),
            )
        ).alias("v")
    )
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestSubstring / TestArrayFunctions / TestDateFunctions
# ============================================================================ #


@pytest.mark.feature("functions.substring")
def test_substring_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [("hELlo woRLd",), (None,), ("",)]
    actual = engine.build_df(rows, schema).select(F.substring("s", 2, 3).alias("sub"))
    expected = spark.createDataFrame(rows, schema).select(SF.substring("s", 2, 3).alias("sub"))
    assert_matches_spark(actual, expected, engine)


_ARR_LONG_SCHEMA = _schema(("arr", SparkArrayType(_LONG)))


@pytest.mark.feature("functions.array_contains_and_size")
def test_array_contains_and_size_against_spark(engine, spark):
    F = engine.functions
    rows = [([1, 2, 3],), ([10],)]
    actual = engine.build_df(rows, _ARR_LONG_SCHEMA).select(
        F.array_contains("arr", 2).alias("has2"),
        F.size("arr").alias("sz"),
    )
    expected = spark.createDataFrame(rows, _ARR_LONG_SCHEMA).select(
        SF.array_contains("arr", SF.lit(2)).alias("has2"),
        SF.size("arr").alias("sz"),
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.array_filter_and_transform")
def test_array_filter_and_transform_against_spark(engine, spark):
    F = engine.functions
    rows = [([1, 2, 3],), ([10],)]
    actual = engine.build_df(rows, _ARR_LONG_SCHEMA).select(
        F.filter("arr", lambda c: c > 1).alias("flt"),
        F.transform("arr", lambda c: c * 2).alias("dbl"),
    )
    expected = spark.createDataFrame(rows, _ARR_LONG_SCHEMA).select(
        SF.filter("arr", lambda x: x > 1).alias("flt"),
        SF.transform("arr", lambda x: x * 2).alias("dbl"),
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.explode")
def test_explode_against_spark(engine, spark):
    F = engine.functions
    rows = [([1, 2],), ([3],)]
    actual = engine.build_df(rows, _ARR_LONG_SCHEMA).select(F.explode("arr").alias("e"))
    expected = spark.createDataFrame(rows, _ARR_LONG_SCHEMA).select(SF.explode("arr").alias("e"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.explode.non_collection_raises")
def test_explode_non_collection_raises(engine, spark):
    """``explode`` needs an array or map input."""
    F = engine.functions
    str_rows, str_schema = [("x",)], _schema(("s", _STR))
    with pytest.raises(Exception):
        spark.createDataFrame(str_rows, str_schema).select(SF.explode("s")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(str_rows, str_schema).select(F.explode("s")))


@pytest.mark.feature("functions.explode.drops_null_and_empty")
def test_explode_drops_null_and_empty_against_spark(engine, spark):
    """``explode`` emits no row for a null or empty array (``explode_outer`` would emit a null)."""
    F = engine.functions
    schema = _schema(("k", _STR), ("arr", SparkArrayType(_LONG)))
    rows = [("a", [1, 2]), ("b", None), ("c", [])]
    actual = engine.build_df(rows, schema).withColumn("e", F.explode("arr")).select("k", "e")
    expected = spark.createDataFrame(rows, schema).withColumn("e", SF.explode("arr")).select("k", "e")
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.date_sub_and_datediff")
def test_date_sub_and_datediff_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("d", _STR))
    rows = [(None,), ("2024-01-10",), ("2024-01-01",)]
    actual = engine.build_df(rows, schema).select(
        F.date_sub("d", 2).alias("sub"),
        F.datediff(F.lit("2024-01-20"), "d").alias("dd"),
    )
    expected = spark.createDataFrame(rows, schema).select(
        SF.date_sub(SF.col("d"), SF.lit(2)).alias("sub"),
        SF.datediff(SF.lit("2024-01-20"), SF.col("d")).alias("dd"),
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.datediff.iso8601_string")
def test_datediff_iso8601_string_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _STR))
    rows = [("2026-04-26T00:00:00Z",)]
    end = "2026-04-27"
    actual = engine.build_df(rows, schema).select(F.datediff(F.lit(end), "a").alias("d"))
    expected = spark.createDataFrame(rows, schema).select(SF.datediff(SF.lit(end), SF.col("a")).alias("d"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.datetime_ge_date_sub")
def test_datetime_ge_date_sub_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("created", _STR))
    rows = [("2026-04-26T00:00:00Z",), ("2026-01-01T00:00:00Z",), (None,)]
    actual = engine.build_df(rows, schema).select(
        (F.col("created") >= F.date_sub(F.current_date(), 30)).alias("passes_30d")
    )
    expected = spark.createDataFrame(rows, schema).select(
        (SF.col("created") >= SF.date_sub(SF.current_date(), SF.lit(30))).alias("passes_30d")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.months_between")
def test_months_between_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("start", _STR), ("end", _STR))
    rows = [("2024-01-15", "2024-03-20")]
    actual = engine.build_df(rows, schema).select(F.months_between("end", "start").alias("mb"))
    expected = spark.createDataFrame(rows, schema).select(SF.months_between("end", "start").alias("mb"))
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestMapFunctions / TestFromJson / TestFirstAgg
# ============================================================================ #


@pytest.mark.feature("functions.map_keys")
def test_map_keys_against_spark(engine, spark):
    F = engine.functions
    actual = engine.build_df(_MAP_STR_DBL_ROWS, _MAP_STR_DBL_SCHEMA).select(F.map_keys("m").alias("keys"))
    expected = spark.createDataFrame(_MAP_STR_DBL_ROWS, _MAP_STR_DBL_SCHEMA).select(SF.map_keys("m").alias("keys"))
    assert_matches_spark(actual, expected, engine)


_KV_ENTRIES_SCHEMA = _schema(
    (
        "entries",
        SparkArrayType(SparkStructType([SparkStructField("key", _STR), SparkStructField("value", _LONG)])),
    )
)


@pytest.mark.feature("functions.map_from_entries.long_value")
def test_map_from_entries_against_spark(engine, spark):
    """map_from_entries produces a map that getItem can look up by key, matching Spark's MapType semantics."""
    F = engine.functions
    rows = [([("a", 1), ("b", 2)],)]
    actual = (
        engine.build_df(rows, _KV_ENTRIES_SCHEMA)
        .select(F.map_from_entries("entries").alias("m"))
        .select(F.col("m").getItem("a").alias("val_a"))
    )
    expected = (
        spark.createDataFrame(rows, _KV_ENTRIES_SCHEMA)
        .select(SF.map_from_entries("entries").alias("m"))
        .select(SF.col("m").getItem("a").alias("val_a"))
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.from_json.struct")
def test_from_json_struct_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("j", _STR))
    rows = [('{"field1": "hello", "field2": 999}',)]
    parsed_schema = SparkStructType([SparkStructField("field1", _STR), SparkStructField("field2", _INT)])
    actual = engine.build_df(rows, schema).select(F.from_json("j", engine.dtype(parsed_schema)).alias("parsed"))
    expected = spark.createDataFrame(rows, schema).select(SF.from_json("j", parsed_schema).alias("parsed"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.from_json.ddl_schema")
@pytest.mark.parametrize(
    "ddl",
    ["a int, b string", "a INT, b STRING, c array<bigint>", "struct<a:int,b:string>", "a struct<x int>"],
)
def test_from_json_ddl_schema_against_spark(engine, spark, ddl):
    """A DDL schema string is a field list (a struct) or a single struct/array/map type, as ``DataType.fromDDL``."""
    F = engine.functions
    schema = _schema(("j", _STR))
    rows = [
        ('{"a": 1, "b": "x", "c": [1, 2]}',),
        ('{"a": "not-an-int", "b": "y"}',),
        ('{"b": "z"}',),
        ("not json",),
        (None,),
    ]
    actual = engine.build_df(rows, schema).select("j", F.from_json("j", ddl).alias("p"))
    expected = spark.createDataFrame(rows, schema).select("j", SF.from_json("j", ddl).alias("p"))
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.from_json.ddl_schema")
def test_from_json_ddl_array_schema_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("j", _STR))
    rows = [('["a", "b"]',), ("[]",), ('{"not": "an array"}',), (None,)]
    actual = engine.build_df(rows, schema).select("j", F.from_json("j", "array<string>").alias("p"))
    expected = spark.createDataFrame(rows, schema).select("j", SF.from_json("j", "array<string>").alias("p"))
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.from_json.ddl_schema")
@pytest.mark.parametrize(
    "ddl",
    ["a intx", "a int,", "a: int, b: string", "int", "string"],
    ids=["unknown_type", "trailing_comma", "top_level_colon", "int", "string"],
)
def test_from_json_malformed_ddl_schema_raises_like_spark(engine, spark, ddl):
    """An unparseable DDL, or one naming a type that is not a struct/array/map, is rejected.

    Unlike ``createDataFrame``'s PySpark-side parser, ``fromDDL`` takes no ``name: type`` field list.
    """
    F = engine.functions
    schema = _schema(("j", _STR))
    rows = [('{"a": 1}',)]
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.from_json("j", ddl)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.from_json("j", ddl)))


@pytest.mark.feature("functions.first_agg")
def test_first_agg_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("g", _LONG), ("v", _LONG))
    rows = [(1, 10), (1, 20), (2, 30), (2, 40)]
    actual = engine.build_df(rows, schema).sort("g", "v").groupBy("g").agg(F.first("v").alias("fv"))
    expected = spark.createDataFrame(rows, schema).orderBy("g", "v").groupBy("g").agg(SF.first("v").alias("fv"))
    assert_matches_spark(actual.orderBy("g"), expected.orderBy("g"), engine)


# ============================================================================ #
# TestWhenLitNonePreservesType
# ============================================================================ #


@pytest.mark.feature("functions.when.lit_none_preserves_double")
def test_when_lit_none_preserves_double_type(engine, spark):
    """lit(None) inside when/then must not force the result to String; Spark infers from non-null branch."""
    F = engine.functions
    schema = _schema(("price", _DOUBLE))
    rows = [(10.0,), (-1.0,), (5.0,)]
    actual = engine.build_df(rows, schema).withColumn(
        "price", F.when(F.col("price") <= 0, F.lit(None)).otherwise(F.col("price"))
    )
    expected = spark.createDataFrame(rows, schema).withColumn(
        "price", SF.when(SF.col("price") <= 0, SF.lit(None)).otherwise(SF.col("price"))
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.when.lit_none_preserves_integer")
def test_when_lit_none_preserves_integer_type(engine, spark):
    F = engine.functions
    schema = _schema(("v", _LONG))
    rows = [(1,), (-2,), (3,)]
    actual = engine.build_df(rows, schema).withColumn("v", F.when(F.col("v") < 0, F.lit(None)).otherwise(F.col("v")))
    expected = spark.createDataFrame(rows, schema).withColumn(
        "v", SF.when(SF.col("v") < 0, SF.lit(None)).otherwise(SF.col("v"))
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.when.lit_none_preserves_string")
def test_when_lit_none_preserves_string_type(engine, spark):
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [("hello",), ("",), ("world",)]
    actual = engine.build_df(rows, schema).withColumn("s", F.when(F.col("s") == "", F.lit(None)).otherwise(F.col("s")))
    expected = spark.createDataFrame(rows, schema).withColumn(
        "s", SF.when(SF.col("s") == "", SF.lit(None)).otherwise(SF.col("s"))
    )
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestToJsonParity / TestLeastGreatest / TestCreateMap / TestArrayParity
# ============================================================================ #


@pytest.mark.feature("functions.to_json.struct")
def test_to_json_struct_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _STR))
    rows = [(1, "x"), (2, "y")]
    actual = engine.build_df(rows, schema).select(F.to_json(F.struct("a", "b")).alias("j"))
    expected = spark.createDataFrame(rows, schema).select(SF.to_json(SF.struct("a", "b")).alias("j"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_json.decimal_scale_preserved")
def test_to_json_decimal_scale_preserved_against_spark(engine, spark):
    """Decimals are written as JSON numbers at the column's scale (``0.0650``, not ``0.065`` or ``"0.0650"``)."""
    F = engine.functions
    schema = _schema(("amount", SparkDecimalType(13, 2)), ("apr", SparkDecimalType(5, 4)), ("rate", _DOUBLE))
    rows = [(Decimal("180000.00"), Decimal("0.0650"), 0.065), (None, Decimal("1.5"), None)]
    options = {"ignoreNullFields": False}
    actual = engine.build_df(rows, schema).select(F.to_json(F.struct("amount", "apr", "rate"), options).alias("j"))
    expected = spark.createDataFrame(rows, schema).select(
        SF.to_json(SF.struct("amount", "apr", "rate"), options).alias("j")
    )
    assert_matches_spark(actual, expected, engine)

    # Malformed: to_json needs a struct, map or array.
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.to_json("amount")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.to_json("amount")))


@pytest.mark.feature("functions.to_json.timestamp_format")
@pytest.mark.parametrize(
    "options",
    [{}, {"timestampFormat": "yyyy-MM-dd HH:mm:ss", "dateFormat": "dd/MM/yyyy"}],
    ids=["default_formats", "custom_formats"],
)
def test_to_json_timestamp_format_against_spark(engine, spark, options):
    """Timestamps default to ``yyyy-MM-dd'T'HH:mm:ss.SSSXXX`` in the session zone; dates to ``yyyy-MM-dd``."""
    F = engine.functions
    schema = _schema(("t", SparkTimestampType()), ("d", SparkDateType()))
    rows = [(datetime(2026, 10, 1, 15, 32, 10), date(2026, 10, 1)), (datetime(2024, 2, 29, 0, 0, 0, 123000), None)]
    actual = engine.build_df(rows, schema).select(F.to_json(F.struct("t", "d"), options).alias("j"))
    expected = spark.createDataFrame(rows, schema).select(SF.to_json(SF.struct("t", "d"), options).alias("j"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_json.array")
def test_to_json_array_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG))
    rows = [(1, 3), (2, 4)]
    actual = engine.build_df(rows, schema).select(F.to_json(F.array("a", "b")).alias("j"))
    expected = spark.createDataFrame(rows, schema).select(SF.to_json(SF.array("a", "b")).alias("j"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.least.basic")
def test_least_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG), ("c", _LONG))
    rows = [(10, 5, 8), (1, None, 2), (None, 3, 7)]
    actual = engine.build_df(rows, schema).select(F.least("a", "b", "c").alias("min"))
    expected = spark.createDataFrame(rows, schema).select(SF.least("a", "b", "c").alias("min"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.greatest.basic")
def test_greatest_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG), ("c", _LONG))
    rows = [(10, 5, 8), (1, None, 2), (None, 3, 7)]
    actual = engine.build_df(rows, schema).select(F.greatest("a", "b", "c").alias("max"))
    expected = spark.createDataFrame(rows, schema).select(SF.greatest("a", "b", "c").alias("max"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.least.all_nulls")
def test_least_all_nulls_returns_null(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG))
    rows = [(None, None), (None, None)]
    actual = engine.build_df(rows, schema).select(F.least("a", "b").alias("min"))
    expected = spark.createDataFrame(rows, schema).select(SF.least("a", "b").alias("min"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.create_map.basic")
def test_create_map_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("k", _STR), ("v", _LONG))
    rows = [("a", 1), ("b", 2)]
    actual = engine.build_df(rows, schema).select(F.to_json(F.create_map("k", "v")).alias("j"))
    expected = spark.createDataFrame(rows, schema).select(SF.to_json(SF.create_map("k", "v")).alias("j"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.array.basic")
def test_array_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG))
    rows = [(1, 4), (2, 5), (3, 6)]
    actual = engine.build_df(rows, schema).select(F.array("a", "b").alias("arr"))
    expected = spark.createDataFrame(rows, schema).select(SF.array("a", "b").alias("arr"))
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestDateFormat / TestFloor / TestPow / TestIsnan / TestTryDivide / TestNullif
# ============================================================================ #


@pytest.mark.feature("functions.date_format.basic")
def test_date_format_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [("2023-01-15 10:30:45",), ("2024-12-25 00:00:00",)]
    actual = engine.build_df(rows, schema).select(F.date_format(F.to_timestamp("ts"), "yyyy-MM-dd").alias("d"))
    expected = spark.createDataFrame(rows, schema).select(
        SF.date_format(SF.to_timestamp("ts"), "yyyy-MM-dd").alias("d")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.date_format.time_parts")
def test_date_format_time_parts_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("ts", _STR))
    rows = [("2023-06-15 14:05:09",)]
    actual = engine.build_df(rows, schema).select(F.date_format(F.to_timestamp("ts"), "HH:mm:ss").alias("t"))
    expected = spark.createDataFrame(rows, schema).select(SF.date_format(SF.to_timestamp("ts"), "HH:mm:ss").alias("t"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.floor")
def test_floor_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("v", _DOUBLE))
    rows = [(1.9,), (2.1,), (-0.5,), (0.0,), (None,)]
    actual = engine.build_df(rows, schema).select(F.floor("v").alias("f"))
    expected = spark.createDataFrame(rows, schema).select(SF.floor("v").alias("f"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.pow.col_col")
def test_pow_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("base", _DOUBLE), ("exp", _DOUBLE))
    rows = [(2.0, 3.0), (3.0, 2.0), (10.0, 0.0)]
    actual = engine.build_df(rows, schema).select(F.pow("base", "exp").alias("p"))
    expected = spark.createDataFrame(rows, schema).select(SF.pow("base", "exp").alias("p"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.pow.literal_exponent")
def test_pow_with_literal_exponent_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("base", _DOUBLE))
    rows = [(2.0,), (3.0,), (4.0,)]
    actual = engine.build_df(rows, schema).select(F.pow(F.col("base"), F.lit(2)).alias("p"))
    expected = spark.createDataFrame(rows, schema).select(SF.pow(SF.col("base"), SF.lit(2)).alias("p"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.pow.integral_operands_return_double")
@pytest.mark.parametrize("dtype", [SparkByteType(), SparkShortType(), _INT, _LONG, SparkFloatType()])
def test_pow_non_double_numeric_operands_against_spark(engine, spark, dtype):
    # Spark casts both operands to double, so 2 ** 3 is 8.0 (not 8) and 2 ** -1 is 0.5.
    F = engine.functions
    schema = _schema(("base", dtype), ("exp", dtype))
    rows = [(2, 3), (3, 0), (2, -1), (5, 2)]
    if isinstance(dtype, SparkFloatType):
        rows = [(float(b), float(e)) for b, e in rows] + [(1.5, 0.5)]
    actual = engine.build_df(rows, schema).select(F.pow("base", "exp").alias("p"))
    expected = spark.createDataFrame(rows, schema).select(SF.pow("base", "exp").alias("p"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.pow.decimal_operands")
def test_pow_decimal_operands_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("base", SparkDecimalType(10, 2)), ("exp", SparkDecimalType(10, 2)))
    rows = [(Decimal("1.50"), Decimal("2.00")), (Decimal("4.00"), Decimal("0.50")), (None, Decimal("1.00"))]
    actual = engine.build_df(rows, schema).select(F.pow("base", "exp").alias("p"))
    expected = spark.createDataFrame(rows, schema).select(SF.pow("base", "exp").alias("p"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.pow.null_operands")
def test_pow_null_operands_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("base", _DOUBLE), ("exp", _DOUBLE))
    rows = [(None, 2.0), (2.0, None), (None, None), (None, 0.0), (3.0, 2.0)]
    df = engine.build_df(rows, schema)
    sdf = spark.createDataFrame(rows, schema)
    actual = df.select(
        F.pow("base", "exp").alias("p"),
        F.pow(F.col("base"), F.lit(None)).alias("null_exp"),
        F.pow(F.lit(None), F.col("exp")).alias("null_base"),
    )
    expected = sdf.select(
        SF.pow("base", "exp").alias("p"),
        SF.pow(SF.col("base"), SF.lit(None)).alias("null_exp"),
        SF.pow(SF.lit(None), SF.col("exp")).alias("null_base"),
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.pow.string_operand_casts_to_double")
def test_pow_string_operands_against_spark(engine, spark):
    # Spark implicitly casts string operands to double (whitespace is trimmed).
    F = engine.functions
    schema = _schema(("base", _STR), ("exp", _DOUBLE))
    rows = [("2", 3.0), (" 4 ", 0.5), ("1.5", 2.0), (None, 1.0)]
    actual = engine.build_df(rows, schema).select(
        F.pow("base", "exp").alias("p"), F.pow("exp", "base").alias("swapped")
    )
    expected = spark.createDataFrame(rows, schema).select(
        SF.pow("base", "exp").alias("p"), SF.pow("exp", "base").alias("swapped")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.pow.malformed_string_raises")
def test_pow_malformed_string_raises_like_spark(engine, spark):
    # ANSI: the implicit string -> double cast is strict, so a malformed string fails the query.
    F = engine.functions
    schema = _schema(("base", _STR), ("exp", _DOUBLE))
    rows = [("2", 3.0), ("abc", 2.0)]
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        spark.createDataFrame(rows, schema).select(SF.pow("base", "exp")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.pow("base", "exp").alias("p")))


@pytest.mark.feature("functions.pow.non_numeric_operand_raises")
@pytest.mark.parametrize("dtype, value", [(SparkBooleanType(), True), (SparkDateType(), date(2024, 1, 1))])
def test_pow_non_numeric_operand_raises_like_spark(engine, spark, dtype, value):
    # Spark rejects booleans / dates at analysis time (DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE).
    F = engine.functions
    schema = _schema(("base", dtype), ("exp", _DOUBLE))
    rows = [(value, 2.0)]
    with pytest.raises(Exception, match="DATATYPE_MISMATCH"):
        spark.createDataFrame(rows, schema).select(SF.pow("base", "exp")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.pow("base", "exp").alias("p")))


def _assert_pow_nan_positions_match(actual, expected, engine):
    """The oracle normalizes NaN to null, so additionally pin *which* ids are NaN (not null)."""

    def nan_ids(records):
        return sorted(r["id"] for r in records if isinstance(r["p"], float) and math.isnan(r["p"]))

    assert nan_ids(engine.to_records(actual)) == nan_ids([r.asDict() for r in expected.collect()])


@pytest.mark.feature("functions.pow.ieee_special_values")
def test_pow_ieee_special_values_against_spark(engine, spark):
    # Where Python's math.pow raises, Spark (Java Math.pow) returns +-Infinity or NaN.
    F = engine.functions
    schema = _schema(("id", _INT), ("base", _DOUBLE), ("exp", _DOUBLE))
    rows = [
        (1, 0.0, -1.0),  # +inf
        (2, -0.0, -1.0),  # -inf: -0.0 to a negative odd integer keeps its sign
        (3, -0.0, -2.0),  # +inf
        (4, -8.0, 1 / 3),  # NaN: negative base, non-integer exponent
        (5, 10.0, 400.0),  # +inf: overflow
        (6, -10.0, 401.0),  # -inf: overflow, negative base, odd exponent
        (7, -10.0, 400.0),  # +inf: overflow, negative base, even exponent
        (8, float("nan"), 0.0),  # 1.0: anything to the 0 is 1
        (9, 2.0, float("-inf")),  # 0.0
        (10, float("inf"), -1.0),  # 0.0
        (11, 2.0, -1074.0),  # smallest subnormal, no underflow to 0
    ]
    actual = engine.build_df(rows, schema).select("id", F.pow("base", "exp").alias("p"))
    expected = spark.createDataFrame(rows, schema).select("id", SF.pow("base", "exp").alias("p"))
    assert_matches_spark(actual, expected, engine)
    _assert_pow_nan_positions_match(actual, expected, engine)


@pytest.mark.feature("functions.pow.java_nan_semantics")
def test_pow_java_nan_semantics_against_spark(engine, spark):
    # Java Math.pow departs from C99 here: 1 ** NaN and (+-1) ** +-inf are NaN, not 1.0.
    F = engine.functions
    schema = _schema(("id", _INT), ("base", _DOUBLE), ("exp", _DOUBLE))
    rows = [
        (1, 1.0, float("nan")),
        (2, -1.0, float("inf")),
        (3, 1.0, float("-inf")),
        (4, float("nan"), 1.0),
        (5, 2.0, float("nan")),
    ]
    actual = engine.build_df(rows, schema).select("id", F.pow("base", "exp").alias("p"))
    expected = spark.createDataFrame(rows, schema).select("id", SF.pow("base", "exp").alias("p"))
    assert_matches_spark(actual, expected, engine)
    _assert_pow_nan_positions_match(actual, expected, engine)


@pytest.mark.feature("column.pow.operator")
def test_column_pow_operator_against_spark(engine, spark):
    # ``col ** x`` and reflected ``x ** col`` are Spark's ``pow``; 2 ** e vs e ** 2 pins operand order.
    F = engine.functions
    schema = _schema(("base", _INT), ("exp", _INT))
    rows = [(2, 3), (3, 2), (4, None)]
    actual = engine.build_df(rows, schema).select(
        (F.col("base") ** F.col("exp")).alias("col_col"),
        (F.col("exp") ** 2).alias("col_lit"),
        (2 ** F.col("exp")).alias("lit_col"),
        (0.5 ** F.col("base")).alias("float_lit_col"),
    )
    expected = spark.createDataFrame(rows, schema).select(
        (SF.col("base") ** SF.col("exp")).alias("col_col"),
        (SF.col("exp") ** 2).alias("col_lit"),
        (2 ** SF.col("exp")).alias("lit_col"),
        (0.5 ** SF.col("base")).alias("float_lit_col"),
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.isnan")
def test_isnan_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("v", _DOUBLE))
    rows = [(1.0,), (float("nan"),), (None,), (0.0,)]
    actual = engine.build_df(rows, schema).select(F.isnan("v").alias("n"))
    expected = spark.createDataFrame(rows, schema).select(SF.isnan("v").alias("n"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.try_divide")
def test_try_divide_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _DOUBLE), ("b", _DOUBLE))
    rows = [(10.0, 2.0), (9.0, 0.0), (None, 3.0), (5.0, None)]
    actual = engine.build_df(rows, schema).select(F.try_divide("a", "b").alias("d"))
    expected = spark.createDataFrame(rows, schema).select(SF.try_divide("a", "b").alias("d"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.nullif.basic")
def test_nullif_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG))
    rows = [(1, 1), (2, 3), (3, 3), (None, None)]
    actual = engine.build_df(rows, schema).select(F.nullif("a", "b").alias("n"))
    expected = spark.createDataFrame(rows, schema).select(SF.nullif("a", "b").alias("n"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.nullif.preserves_e1")
def test_nullif_with_nulls_preserves_e1(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG))
    rows = [(5, None), (None, 2), (3, None)]
    actual = engine.build_df(rows, schema).select(F.nullif("a", "b").alias("n"))
    expected = spark.createDataFrame(rows, schema).select(SF.nullif("a", "b").alias("n"))
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestSortArray / TestCount / TestAbs (extra)
# ============================================================================ #


@pytest.mark.feature("functions.sort_array.asc")
def test_sort_array_asc_against_spark(engine, spark):
    F = engine.functions
    rows = [([3, 1, 2],), ([6, 4, 5],), (None,)]
    actual = engine.build_df(rows, _ARR_LONG_SCHEMA).select(F.sort_array("arr").alias("s"))
    expected = spark.createDataFrame(rows, _ARR_LONG_SCHEMA).select(SF.sort_array("arr").alias("s"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.sort_array.desc")
def test_sort_array_desc_against_spark(engine, spark):
    F = engine.functions
    rows = [([3, 1, 2],), ([6, 4, 5],)]
    actual = engine.build_df(rows, _ARR_LONG_SCHEMA).select(F.sort_array("arr", asc=False).alias("s"))
    expected = spark.createDataFrame(rows, _ARR_LONG_SCHEMA).select(SF.sort_array("arr", asc=False).alias("s"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.count.star")
def test_count_star_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("g", _STR), ("v", _LONG))
    rows = [("a", 1), ("a", None), ("b", 3), ("b", 4), ("b", None)]
    actual = engine.build_df(rows, schema).groupBy("g").agg(F.count("*").alias("cnt")).orderBy("g")
    expected = spark.createDataFrame(rows, schema).groupBy("g").agg(SF.count("*").alias("cnt")).orderBy("g")
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.count.column")
def test_count_column_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("g", _STR), ("v", _LONG))
    rows = [("a", 1), ("a", None), ("b", 3), ("b", 4), ("b", None)]
    actual = engine.build_df(rows, schema).groupBy("g").agg(F.count("v").alias("cnt")).orderBy("g")
    expected = spark.createDataFrame(rows, schema).groupBy("g").agg(SF.count("v").alias("cnt")).orderBy("g")
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.count.star_no_groupby")
def test_count_star_no_groupby_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("v", _LONG))
    rows = [(1,), (None,), (3,)]
    actual = engine.build_df(rows, schema).select(F.count("*").alias("cnt"))
    expected = spark.createDataFrame(rows, schema).select(SF.count("*").alias("cnt"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.abs.float")
def test_abs_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("v", _DOUBLE))
    rows = [(-3.5,), (0.0,), (2.1,), (-7.0,), (None,)]
    actual = engine.build_df(rows, schema).select(F.abs(F.col("v")).alias("a"))
    expected = spark.createDataFrame(rows, schema).select(SF.abs(SF.col("v")).alias("a"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.abs.integer")
def test_abs_integer_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("v", _LONG))
    rows = [(-10,), (0,), (5,), (-1,), (None,)]
    actual = engine.build_df(rows, schema).select(F.abs("v").alias("a"))
    expected = spark.createDataFrame(rows, schema).select(SF.abs("v").alias("a"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.abs.expression")
def test_abs_expression_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _DOUBLE), ("b", _DOUBLE))
    rows = [(1.0, 4.0), (5.0, 2.0), (3.0, 3.0)]
    actual = engine.build_df(rows, schema).withColumn("diff", F.abs(F.col("a") - F.col("b")))
    expected = spark.createDataFrame(rows, schema).withColumn("diff", SF.abs(SF.col("a") - SF.col("b")))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.abs.of_subtraction_withcolumn")
def test_abs_of_subtraction_withcolumn_against_spark(engine, spark):
    """Regression: arithmetic with unresolved dtypes chained with abs() via withColumn must produce a typed result."""
    F = engine.functions
    schema = _schema(("x", _DOUBLE), ("y", _DOUBLE))
    rows = [(1.0, 4.0), (5.0, 2.0), (3.0, 3.0)]
    actual = engine.build_df(rows, schema).withColumn("d", F.abs(F.col("x") - F.col("y")))
    expected = spark.createDataFrame(rows, schema).withColumn("d", SF.abs(SF.col("x") - SF.col("y")))
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestStructFieldNaming (parity subset)
# ============================================================================ #


@pytest.mark.feature("functions.struct.aliased_columns")
def test_struct_from_aliased_columns_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _STR))
    rows = [(1, "x"), (2, "y")]
    actual = engine.build_df(rows, schema).select(F.struct(F.col("a").alias("k"), F.col("b").alias("v")).alias("s"))
    expected = spark.createDataFrame(rows, schema).select(
        SF.struct(SF.col("a").alias("k"), SF.col("b").alias("v")).alias("s")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.struct.aliased_literals")
def test_struct_from_aliased_literals_against_spark(engine, spark):
    """lit(...).alias('name') inside struct must use the alias as the field name."""
    F = engine.functions
    schema = _schema(("x", _LONG))
    rows = [(1,), (2,)]
    actual = engine.build_df(rows, schema).select(
        F.struct(F.lit("aaa").alias("f1"), F.lit("bbb").alias("f2"), F.lit("ccc").alias("f3")).alias("s")
    )
    expected = spark.createDataFrame(rows, schema).select(
        SF.struct(SF.lit("aaa").alias("f1"), SF.lit("bbb").alias("f2"), SF.lit("ccc").alias("f3")).alias("s")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_json.struct_aliased_literals")
def test_to_json_struct_aliased_literals_against_spark(engine, spark):
    """to_json(struct(lit(...).alias(...), ...)) must use alias names as JSON keys."""
    F = engine.functions
    schema = _schema(("x", _LONG))
    rows = [(1,), (2,)]
    actual = engine.build_df(rows, schema).select(
        F.to_json(F.struct(F.lit("aaa").alias("f1"), F.lit("bbb").alias("f2"), F.lit("ccc").alias("f3"))).alias("j")
    )
    expected = spark.createDataFrame(rows, schema).select(
        SF.to_json(SF.struct(SF.lit("aaa").alias("f1"), SF.lit("bbb").alias("f2"), SF.lit("ccc").alias("f3"))).alias(
            "j"
        )
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.struct.mixed_aliased_lit_and_plain_col")
def test_struct_mixed_aliased_lit_and_plain_col_against_spark(engine, spark):
    """Struct with a mix of aliased literals and plain column references."""
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _STR))
    rows = [(1, "x"), (2, "y")]
    actual = engine.build_df(rows, schema).select(F.struct(F.col("b"), F.lit("fixed").alias("tag")).alias("s"))
    expected = spark.createDataFrame(rows, schema).select(
        SF.struct(SF.col("b"), SF.lit("fixed").alias("tag")).alias("s")
    )
    assert_matches_spark(actual, expected, engine)


# ============================================================================ #
# TestTransformStruct / TestFilterStructField / TestMapFromEntries (extra) /
# TestElementAtWithLitIndex / TestGetItemOnListInFilter
# ============================================================================ #


@pytest.mark.feature("functions.transform.struct_and_explode")
def test_transform_struct_and_explode_against_spark(engine, spark):
    """Regression: transform with a struct-producing lambda must produce explodable results matching Spark."""
    F = engine.functions
    schema = _schema(("items", SparkArrayType(_STR)))
    rows = [(["a=1", "b=2"],), (["c=3"],)]

    sf = engine.build_df(rows, schema).withColumn("items", F.transform(F.col("items"), lambda x: F.split(x, "=", 2)))
    sp = spark.createDataFrame(rows, schema).withColumn(
        "items", SF.transform(SF.col("items"), lambda x: SF.split(x, "=", 2))
    )

    sf = sf.withColumn(
        "items",
        F.transform(
            F.col("items"),
            lambda x: F.struct(F.element_at(x, 1).alias("key"), F.element_at(x, 2).alias("value")),
        ),
    )
    sp = sp.withColumn(
        "items",
        SF.transform(
            SF.col("items"),
            lambda x: SF.struct(SF.element_at(x, 1).alias("key"), SF.element_at(x, 2).alias("value")),
        ),
    )

    sf_exploded = sf.withColumn("item", F.explode("items")).select("item")
    sp_exploded = sp.withColumn("item", SF.explode_outer(SF.col("items"))).select("item")
    assert_matches_spark(sf_exploded, sp_exploded, engine)


@pytest.mark.feature("functions.transform.struct_in_when_otherwise")
def test_transform_struct_inside_when_otherwise_against_spark(engine, spark):
    """Regression: transform+struct wrapped in when/otherwise must preserve List(Struct); lit(None) must not collapse it."""
    F = engine.functions
    schema = _schema(("items", SparkArrayType(_STR)))
    rows = [(["a=1", "b=2"],), (None,), (["c=3"],)]

    def sf_struct_lambda(x):
        return F.struct(F.element_at(x, 1).alias("key"), F.element_at(x, 2).alias("value"))

    def sp_struct_lambda(x):
        return SF.struct(SF.element_at(x, 1).alias("key"), SF.element_at(x, 2).alias("value"))

    sf = engine.build_df(rows, schema).withColumn("items", F.transform(F.col("items"), lambda x: F.split(x, "=", 2)))
    sp = spark.createDataFrame(rows, schema).withColumn(
        "items", SF.transform(SF.col("items"), lambda x: SF.split(x, "=", 2))
    )

    sf = sf.withColumn(
        "items",
        F.when(F.col("items").isNotNull(), F.transform(F.col("items"), sf_struct_lambda)).otherwise(F.lit(None)),
    )
    sp = sp.withColumn(
        "items",
        SF.when(SF.col("items").isNotNull(), SF.transform(SF.col("items"), sp_struct_lambda)).otherwise(SF.lit(None)),
    )

    sf = sf.withColumn("items", F.filter(F.col("items"), lambda x: x.isNotNull()))
    sp = sp.withColumn("items", SF.filter(SF.col("items"), lambda x: x.isNotNull()))
    assert_matches_spark(sf, sp, engine)


_KV_STR_ELEM = SparkArrayType(SparkStructType([SparkStructField("key", _STR), SparkStructField("value", _STR)]))
_PAIRS_SCHEMA = _schema(("pairs", _KV_STR_ELEM))
_ENTRIES_SCHEMA = _schema(("entries", _KV_STR_ELEM))


@pytest.mark.feature("functions.filter.by_struct_field_equality")
def test_filter_by_struct_field_equality_against_spark(engine, spark):
    """Regression: filter lambda using getItem + == on a List(Struct) column must produce native expressions."""
    F = engine.functions
    rows = [
        ([("utm_source", "google"), ("utm_medium", "cpc")],),
        ([("other", "x")],),
    ]
    sf = engine.build_df(rows, _PAIRS_SCHEMA).withColumn(
        "pairs", F.filter(F.col("pairs"), lambda x: x.getItem("key") == F.lit("utm_source"))
    )
    sp = spark.createDataFrame(rows, _PAIRS_SCHEMA).withColumn(
        "pairs", SF.filter(SF.col("pairs"), lambda x: x.getItem("key") == SF.lit("utm_source"))
    )
    assert_matches_spark(sf, sp, engine)


@pytest.mark.feature("functions.transform.getitem_struct_inside")
def test_getitem_on_struct_inside_transform_against_spark(engine, spark):
    """getItem on a struct element inside transform must return typed values."""
    F = engine.functions
    rows = [([("a", "1"), ("b", "2")],)]
    sf = engine.build_df(rows, _PAIRS_SCHEMA).withColumn(
        "keys", F.transform(F.col("pairs"), lambda x: x.getItem("key"))
    )
    sp = spark.createDataFrame(rows, _PAIRS_SCHEMA).withColumn(
        "keys", SF.transform(SF.col("pairs"), lambda x: x.getItem("key"))
    )
    assert_matches_spark(sf, sp, engine)


@pytest.mark.feature("functions.map_from_entries.getitem_lookup")
def test_map_from_entries_getitem_against_spark(engine, spark):
    """map_from_entries on List(Struct(key, value)) should produce a map that getItem can look up by key."""
    F = engine.functions
    rows = [
        ([("utm_source", "google"), ("utm_medium", "cpc")],),
        ([("plan", "premium")],),
    ]
    sf = (
        engine.build_df(rows, _ENTRIES_SCHEMA)
        .withColumn("m", F.map_from_entries(F.col("entries")))
        .withColumn("src", F.col("m").getItem("utm_source"))
    )
    sp = (
        spark.createDataFrame(rows, _ENTRIES_SCHEMA)
        .withColumn("m", SF.map_from_entries(SF.col("entries")))
        .withColumn("src", SF.col("m").getItem("utm_source"))
    )
    assert_matches_spark(sf.select("src"), sp.select("src"), engine)


@pytest.mark.feature("functions.create_map.getitem_lookup")
def test_create_map_getitem_against_spark(engine, spark):
    """create_map + getItem should look up a value by key."""
    F = engine.functions
    schema = _schema(("x", _STR), ("y", _STR))
    rows = [("hello", "world")]
    sf = (
        engine.build_df(rows, schema)
        .withColumn("m", F.create_map(F.lit("x_val"), F.col("x"), F.lit("y_val"), F.col("y")))
        .withColumn("got", F.col("m").getItem("x_val"))
    )
    sp = (
        spark.createDataFrame(rows, schema)
        .withColumn("m", SF.create_map(SF.lit("x_val"), SF.col("x"), SF.lit("y_val"), SF.col("y")))
        .withColumn("got", SF.col("m").getItem("x_val"))
    )
    assert_matches_spark(sf.select("got"), sp.select("got"), engine)


@pytest.mark.feature("functions.try_element_at.lit_int_index")
def test_try_element_at_with_lit_int_against_spark(engine, spark):
    """try_element_at(array, F.lit(1)) should extract the first element."""
    F = engine.functions
    rows = [(["a", "b", "c"],)]
    sf = engine.build_df(rows, _STR_ARRAY_SCHEMA).withColumn("first", F.try_element_at(F.col("arr"), F.lit(1)))
    sp = spark.createDataFrame(rows, _STR_ARRAY_SCHEMA).withColumn(
        "first", SF.try_element_at(SF.col("arr"), SF.lit(1))
    )
    assert_matches_spark(sf.select("first"), sp.select("first"), engine)


@pytest.mark.feature("functions.element_at.lit_int_index")
def test_element_at_with_lit_int_against_spark(engine, spark):
    """element_at(array, F.lit(2)) should extract the second element."""
    F = engine.functions
    rows = [(["x", "y", "z"],)]
    sf = engine.build_df(rows, _STR_ARRAY_SCHEMA).withColumn("second", F.element_at(F.col("arr"), F.lit(2)))
    sp = spark.createDataFrame(rows, _STR_ARRAY_SCHEMA).withColumn("second", SF.element_at(SF.col("arr"), SF.lit(2)))
    assert_matches_spark(sf.select("second"), sp.select("second"), engine)


@pytest.mark.feature("functions.filter.getitem_int_in_filter")
def test_getitem_int_in_filter_against_spark(engine, spark):
    """filter(col('arr').getItem(0).isNotNull()) should keep non-null first elements."""
    F = engine.functions
    rows = [(["a", "b"],), (None,), (["c"],)]
    sf = engine.build_df(rows, _STR_ARRAY_SCHEMA).filter(F.col("arr").getItem(0).isNotNull())
    sp = spark.createDataFrame(rows, _STR_ARRAY_SCHEMA).filter(SF.col("arr").getItem(0).isNotNull())
    assert_matches_spark(sf, sp, engine)


@pytest.mark.feature("functions.filter.getitem_string_key_on_map_in_filter")
def test_getitem_string_key_on_map_in_filter_against_spark(engine, spark):
    """getItem on a map-as-struct column inside filter must resolve values."""
    F = engine.functions
    rows = [([("a", "1")],), ([("b", "2")],)]
    sf = (
        engine.build_df(rows, _ENTRIES_SCHEMA)
        .withColumn("m", F.map_from_entries(F.col("entries")))
        .withColumn("got", F.col("m").getItem("a"))
        .filter(F.col("got").isNotNull())
    )
    sp = (
        spark.createDataFrame(rows, _ENTRIES_SCHEMA)
        .withColumn("m", SF.map_from_entries(SF.col("entries")))
        .withColumn("got", SF.col("m").getItem("a"))
        .filter(SF.col("got").isNotNull())
    )
    assert_matches_spark(sf.select("got"), sp.select("got"), engine)


# ============================================================================ #
# nba surface: create_map / map_entries / first / from_json / get_json_object /
# greatest / regexp_replace / round / split / to_json / try_divide / try_to_date
# ============================================================================ #


@pytest.mark.feature("functions.create_map.empty")
@pytest.mark.parametrize("as_list", [False, True], ids=["varargs", "empty_list"])
def test_create_map_empty_against_spark(engine, spark, as_list):
    """``create_map()`` / ``create_map([])`` (datahub ``_to_map`` of an empty mapping) is an empty map."""
    F = engine.functions
    schema = _schema(("k", _STR))
    rows = [("a",), (None,)]

    def _cols(fn):
        m = fn.create_map([]) if as_list else fn.create_map()
        return [fn.size(m).alias("n"), fn.to_json(m).alias("j"), fn.map_keys(m).alias("keys")]

    actual = engine.build_df(rows, schema).select("k", *_cols(F))
    expected = spark.createDataFrame(rows, schema).select("k", *_cols(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: an odd number of arguments (a key without a value) is rejected.
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.create_map(SF.lit("a")).alias("m")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.create_map(F.lit("a")).alias("m")))


@pytest.mark.feature("functions.create_map.null_key_raises")
def test_create_map_null_key_raises_like_spark(engine, spark):
    """Spark 4 rejects a null map key at runtime (``NULL_MAP_KEY``); null values are fine."""
    F = engine.functions
    schema = _schema(("k", _STR), ("v", _STR))
    rows = [("a", "1"), ("b", None)]

    def _cols(fn):
        # ``to_json`` materializes the map: Spark's optimizer folds a bare ``create_map(...)[key]`` lookup
        # into a CASE WHEN, which would skip the null-key check.
        m = fn.create_map(fn.col("k"), fn.col("v"), fn.lit("const"), fn.lit("c"))
        return [fn.to_json(m).alias("j"), m.getItem("a").alias("a"), m.getItem("const").alias("const")]

    actual = engine.build_df(rows, schema).select(*_cols(F))
    expected = spark.createDataFrame(rows, schema).select(*_cols(SF))
    assert_matches_spark(actual, expected, engine)

    bad_rows = rows + [(None, "x")]
    with pytest.raises(Exception, match="NULL_MAP_KEY"):
        spark.createDataFrame(bad_rows, schema).select(*_cols(SF)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(bad_rows, schema).select(*_cols(F)))


@pytest.mark.feature("functions.first.ignorenulls")
def test_first_ignorenulls_against_spark(engine, spark):
    """datahub ``application``: ``groupBy(id, key).agg(first("value", ignorenulls=True))`` skips nulls."""
    F = engine.functions
    schema = _schema(("app", _STR), ("key", _STR), ("value", _STR))
    rows = [
        ("a1", "utm_source", None),
        ("a1", "utm_source", "google"),
        ("a1", "utm_source", None),
        ("a1", "gclid", None),
        ("a1", "gclid", None),
        ("a2", "utm_source", "bing"),
        ("a2", "plan", None),
    ]
    actual = engine.build_df(rows, schema).groupBy("app", "key").agg(F.first("value", ignorenulls=True).alias("v"))
    expected = (
        spark.createDataFrame(rows, schema).groupBy("app", "key").agg(SF.first("value", ignorenulls=True).alias("v"))
    )
    assert_matches_spark(actual, expected, engine)

    # Global aggregate (no groupBy) over a column whose only non-null value is in the middle.
    actual = engine.build_df(rows, schema).filter(F.col("app") == "a1").agg(F.first("value", True).alias("v"))
    expected = (
        spark.createDataFrame(rows, schema).filter(SF.col("app") == "a1").agg(SF.first("value", True).alias("v"))
    )
    assert_matches_spark(actual, expected, engine)

    # Malformed: an unresolvable column fails analysis.
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).groupBy("app").agg(SF.first("missing", ignorenulls=True)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).groupBy("app").agg(F.first("missing", ignorenulls=True)))


_FROM_JSON_STRUCT = SparkStructType(
    [SparkStructField("id", _STR), SparkStructField("count", _INT), SparkStructField("tags", SparkArrayType(_STR))]
)


@pytest.mark.feature("functions.from_json.malformed_permissive")
def test_from_json_malformed_permissive_against_spark(engine, spark):
    """Default PERMISSIVE mode: unparseable records become a struct of nulls; null input stays null."""
    F = engine.functions
    schema = _schema(("j", _STR))
    rows = [
        ('{"id": "x", "count": 3, "tags": ["a", "b"]}',),
        ('{"id": "y"}',),
        ('{"id": "z", "count": "not-an-int"}',),
        ('{"id": "w", "count": 1',),
        ("not json",),
        ("",),
        (None,),
    ]
    actual = engine.build_df(rows, schema).select("j", F.from_json("j", engine.dtype(_FROM_JSON_STRUCT)).alias("p"))
    expected = spark.createDataFrame(rows, schema).select("j", SF.from_json("j", _FROM_JSON_STRUCT).alias("p"))
    assert_matches_spark(actual, expected, engine)

    # Strict counterpart: FAILFAST raises on the same malformed record.
    bad_rows = [('{"id": "w", "count": 1',)]
    options = {"mode": "FAILFAST"}
    with pytest.raises(Exception):
        spark.createDataFrame(bad_rows, schema).select(SF.from_json("j", _FROM_JSON_STRUCT, options)).collect()
    with pytest.raises(Exception):
        engine.to_records(
            engine.build_df(bad_rows, schema).select(F.from_json("j", engine.dtype(_FROM_JSON_STRUCT), options))
        )


_INSURANCE_STRUCT = SparkStructType(
    [
        SparkStructField("enabled", SparkBooleanType()),
        SparkStructField("provider", _STR),
        SparkStructField("notSupportedLoanTypes", SparkArrayType(_STR)),
    ]
)


@pytest.mark.parametrize(
    "rows, parsed_type, lookups",
    [
        pytest.param(
            [
                ('{"name": "Visa", "limit": 5000, "secured": true, "extra": {"x": 1}, "none": null}',),
                ("{}",),
                ('{"name": "broken"',),
                ('["not", "an", "object"]',),
                (None,),
            ],
            SparkMapType(_STR, _STR),
            ["name", "limit", "secured", "extra", "none", "missing"],
            id="map_string_string",
        ),
        pytest.param(
            [
                ('{"LIFE": {"enabled": true, "provider": "TRUSTAGE", "notSupportedLoanTypes": ["HELOC"]}}',),
                ('{"GAP": {"enabled": false}}',),
                ('{"LIFE": {"enabled": "maybe"}}',),
                ("not json",),
                (None,),
            ],
            SparkMapType(_STR, _INSURANCE_STRUCT),
            ["LIFE", "GAP"],
            id="map_string_struct",
        ),
    ],
)
@pytest.mark.feature("functions.from_json.map_and_array_schemas")
def test_from_json_map_schemas_against_spark(engine, spark, rows, parsed_type, lookups):
    """Map results are compared via key lookups + ``size`` (the oracle does not compare map columns)."""
    F = engine.functions
    schema = _schema(("j", _STR))

    def _cols(fn, dtype):
        m = fn.from_json("j", dtype)
        return [fn.size(m).alias("n")] + [m.getItem(k).alias(f"k_{k}") for k in lookups]

    actual = engine.build_df(rows, schema).select("j", *_cols(F, engine.dtype(parsed_type)))
    expected = spark.createDataFrame(rows, schema).select("j", *_cols(SF, parsed_type))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.from_json.map_and_array_schemas")
def test_from_json_array_schema_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("j", _STR))
    rows = [
        ('["HELOC", "AUTO"]',),
        ("[]",),
        ('["a", null, 3]',),
        ('["unterminated"',),
        ('{"not": "an array"}',),
        (None,),
    ]
    parsed_type = SparkArrayType(_STR)
    actual = engine.build_df(rows, schema).select("j", F.from_json("j", engine.dtype(parsed_type)).alias("a"))
    expected = spark.createDataFrame(rows, schema).select("j", SF.from_json("j", parsed_type).alias("a"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.get_json_object.malformed_json_null")
@pytest.mark.parametrize("path", ["$.a", "$.a.b", "$.a[0]", "$"])
def test_get_json_object_malformed_json_returns_null(engine, spark, path):
    """Malformed JSON never raises: get_json_object returns null (valid rows still extract)."""
    F = engine.functions
    schema = _schema(("j", _STR))
    rows = [
        ('{"a": {"b": 1}}',),
        ('{"a": [7, 8]}',),
        ('{"a": ',),
        ('{"a": 1,}',),
        ("{'a': 1}",),
        ("not json",),
        ("",),
        (None,),
    ]
    actual = engine.build_df(rows, schema).select("j", F.get_json_object("j", path).alias("r"))
    expected = spark.createDataFrame(rows, schema).select("j", SF.get_json_object("j", path).alias("r"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.get_json_object.malformed_json_null")
@pytest.mark.parametrize("path", ["a", "$.", "$[", "$.a[x]", ""])
def test_get_json_object_invalid_path_returns_null(engine, spark, path):
    F = engine.functions
    schema = _schema(("j", _STR))
    rows = [('{"a": [1, 2]}',), (None,)]
    actual = engine.build_df(rows, schema).select("j", F.get_json_object("j", path).alias("r"))
    expected = spark.createDataFrame(rows, schema).select("j", SF.get_json_object("j", path).alias("r"))
    assert_matches_spark(actual, expected, engine)


_TRADELINE_JSON = json.dumps(
    {
        "BUREAU": ["EQUIFAX"],
        "TRADETYPE": [{"CODE": ["I"], "DESCRIPTION": ["Installment"]}],
        "STATUS": [{"DESCRIPTION": ["Current"]}],
        "ORIGINALAMOUNT": ["25000"],
        "PAYMENTPATTERN": [],
    }
)
_SNAPSHOT_JSON = json.dumps(
    {
        "firstName": 'Ana "Bea"',
        "annualIncome": 85000.50,
        "monthsAtEmployer": 14,
        "authenticated": True,
        "middleName": None,
        "metadata": {"probabilityOfApprovalMLModel": {"approvalChance": "HIGH", "score": 0.82}},
        "providers": {"TRUSTAGE": {"cuId": "CU-1"}},
        "insurances": {"LIFE": {"enabled": True, "notSupportedLoanTypes": ["HELOC"]}},
        "deal": {"id": 42},
        "utmVariables": "utm_source=google&utm_medium=cpc",
    }
)


@pytest.mark.feature("functions.get_json_object.nested_and_array_paths")
@pytest.mark.parametrize(
    "path",
    [
        "$.firstName",
        "$.annualIncome",
        "$.monthsAtEmployer",
        "$.authenticated",
        "$.middleName",
        "$.utmVariables",
        "$.metadata.probabilityOfApprovalMLModel",
        "$.providers.TRUSTAGE.cuId",
        "$.insurances",
        "$.deal.id",
        "$.BUREAU[0]",
        "$.TRADETYPE[0].CODE[0]",
        "$.TRADETYPE[0].DESCRIPTION[0]",
        "$.STATUS[0].DESCRIPTION[0]",
        "$.ORIGINALAMOUNT[0]",
        "$.TRADETYPE",
        "$.TRADETYPE[0]",
        "$.PAYMENTPATTERN[0]",
        "$.BUREAU[5]",
        "$.missing",
        "$.deal.missing",
        "$.firstName.nested",
    ],
)
def test_get_json_object_nested_and_array_paths_against_spark(engine, spark, path):
    """Real datahub path shapes; objects/arrays come back as compact JSON text, scalars unquoted."""
    F = engine.functions
    schema = _schema(("j", _STR))
    rows = [(_SNAPSHOT_JSON,), (_TRADELINE_JSON,), ('{"broken": ',), (None,)]
    actual = engine.build_df(rows, schema).select("j", F.get_json_object("j", path).alias("r"))
    expected = spark.createDataFrame(rows, schema).select("j", SF.get_json_object("j", path).alias("r"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.greatest.mixed_numeric_types")
@pytest.mark.parametrize(
    "a_type, b_type, rows",
    [
        (_INT, _LONG, [(1, 5), (-3, -7), (None, 2), (None, None), (2147483647, 2147483648)]),
        (_INT, _DOUBLE, [(1, 0.5), (2, 2.5), (-1, -1.5), (None, 3.25), (4, None)]),
        (_LONG, SparkFloatType(), [(1, 1.5), (10, 2.5), (None, None)]),
        (
            _INT,
            SparkDecimalType(10, 2),
            [(1, Decimal("1.25")), (3, Decimal("2.50")), (-2, Decimal("-1.75")), (None, Decimal("0.10"))],
        ),
        (
            SparkDecimalType(5, 1),
            SparkDecimalType(10, 3),
            [(Decimal("1.5"), Decimal("1.499")), (Decimal("-0.1"), Decimal("-0.125")), (None, Decimal("7.000"))],
        ),
        (_DOUBLE, SparkDecimalType(10, 2), [(0.1, Decimal("0.05")), (1e-9, Decimal("0.00")), (None, None)]),
    ],
)
def test_greatest_mixed_numeric_types_against_spark(engine, spark, a_type, b_type, rows):
    """greatest widens to a common type (value + string rendering pin the result type and scale)."""
    F = engine.functions
    schema = _schema(("a", a_type), ("b", b_type))
    string = engine.dtype(_STR)

    actual = engine.build_df(rows, schema).select(
        F.greatest("a", "b").alias("g"),
        F.greatest("a", "b").cast(string).alias("g_str"),
        F.greatest(F.col("a"), F.lit(1e-9)).alias("g_lit"),
    )
    expected = spark.createDataFrame(rows, schema).select(
        SF.greatest("a", "b").alias("g"),
        SF.greatest("a", "b").cast(_STR).alias("g_str"),
        SF.greatest(SF.col("a"), SF.lit(1e-9)).alias("g_lit"),
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.greatest.mixed_numeric_types")
def test_greatest_invalid_arguments_raise_like_spark(engine, spark):
    """greatest needs >= 2 arguments of a common type; a numeric vs array mix fails analysis."""
    F = engine.functions
    schema = _schema(("a", _INT), ("arr", SparkArrayType(_INT)))
    rows = [(1, [1, 2])]
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.greatest("a")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.greatest("a")))
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.greatest("a", "arr")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.greatest("a", "arr")))


@pytest.mark.feature("functions.map_entries")
def test_map_entries_against_spark(engine, spark):
    F = engine.functions
    rows = [({"a": 1},), ({},), ({"b": None},), (None,)]
    schema = _schema(("m", SparkMapType(_STR, _LONG)))
    actual = engine.build_df(rows, schema).select(F.map_entries("m").alias("e"))
    expected = spark.createDataFrame(rows, schema).select(SF.map_entries("m").alias("e"))
    assert_matches_spark(actual, expected, engine)

    # Ordered multi-entry maps built with create_map, then the datahub ``transform(map_entries(...))`` shape.
    schema = _schema(("k", _STR), ("v", _LONG))
    rows = [("x", 1), ("y", None)]

    def _cols(fn):
        e = fn.map_entries(fn.create_map(fn.col("k"), fn.col("v"), fn.lit("z"), fn.lit(9)))
        return [
            e.alias("e"),
            fn.transform(e, lambda entry: fn.struct(entry["key"].alias("type"), entry["value"].alias("n"))).alias("t"),
        ]

    actual = engine.build_df(rows, schema).select(*_cols(F))
    expected = spark.createDataFrame(rows, schema).select(*_cols(SF))
    assert_matches_spark(actual, expected, engine)

    # Malformed: map_entries requires a map argument.
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.map_entries("k")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.map_entries("k")))


@pytest.mark.feature("functions.regexp_replace.nba_patterns")
@pytest.mark.parametrize(
    "pattern, replacement, values",
    [
        ("[{}]", "", ["{6F9619FF-8B86-D011}", "no-braces", "}{", "", None]),
        (r"\\", "/", ["N\\A", "a\\b\\c", "n/a", "", None]),
        ("^0+", "", ["000123", "0", "1000", "a00", "", None]),
        ("[^0-9.]", "", ["6.99%", "APR 7.25 %", "abc", "1,234.50", None]),
        ("Y", "", ["5Y", "YY", "10y", None]),
        ("M", "", ["60M", "MM", "m", None]),
        (r"\s+", " ", ["a  b\t\nc", "  lead", "none", None]),
        ('"', "'", ['say "hi"', "none", None]),
        (r"^(https?://)?(www\.)?", "", ["https://www.cu.org", "http://cu.org", "www.x.com", "cu.org", None]),
        ("[\\s/-]", "_", ["Pending Review", "approved/funded", "re-opened", None]),
    ],
)
def test_regexp_replace_nba_patterns_against_spark(engine, spark, pattern, replacement, values):
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [(v,) for v in values]
    actual = engine.build_df(rows, schema).select("s", F.regexp_replace("s", pattern, replacement).alias("r"))
    expected = spark.createDataFrame(rows, schema).select("s", SF.regexp_replace("s", pattern, replacement).alias("r"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.regexp_replace.nba_patterns")
@pytest.mark.parametrize("pattern", ["[{}", "(unclosed", "*0+"])
def test_regexp_replace_invalid_pattern_raises_like_spark(engine, spark, pattern):
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [("abc",)]
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.regexp_replace("s", pattern, "")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.regexp_replace("s", pattern, "")))


@pytest.mark.feature("functions.round.half_up_double_and_decimal")
@pytest.mark.parametrize("scale", [0, 2])
def test_round_half_up_double_against_spark(engine, spark, scale):
    """Spark rounds doubles HALF_UP on the decimal rendering (2.5 -> 3, 1.005 -> 1.01), not banker's."""
    F = engine.functions
    schema = _schema(("x", _DOUBLE))
    values = [0.5, 1.5, 2.5, -0.5, -2.5, 0.125, -0.125, 1.005, 2.675, 0.0725, 123.455, -1.115, 0.0, None]
    rows = [(v,) for v in values]
    string = engine.dtype(_STR)
    actual = engine.build_df(rows, schema).select(
        "x",
        F.round("x", scale).alias("r"),
        F.round("x", scale).cast(string).alias("r_str"),
        F.round(F.col("x") * 100, scale).alias("pct"),
    )
    expected = spark.createDataFrame(rows, schema).select(
        "x",
        SF.round("x", scale).alias("r"),
        SF.round("x", scale).cast(_STR).alias("r_str"),
        SF.round(SF.col("x") * 100, scale).alias("pct"),
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.round.half_up_double_and_decimal")
@pytest.mark.parametrize("scale", [0, 2])
def test_round_half_up_decimal_against_spark(engine, spark, scale):
    """Decimal rounding is HALF_UP and the result decimal type carries the target scale."""
    F = engine.functions
    schema = _schema(("x", SparkDecimalType(10, 3)))
    values = ["0.500", "1.500", "2.500", "-2.500", "2.345", "-2.345", "2.344", "99999.995", "0.000", None]
    rows = [(Decimal(v) if v is not None else None,) for v in values]
    string = engine.dtype(_STR)
    actual = engine.build_df(rows, schema).select(
        F.round("x", scale).cast(string).alias("r_str"),
        F.round(F.col("x") * 100, scale).cast(string).alias("pct_str"),
    )
    expected = spark.createDataFrame(rows, schema).select(
        SF.round("x", scale).cast(_STR).alias("r_str"),
        SF.round(SF.col("x") * 100, scale).cast(_STR).alias("pct_str"),
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.round.half_up_double_and_decimal")
def test_round_malformed_string_raises_like_spark(engine, spark):
    """ANSI: round over a string implicitly casts to double and raises on an unparseable value."""
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [("2.5",), ("not-a-number",)]
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.round("s", 0)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.round("s", 0)))


@pytest.mark.feature("functions.split.with_limit")
@pytest.mark.parametrize(
    "pattern, limit",
    [("=", 2), ("=", 1), ("=", 0), ("=", -1), ("=", 5), ("&", -1), ("-", 2), (".", 2), (r"\.", 2)],
)
def test_split_with_limit_against_spark(engine, spark, pattern, limit):
    """datahub ``split(x, "=", 2)`` on utm pairs; limit > 0 caps the parts, limit <= 0 keeps trailing empties."""
    F = engine.functions
    schema = _schema(("s", _STR))
    values = ["utm_source=google", "a=b=c", "key=", "=value", "noeq", "", "a&b&&c&", "cu-name-1", "a.b.c", None]
    rows = [(v,) for v in values]
    actual = engine.build_df(rows, schema).select(
        "s", F.split("s", pattern, limit).alias("parts"), F.split("s", pattern, limit)[0].alias("first")
    )
    expected = spark.createDataFrame(rows, schema).select(
        "s", SF.split("s", pattern, limit).alias("parts"), SF.split("s", pattern, limit)[0].alias("first")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.split.with_limit")
def test_split_invalid_pattern_raises_like_spark(engine, spark):
    F = engine.functions
    schema = _schema(("s", _STR))
    rows = [("a=b",)]
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.split("s", "[=", 2)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.split("s", "[=", 2)))


@pytest.mark.feature("functions.to_json.options")
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"ignoreNullFields": True},
        {"ignoreNullFields": False},
        {"ignoreNullFields": "false"},
        {"ignoreNullFields": False, "timestampFormat": "yyyy/MM/dd HH:mm", "dateFormat": "MM-dd-yyyy"},
        {"timestampFormat": "dd.MM.yyyy'T'HH:mm:ss", "dateFormat": "yyyyMMdd"},
    ],
    ids=["default", "ignore_true", "ignore_false", "ignore_false_str", "ignore_false_formats", "formats_only"],
)
def test_to_json_options_against_spark(engine, spark, options):
    """``ignoreNullFields`` drops/keeps null struct fields at every nesting level; arrays keep their nulls."""
    F = engine.functions
    schema = _schema(
        ("id", _LONG),
        ("name", _STR),
        ("t", SparkTimestampType()),
        ("d", SparkDateType()),
        ("tags", SparkArrayType(_STR)),
    )
    rows = [
        (1, "a", datetime(2026, 10, 1, 15, 32, 10), date(2026, 10, 1), ["x", None]),
        (2, None, None, None, None),
    ]

    def _payload(fn):
        inner = fn.struct(fn.col("name").alias("name"), fn.col("d").alias("d"))
        return fn.struct("id", "name", "t", "d", "tags", inner.alias("inner"))

    actual = engine.build_df(rows, schema).select(F.to_json(_payload(F), options).alias("j"))
    expected = spark.createDataFrame(rows, schema).select(SF.to_json(_payload(SF), options).alias("j"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("functions.to_json.options")
def test_to_json_invalid_timestamp_format_raises_like_spark(engine, spark):
    F = engine.functions
    schema = _schema(("t", SparkTimestampType()))
    rows = [(datetime(2026, 10, 1, 15, 32, 10),)]
    options = {"timestampFormat": "yyyy-MM-dd bbb"}
    with pytest.raises(Exception):
        spark.createDataFrame(rows, schema).select(SF.to_json(SF.struct("t"), options)).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).select(F.to_json(F.struct("t"), options)))


@pytest.mark.feature("functions.try_divide.by_zero_null")
@pytest.mark.parametrize(
    "dtype, rows",
    [
        (_INT, [(10, 4), (7, 0), (0, 0), (-3, 0), (None, 0), (5, None)]),
        (_LONG, [(9, 3), (1, 0), (0, 0)]),
        (SparkDecimalType(10, 2), [(Decimal("10.00"), Decimal("4.00")), (Decimal("1.50"), Decimal("0.00"))]),
        (_DOUBLE, [(1.0, 3.0), (0.0, 0.0), (-2.5, 0.0)]),
    ],
)
def test_try_divide_by_zero_returns_null_against_spark(engine, spark, dtype, rows):
    """try_divide returns null on a zero divisor for every numeric type (incl. the datahub ``a / (a + b)``)."""
    F = engine.functions
    schema = _schema(("a", dtype), ("b", dtype))
    string = engine.dtype(_STR)
    actual = engine.build_df(rows, schema).select(
        F.try_divide("a", "b").alias("q"),
        F.try_divide("a", "b").cast(string).alias("q_str"),
        F.try_divide(F.col("a"), F.col("a") + F.col("b")).alias("ratio"),
    )
    expected = spark.createDataFrame(rows, schema).select(
        SF.try_divide("a", "b").alias("q"),
        SF.try_divide("a", "b").cast(_STR).alias("q_str"),
        SF.try_divide(SF.col("a"), SF.col("a") + SF.col("b")).alias("ratio"),
    )
    assert_matches_spark(actual, expected, engine)

    # Strict counterpart: ANSI ``/`` raises DIVIDE_BY_ZERO on the same zero divisor.
    zero_rows = [r for r in rows if r[0] is not None and r[1] is not None and r[1] == 0]
    with pytest.raises(Exception, match="DIVIDE_BY_ZERO"):
        spark.createDataFrame(zero_rows, schema).select((SF.col("a") / SF.col("b")).alias("q")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(zero_rows, schema).select((F.col("a") / F.col("b")).alias("q")))


@pytest.mark.feature("functions.try_to_date.nba_patterns")
@pytest.mark.parametrize(
    "fmt, values",
    [
        (
            "yyyy-MM-dd",
            ["2024-02-29", "1990-01-05", "2023-02-29", "2024-13-01", "2024-1-5", "05/01/2024", "2024-01-01 10:00:00"],
        ),
        ("M/d/yyyy", ["1/5/2024", "12/31/1999", "01/05/2024", "2/30/2024", "13/1/2024", "1/5/24", "2024-01-05"]),
        ("M/d/yy", ["1/5/24", "12/31/99", "01/05/70", "2/30/24", "13/1/24", "1/5/2024", "2024-01-05"]),
        ("dd/MM/yy", ["31/12/24", "05/01/99", "5/1/24", "30/02/24", "12/31/24", "31/12/2024", "2024-12-31"]),
        ("d/M/yy", ["5/1/24", "31/12/99", "05/01/24", "30/2/24", "1/13/24", "5/1/2024", "2024-01-05"]),
        ("MM/yyyy", ["01/2024", "12/1999", "1/2024", "13/2024", "00/2024", "01/24", "2024-01"]),
        ("MM/yy", ["01/24", "12/99", "1/24", "13/24", "01/2024", "2024-01"]),
        (
            "MM/dd/yyyy",
            ["02/29/2024", "12/31/1999", "2/9/2024", "02/30/2024", "13/01/2024", "31/12/2024", "02/29/24"],
        ),
    ],
)
def test_try_to_date_nba_patterns_against_spark(engine, spark, fmt, values):
    """datahub ``utils.try_to_date`` formats: valid, wrong-pattern and out-of-range values (the latter -> null)."""
    F = engine.functions
    schema = _schema(("d", _STR))
    rows = [(v,) for v in values + ["", "not-a-date", None]]
    actual = engine.build_df(rows, schema).select("d", F.try_to_date("d", fmt).alias("r"))
    expected = spark.createDataFrame(rows, schema).select("d", SF.try_to_date("d", fmt).alias("r"))
    assert_matches_spark(actual, expected, engine)
