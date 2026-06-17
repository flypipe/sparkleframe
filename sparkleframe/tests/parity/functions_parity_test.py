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

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import BinaryType as SparkBinaryType
from pyspark.sql.types import DoubleType as SparkDoubleType
from pyspark.sql.types import IntegerType as SparkIntegerType
from pyspark.sql.types import LongType as SparkLongType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import StringType as SparkStringType
from pyspark.sql.types import StructField as SparkStructField
from pyspark.sql.types import StructType as SparkStructType

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
    expected = spark.createDataFrame(rows, schema).select(SF.monotonically_increasing_id().alias("id"))
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
    rows = [([1, 2],), (None,), ([],)]
    actual = engine.build_df(rows, _ARR_LONG_SCHEMA).select(F.explode("arr").alias("e"))
    expected = spark.createDataFrame(rows, _ARR_LONG_SCHEMA).select(SF.explode_outer("arr").alias("e"))
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
@pytest.mark.xfail(
    reason="Parity gap: Spark's months_between rounds to 8 decimal places (roundOff=True); "
    "SparkleFrame returns full precision (e.g. 2.16129032258 vs Spark's 2.16129032).",
    strict=False,
)
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
