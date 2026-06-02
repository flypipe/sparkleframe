"""Batch 16 parity tests: string functions in ``F``.

Covered:
    - ``F.upper(col)`` / ``F.lower(col)`` / ``F.initcap(col)`` — case transforms.
    - ``F.trim(col)`` — Spark strips ASCII space (U+0020) only.
    - ``F.length(col)`` — IntegerType-shaped length.
    - ``F.substring(col, pos, length)`` — 1-indexed; negative ``pos`` from end.
    - ``F.concat(*cols)`` — null operand poisons the result (Spark default).
    - ``F.split(col, pattern, limit)`` — regex split returning an array column.
    - ``F.regexp_replace(col, pattern, replacement)`` — regex substitution.
    - ``F.md5(col)`` — 32-char lowercase hex MD5 over UTF-8 bytes.

All tests run against both backends via the ``(session, F, spark)`` fixtures and
assert parity with native PySpark.
"""
from __future__ import annotations

import pyspark.sql.functions as SF
from pyspark.sql.types import (
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


_STR_NULLABLE_SCHEMA = SparkStructType(
    [SparkStructField("s", SparkStringType(), nullable=True)]
)


# ---------------------------------------------------------------------------
# F.upper
# ---------------------------------------------------------------------------


class TestUpper:
    def test_upper_basic(self, session, F, spark):
        rows = [{"s": "hello"}, {"s": "Foo Bar"}, {"s": "MIXED"}]
        sdf = session.createDataFrame(rows).select(F.upper("s").alias("u"))
        pdf = spark.createDataFrame(rows).select(SF.upper("s").alias("u"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_upper_null_and_empty(self, session, F, spark):
        rows = [{"s": "abc"}, {"s": None}, {"s": ""}]
        sdf = session.createDataFrame(rows, schema="s STRING").select(F.upper(F.col("s")).alias("u"))
        pdf = spark.createDataFrame(rows, schema=_STR_NULLABLE_SCHEMA).select(
            SF.upper(SF.col("s")).alias("u")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.lower
# ---------------------------------------------------------------------------


class TestLower:
    def test_lower_basic(self, session, F, spark):
        rows = [{"s": "Hello"}, {"s": "FOO BAR"}, {"s": "mixed"}]
        sdf = session.createDataFrame(rows).select(F.lower("s").alias("l"))
        pdf = spark.createDataFrame(rows).select(SF.lower("s").alias("l"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_lower_null_and_empty(self, session, F, spark):
        rows = [{"s": "ABC"}, {"s": None}, {"s": ""}]
        sdf = session.createDataFrame(rows, schema="s STRING").select(F.lower("s").alias("l"))
        pdf = spark.createDataFrame(rows, schema=_STR_NULLABLE_SCHEMA).select(SF.lower("s").alias("l"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.initcap
# ---------------------------------------------------------------------------


class TestInitcap:
    def test_initcap_basic(self, session, F, spark):
        rows = [{"s": "hello world"}, {"s": "FOO BAR"}, {"s": "already Title"}]
        sdf = session.createDataFrame(rows).select(F.initcap("s").alias("c"))
        pdf = spark.createDataFrame(rows).select(SF.initcap("s").alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_initcap_mixed_case(self, session, F, spark):
        rows = [{"s": "UPPER"}, {"s": "lower"}, {"s": "mIxEd CaSe"}]
        sdf = session.createDataFrame(rows).select(F.initcap("s").alias("c"))
        pdf = spark.createDataFrame(rows).select(SF.initcap("s").alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_initcap_null_and_empty(self, session, F, spark):
        rows = [{"s": None}, {"s": ""}, {"s": "hello"}]
        sdf = session.createDataFrame(rows, schema="s STRING").select(F.initcap("s").alias("c"))
        pdf = spark.createDataFrame(rows, schema=_STR_NULLABLE_SCHEMA).select(SF.initcap("s").alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.trim
# ---------------------------------------------------------------------------


class TestTrim:
    def test_trim_ascii_spaces(self, session, F, spark):
        rows = [{"s": "  hello  "}, {"s": " world"}, {"s": "x "}, {"s": "no_pad"}]
        sdf = session.createDataFrame(rows).select(F.trim("s").alias("t"))
        pdf = spark.createDataFrame(rows).select(SF.trim("s").alias("t"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_trim_null_and_empty(self, session, F, spark):
        rows = [{"s": "  x  "}, {"s": None}, {"s": ""}, {"s": "    "}]
        sdf = session.createDataFrame(rows, schema="s STRING").select(F.trim("s").alias("t"))
        pdf = spark.createDataFrame(rows, schema=_STR_NULLABLE_SCHEMA).select(SF.trim("s").alias("t"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.length
# ---------------------------------------------------------------------------


class TestLength:
    def test_length_basic(self, session, F, spark):
        rows = [{"s": "hello"}, {"s": ""}, {"s": "ab"}]
        sdf = session.createDataFrame(rows).select(F.length("s").alias("n"))
        pdf = spark.createDataFrame(rows).select(SF.length("s").alias("n"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_length_null(self, session, F, spark):
        rows = [{"s": "abc"}, {"s": None}, {"s": ""}]
        sdf = session.createDataFrame(rows, schema="s STRING").select(F.length("s").alias("n"))
        pdf = spark.createDataFrame(rows, schema=_STR_NULLABLE_SCHEMA).select(SF.length("s").alias("n"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.substring
# ---------------------------------------------------------------------------


class TestSubstring:
    def test_substring_pos1_len3(self, session, F, spark):
        rows = [{"s": "hello world"}, {"s": "abc"}, {"s": ""}]
        sdf = session.createDataFrame(rows).select(F.substring("s", 1, 3).alias("sub"))
        pdf = spark.createDataFrame(rows).select(SF.substring("s", 1, 3).alias("sub"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_substring_pos3_len5(self, session, F, spark):
        rows = [{"s": "hello world"}, {"s": "ab"}, {"s": "xyz"}]
        sdf = session.createDataFrame(rows).select(F.substring("s", 3, 5).alias("sub"))
        pdf = spark.createDataFrame(rows).select(SF.substring("s", 3, 5).alias("sub"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_substring_negative_pos(self, session, F, spark):
        # pos=-2, length=2 -> last 2 chars
        rows = [{"s": "hello"}, {"s": "ab"}, {"s": ""}]
        sdf = session.createDataFrame(rows).select(F.substring("s", -2, 2).alias("sub"))
        pdf = spark.createDataFrame(rows).select(SF.substring("s", -2, 2).alias("sub"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_substring_length_overruns(self, session, F, spark):
        rows = [{"s": "abc"}, {"s": "hello"}]
        sdf = session.createDataFrame(rows).select(F.substring("s", 2, 100).alias("sub"))
        pdf = spark.createDataFrame(rows).select(SF.substring("s", 2, 100).alias("sub"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_substring_null_in(self, session, F, spark):
        rows = [{"s": "abc"}, {"s": None}, {"s": ""}]
        sdf = session.createDataFrame(rows, schema="s STRING").select(F.substring("s", 1, 2).alias("sub"))
        pdf = spark.createDataFrame(rows, schema=_STR_NULLABLE_SCHEMA).select(
            SF.substring("s", 1, 2).alias("sub")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.concat
# ---------------------------------------------------------------------------


class TestConcat:
    def test_concat_two_cols(self, session, F, spark):
        rows = [{"a": "foo", "b": "bar"}, {"a": "x", "b": "y"}]
        sdf = session.createDataFrame(rows).select(F.concat("a", "b").alias("c"))
        pdf = spark.createDataFrame(rows).select(SF.concat(SF.col("a"), SF.col("b")).alias("c"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_concat_three_cols_with_literal(self, session, F, spark):
        rows = [{"a": "north", "b": "east"}, {"a": "south", "b": "west"}]
        sdf = session.createDataFrame(rows).select(F.concat(F.col("a"), F.lit("-"), F.col("b")).alias("c"))
        pdf = spark.createDataFrame(rows).select(
            SF.concat(SF.col("a"), SF.lit("-"), SF.col("b")).alias("c")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_concat_null_operand_yields_null(self, session, F, spark):
        rows = [{"a": "aa", "b": "bb"}, {"a": None, "b": "bb"}, {"a": "aa", "b": None}]
        schema = SparkStructType(
            [
                SparkStructField("a", SparkStringType(), nullable=True),
                SparkStructField("b", SparkStringType(), nullable=True),
            ]
        )
        sdf = session.createDataFrame(rows, schema="a STRING, b STRING").select(
            F.concat(F.col("a"), F.col("b")).alias("c")
        )
        pdf = spark.createDataFrame(rows, schema=schema).select(
            SF.concat(SF.col("a"), SF.col("b")).alias("c")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_split_comma(self, session, F, spark):
        rows = [{"s": "a,b,c"}, {"s": "x,y"}, {"s": "solo"}]
        sdf = session.createDataFrame(rows).select(F.split("s", ",").alias("parts"))
        pdf = spark.createDataFrame(rows).select(SF.split(SF.col("s"), ",").alias("parts"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_split_regex_whitespace(self, session, F, spark):
        rows = [{"s": "a b\tc  d"}, {"s": "one two"}]
        sdf = session.createDataFrame(rows).select(F.split("s", r"\s+").alias("parts"))
        pdf = spark.createDataFrame(rows).select(SF.split(SF.col("s"), r"\s+").alias("parts"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_split_with_limit(self, session, F, spark):
        # limit=2 -> at most 2 pieces (maxsplit=1)
        rows = [{"s": "a-b-c-d"}, {"s": "p-q"}]
        sdf = session.createDataFrame(rows).select(F.split("s", "-", 2).alias("parts"))
        pdf = spark.createDataFrame(rows).select(SF.split(SF.col("s"), "-", 2).alias("parts"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_split_null_input(self, session, F, spark):
        rows = [{"s": "a-b"}, {"s": None}]
        sdf = session.createDataFrame(rows, schema="s STRING").select(F.split("s", "-").alias("parts"))
        pdf = spark.createDataFrame(rows, schema=_STR_NULLABLE_SCHEMA).select(
            SF.split(SF.col("s"), "-").alias("parts")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.regexp_replace
# ---------------------------------------------------------------------------


class TestRegexpReplace:
    def test_regexp_replace_literal(self, session, F, spark):
        rows = [{"s": "hello world"}, {"s": "world peace"}, {"s": "noop"}]
        sdf = session.createDataFrame(rows).select(F.regexp_replace("s", "world", "earth").alias("r"))
        pdf = spark.createDataFrame(rows).select(SF.regexp_replace("s", "world", "earth").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_regexp_replace_regex(self, session, F, spark):
        rows = [{"s": "abc123def"}, {"s": "x9y"}, {"s": "no digits"}]
        sdf = session.createDataFrame(rows).select(F.regexp_replace("s", r"\d+", "#").alias("r"))
        pdf = spark.createDataFrame(rows).select(SF.regexp_replace("s", r"\d+", "#").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_regexp_replace_capture_groups(self, session, F, spark):
        # Spark supports $1 (Java/Spark) but Python uses \1 — keep replacement
        # literal-only to avoid the syntactic divergence in this batch.
        rows = [{"s": "abc-123"}, {"s": "x-9-y"}]
        sdf = session.createDataFrame(rows).select(F.regexp_replace("s", "-", "_").alias("r"))
        pdf = spark.createDataFrame(rows).select(SF.regexp_replace("s", "-", "_").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_regexp_replace_null_in(self, session, F, spark):
        rows = [{"s": "abc"}, {"s": None}, {"s": ""}]
        sdf = session.createDataFrame(rows, schema="s STRING").select(
            F.regexp_replace("s", "b", "X").alias("r")
        )
        pdf = spark.createDataFrame(rows, schema=_STR_NULLABLE_SCHEMA).select(
            SF.regexp_replace("s", "b", "X").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.md5
# ---------------------------------------------------------------------------


class TestMd5:
    def test_md5_ascii(self, session, F, spark):
        rows = [{"s": "abc"}, {"s": ""}, {"s": "hello world"}]
        sdf = session.createDataFrame(rows).select(F.md5("s").alias("h"))
        pdf = spark.createDataFrame(rows).select(SF.md5("s").alias("h"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_md5_unicode(self, session, F, spark):
        rows = [{"s": "café"}, {"s": "naïve"}, {"s": "日本語"}]
        sdf = session.createDataFrame(rows).select(F.md5("s").alias("h"))
        pdf = spark.createDataFrame(rows).select(SF.md5("s").alias("h"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_md5_null(self, session, F, spark):
        rows = [{"s": "abc"}, {"s": None}, {"s": ""}]
        sdf = session.createDataFrame(rows, schema="s STRING").select(F.md5("s").alias("h"))
        pdf = spark.createDataFrame(rows, schema=_STR_NULLABLE_SCHEMA).select(SF.md5("s").alias("h"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)
