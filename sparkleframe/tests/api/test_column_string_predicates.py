"""Batch 4 parity tests: Column string predicates.

Covers:
    - ``contains(substring)`` — literal substring, case-sensitive, null-propagating.
    - ``startswith(prefix)`` / ``endswith(suffix)`` — null-propagating, empty arg.
    - ``like(pattern)`` — SQL LIKE; ``%`` multi-char, ``_`` single-char,
      ``\\`` escape; anchored to full string.
    - ``rlike(pattern)`` — regex search semantics (Spark uses Java regex).

Every test runs against both backends (``polarsdf``, ``pythondf``) and asserts
parity against native PySpark via ``assert_sparkle_spark_frame_are_equal``.
"""
from __future__ import annotations

import pyspark.sql.functions as SF
from pyspark.sql.types import (
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
)

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


def _string_schema(*names):
    return SparkStructType(
        [SparkStructField(n, SparkStringType(), nullable=True) for n in names]
    )


def _run(session, F, spark, rows_tuples, expr_builder):
    """Build a 1-column ``x`` DataFrame on both backends, project ``r`` from ``expr_builder``.

    ``expr_builder(F_module)`` returns the projected column expression.
    """
    rows = [{"x": t[0]} for t in rows_tuples]
    schema = _string_schema("x")
    spark_df = (
        spark.createDataFrame(rows_tuples, schema=schema)
        .withColumn("r", expr_builder(SF))
        .select("r")
    )
    df = (
        session.createDataFrame(rows)
        .withColumn("r", expr_builder(F))
        .select("r")
    )
    assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# contains
# -----------------------------------------------------------------------------


class TestContains:
    def test_contains_match_and_no_match(self, session, F, spark):
        rows = [("apple",), ("banana",), ("apricot",), ("cherry",)]
        _run(session, F, spark, rows, lambda M: M.col("x").contains("ap"))

    def test_contains_with_null(self, session, F, spark):
        rows = [("apple",), (None,), ("apricot",), ("banana",)]
        _run(session, F, spark, rows, lambda M: M.col("x").contains("ap"))

    def test_contains_case_sensitive(self, session, F, spark):
        rows = [("Spark",), ("spark",), ("SPARK",), ("sParK",)]
        _run(session, F, spark, rows, lambda M: M.col("x").contains("spark"))

    def test_contains_literal_dot(self, session, F, spark):
        # '.' must be treated as a literal character, not a regex wildcard.
        rows = [("a.b",), ("ab",), ("a*b",), ("axb",)]
        _run(session, F, spark, rows, lambda M: M.col("x").contains("."))

    def test_contains_literal_star(self, session, F, spark):
        # '*' must be treated as a literal character.
        rows = [("a*b",), ("ab",), ("aab",), ("a.b",)]
        _run(session, F, spark, rows, lambda M: M.col("x").contains("*"))

    def test_contains_empty_substring(self, session, F, spark):
        # Empty substring matches every (non-null) value; null propagates.
        rows = [("apple",), ("",), (None,)]
        _run(session, F, spark, rows, lambda M: M.col("x").contains(""))


# -----------------------------------------------------------------------------
# startswith
# -----------------------------------------------------------------------------


class TestStartswith:
    def test_startswith_match(self, session, F, spark):
        rows = [("apple",), ("apricot",), ("banana",), ("avocado",)]
        _run(session, F, spark, rows, lambda M: M.col("x").startswith("ap"))

    def test_startswith_no_match(self, session, F, spark):
        rows = [("zebra",), ("yellow",), ("xylophone",)]
        _run(session, F, spark, rows, lambda M: M.col("x").startswith("ap"))

    def test_startswith_with_null(self, session, F, spark):
        rows = [("apple",), (None,), ("apricot",)]
        _run(session, F, spark, rows, lambda M: M.col("x").startswith("ap"))

    def test_startswith_empty_prefix(self, session, F, spark):
        # Every non-null value starts with the empty string.
        rows = [("apple",), ("",), (None,)]
        _run(session, F, spark, rows, lambda M: M.col("x").startswith(""))

    def test_startswith_literal_special_chars(self, session, F, spark):
        rows = [(".dot",), ("dot.",), ("*star",), ("star*",)]
        _run(session, F, spark, rows, lambda M: M.col("x").startswith("."))


# -----------------------------------------------------------------------------
# endswith
# -----------------------------------------------------------------------------


class TestEndswith:
    def test_endswith_match(self, session, F, spark):
        rows = [("apple",), ("staple",), ("banana",), ("ample",)]
        _run(session, F, spark, rows, lambda M: M.col("x").endswith("ple"))

    def test_endswith_no_match(self, session, F, spark):
        rows = [("zebra",), ("yellow",), ("xylophone",)]
        _run(session, F, spark, rows, lambda M: M.col("x").endswith("ple"))

    def test_endswith_with_null(self, session, F, spark):
        rows = [("apple",), (None,), ("staple",)]
        _run(session, F, spark, rows, lambda M: M.col("x").endswith("ple"))

    def test_endswith_empty_suffix(self, session, F, spark):
        # Every non-null value ends with the empty string.
        rows = [("apple",), ("",), (None,)]
        _run(session, F, spark, rows, lambda M: M.col("x").endswith(""))

    def test_endswith_literal_special_chars(self, session, F, spark):
        rows = [("file.",), (".file",), ("a*",), ("*a",)]
        _run(session, F, spark, rows, lambda M: M.col("x").endswith("."))


# -----------------------------------------------------------------------------
# like
# -----------------------------------------------------------------------------


class TestLike:
    def test_like_percent_at_end(self, session, F, spark):
        rows = [("apple",), ("apricot",), ("banana",), ("avocado",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("ap%"))

    def test_like_percent_at_start(self, session, F, spark):
        rows = [("apple",), ("staple",), ("banana",), ("ple",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("%ple"))

    def test_like_percent_in_middle(self, session, F, spark):
        rows = [("apricot",), ("aXot",), ("aot",), ("aXXXot",), ("apple",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("a%ot"))

    def test_like_underscore_single_char(self, session, F, spark):
        # 'c_t' matches 'cat'/'cut' but not 'caat' or 'ct'.
        rows = [("cat",), ("cut",), ("caat",), ("ct",), ("cot",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("c_t"))

    def test_like_underscore_and_percent(self, session, F, spark):
        rows = [("apple",), ("apricot",), ("a",), ("ab",), ("axXz",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("a_%"))

    def test_like_full_string_anchored(self, session, F, spark):
        # 'ap' without wildcards matches *only* the exact string 'ap'.
        rows = [("ap",), ("apple",), ("apx",), ("xap",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("ap"))

    def test_like_with_null(self, session, F, spark):
        rows = [("apple",), (None,), ("apricot",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("ap%"))

    def test_like_literal_special_regex_chars(self, session, F, spark):
        # '.' is regex-wildcard in regex but literal in LIKE.
        rows = [("a.b",), ("axb",), ("a..b",), ("ab",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("a.b"))

    def test_like_escape_percent(self, session, F, spark):
        # Spark's default escape char is backslash: '\%' matches literal '%'.
        rows = [("100%",), ("100",), ("1000",), ("%",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("100\\%"))

    def test_like_escape_underscore(self, session, F, spark):
        rows = [("a_b",), ("axb",), ("ab",), ("a__b",)]
        _run(session, F, spark, rows, lambda M: M.col("x").like("a\\_b"))


# -----------------------------------------------------------------------------
# rlike
# -----------------------------------------------------------------------------


class TestRlike:
    def test_rlike_starts_with(self, session, F, spark):
        rows = [("apple",), ("banana",), ("apricot",), ("avocado",)]
        _run(session, F, spark, rows, lambda M: M.col("x").rlike("^a"))

    def test_rlike_ends_with(self, session, F, spark):
        rows = [("apple",), ("banana",), ("apricot",), ("avocado",)]
        _run(session, F, spark, rows, lambda M: M.col("x").rlike("a$"))

    def test_rlike_character_class(self, session, F, spark):
        rows = [("abc",), ("ABC",), ("123",), ("a1b",), ("",)]
        _run(session, F, spark, rows, lambda M: M.col("x").rlike("[0-9]"))

    def test_rlike_quantifier(self, session, F, spark):
        rows = [("a",), ("aa",), ("aaa",), ("b",), ("ab",)]
        _run(session, F, spark, rows, lambda M: M.col("x").rlike("a{2,}"))

    def test_rlike_digits_only_anchored(self, session, F, spark):
        rows = [("123",), ("abc",), ("12a",), ("",), ("9",)]
        _run(session, F, spark, rows, lambda M: M.col("x").rlike(r"^\d+$"))

    def test_rlike_substring_match(self, session, F, spark):
        # rlike is a *search*, not anchored — should match anywhere.
        rows = [("xyzapple",), ("apple",), ("zebra",), ("appletree",)]
        _run(session, F, spark, rows, lambda M: M.col("x").rlike("app"))

    def test_rlike_with_null(self, session, F, spark):
        rows = [("apple",), (None,), ("banana",)]
        _run(session, F, spark, rows, lambda M: M.col("x").rlike("^a"))
