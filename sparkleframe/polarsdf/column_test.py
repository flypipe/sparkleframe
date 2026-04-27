from datetime import date, datetime

import polars as pl
import pyspark.sql.functions as F
import pytest

from sparkleframe.polarsdf import DataFrame, StringType
from sparkleframe.polarsdf.functions import col, current_date, date_sub, lit
from sparkleframe.polarsdf.types import (
    SPARK_TYPE_NAME_MAP,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    TimestampType,
)


@pytest.fixture
def sample_df():
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": ["cat", "dog", "bird"]})


class TestColumn:
    def evaluate_expr(self, expr, df: pl.DataFrame) -> pl.Series:
        return df.select(expr.to_native()).to_series()

    @pytest.mark.parametrize(
        "op_name,expr_func",
        [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
            ("/", lambda a, b: a / b),
        ],
    )
    def test_arithmetics(self, sample_df, op_name, expr_func):
        result = self.evaluate_expr(expr_func(col("a"), col("b")), sample_df)

        expected = sample_df.select((expr_func(pl.col("a"), pl.col("b"))).alias("result")).to_series()
        assert result.to_list() == expected.to_list()

    @pytest.mark.parametrize(
        "expr_func, expected_func",
        [
            (lambda a: 10 + col("a"), lambda df: 10 + pl.col("a")),
            (lambda a: 10 - col("a"), lambda df: 10 - pl.col("a")),
            (lambda a: 10 * col("a"), lambda df: 10 * pl.col("a")),
            (lambda a: 10 / col("a"), lambda df: 10 / pl.col("a")),
        ],
    )
    def test_reverse_arithmetics(self, sample_df, expr_func, expected_func):
        expr = expr_func(col("a"))
        result = self.evaluate_expr(expr, sample_df)

        expected = sample_df.select(expected_func(sample_df).alias("result")).to_series()
        assert result.to_list() == expected.to_list()

    @pytest.mark.parametrize(
        "op_name, expr_func",
        [
            ("==", lambda a, b: a == b),
            ("!=", lambda a, b: a != b),
            ("<", lambda a, b: a < b),
            ("<=", lambda a, b: a <= b),
            (">", lambda a, b: a > b),
            (">=", lambda a, b: a >= b),
        ],
    )
    def test_comparisons(self, sample_df, op_name, expr_func):
        result = self.evaluate_expr(expr_func(col("a"), col("b")), sample_df)

        expected = sample_df.select((expr_func(pl.col("a"), pl.col("b"))).alias("result")).to_series()
        assert result.to_list() == expected.to_list()

    def test_chained_expression(self, sample_df):
        expr = (col("c") - col("a") + col("b")) * col("b") / col("a")
        result = self.evaluate_expr(expr, sample_df)

        expected = sample_df.select(
            ((pl.col("c") - pl.col("a") + pl.col("b")) * pl.col("b") / pl.col("a")).alias("result")
        ).to_series()
        assert result.to_list() == expected.to_list()

    def test_alias(self, sample_df):
        expr = (col("a") + col("b")).alias("sum_ab")
        result = sample_df.select(expr.to_native()).to_series()

        expected = sample_df.select((pl.col("a") + pl.col("b")).alias("sum_ab")).to_series()
        assert result.to_list() == expected.to_list()

    @pytest.mark.parametrize(
        "column_name, values, use_variadic",
        [
            ("a", [1, 2], False),
            ("a", [1, 2], True),
            ("a", [10], False),
            ("a", [10], True),
            ("d", ["cat", "dog"], False),
            ("d", ["cat", "dog"], True),
            ("d", ["fish"], False),
            ("d", ["fish"], True),
            ("d", [], False),
            ("d", [], True),
        ],
    )
    def test_isin(self, sample_df, column_name, values, use_variadic):
        # Build expression using either list or variadic form
        if use_variadic:
            expr = col(column_name).isin(*values)
        else:
            expr = col(column_name).isin(values)

        result = sample_df.select(expr.to_native().alias("result")).to_series()
        expected = sample_df.select(pl.col(column_name).is_in(values).alias("result")).to_series()

        assert result.to_list() == expected.to_list()

    def test_otherwise(self, sample_df):
        from sparkleframe.polarsdf.functions import when

        expr = when(col("a") > 2, "yes").otherwise("no")
        result = self.evaluate_expr(expr, sample_df)

        expected = sample_df.select(
            pl.when(pl.col("a") > 2).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("result")
        ).to_series()

        assert result.to_list() == expected.to_list()

    @pytest.mark.parametrize(
        "data_type_class, expected_polars_dtype",
        [
            (StringType, pl.Utf8),
            (IntegerType, pl.Int32),
            (LongType, pl.Int64),
            (FloatType, pl.Float32),
            (DoubleType, pl.Float64),
            (BooleanType, pl.Boolean),
            (DateType, pl.Date),
            (TimestampType, pl.Datetime),
            (ByteType, pl.Int8),
            (ShortType, pl.Int16),
            (BinaryType, pl.Binary),
        ],
    )
    def test_cast_types(self, sample_df, data_type_class, expected_polars_dtype):
        # Cast column 'a' to the specified type
        expr = col("a").cast(data_type_class())
        result_df = DataFrame(sample_df).select(expr.alias("casted"))

        # Assert the column's dtype is as expected
        assert result_df.to_native_df().schema["casted"] == expected_polars_dtype

    # ---- try_cast tests ----

    @pytest.mark.parametrize(
        "data_type_class",
        [
            StringType,
            IntegerType,
            LongType,
            FloatType,
            DoubleType,
            BooleanType,
        ],
    )
    def test_try_cast_datatype_valid(self, sample_df, data_type_class):
        expected_polars_dtype = data_type_class().to_native()
        expr = col("a").try_cast(data_type_class())
        result_df = DataFrame(sample_df).select(expr.alias("casted"))
        assert result_df.to_native_df().schema["casted"] == expected_polars_dtype

    @pytest.mark.parametrize(
        "type_name",
        [
            "string",
            "int",
            "integer",
            "bigint",
            "long",
            "double",
            "float",
            "boolean",
            "date",
            "timestamp",
        ],
    )
    def test_try_cast_string_type_name(self, sample_df, type_name: str):
        expected_polars_dtype = SPARK_TYPE_NAME_MAP[type_name]
        expr = col("a").try_cast(type_name)
        result_df = DataFrame(sample_df).select(expr.alias("casted"))
        assert result_df.to_native_df().schema["casted"] == expected_polars_dtype

    def test_try_cast_invalid_returns_null(self):
        df = pl.DataFrame({"name": ["123", "Bob", None]})
        result = DataFrame(df).select(col("name").try_cast(LongType()).alias("v")).to_native_df()
        assert result["v"].to_list() == [123, None, None]

    def test_try_cast_string_invalid_returns_null(self):
        df = pl.DataFrame({"name": ["123", "Bob", None]})
        result = DataFrame(df).select(col("name").try_cast("double").alias("v")).to_native_df()
        assert result["v"].to_list() == [123.0, None, None]

    def test_is_not_null(self):
        df = pl.DataFrame({"x": [1, None, 3, None, 5]})

        expr = col("x").isNotNull()
        result = df.select(expr.to_native().alias("result")).to_series()

        expected = df.select(pl.col("x").is_not_null().alias("result")).to_series()

        assert result.to_list() == expected.to_list()

    @pytest.mark.parametrize(
        "expr_func, description",
        [
            (lambda a, b, c: (a > 1) & (b < 6), "AND"),
            (lambda a, b, c: (a > 2) | (b < 6), "OR"),
            (lambda a, b, c: ((a > 1) & (b < 6)) | (c > 7), "chained AND-OR"),
            (lambda a, b, c: (a < 2) | ((b == 5) & (c < 9)), "chained OR-AND"),
        ],
    )
    def test_logical_operations_chained(self, sample_df, expr_func, description):
        result_expr = expr_func(col("a"), col("b"), col("c")).alias("result")
        expected_expr = expr_func(pl.col("a"), pl.col("b"), pl.col("c")).alias("result")

        result = sample_df.select(result_expr.to_native()).to_series()
        expected = sample_df.select(expected_expr).to_series()

        assert result.to_list() == expected.to_list()

    @pytest.mark.parametrize(
        "op_name, expr_func",
        [
            ("==", lambda col, val: col == val),
            ("!=", lambda col, val: col != val),
            ("<", lambda col, val: col < val),
            ("<=", lambda col, val: col <= val),
            (">", lambda col, val: col > val),
            (">=", lambda col, val: col >= val),
        ],
    )
    def test_temporal_column_string_comparison_completes(self, op_name, expr_func) -> None:
        """Comparisons coerce like Spark: string/timestamp mix evaluates without error."""
        df = pl.DataFrame({"birth_date": [datetime(1990, 1, 1), datetime(1985, 5, 15), datetime(1970, 12, 30)]})
        sparkle_df = DataFrame(df)

        for other in ("2024-01-01", lit("2024-01-01")):
            expr = expr_func(col("birth_date"), other)
            out = sparkle_df.select(expr.alias("result")).to_native_df()["result"]
            assert out.dtype == pl.Boolean

        expr = expr_func(col("birth_date"), datetime(2024, 1, 1))
        out = sparkle_df.select(expr.alias("result")).to_native_df()["result"]
        assert out.dtype == pl.Boolean

    @pytest.mark.parametrize(
        "values, pattern, expected",
        [
            (["apple", "banana", "apricot"], "^a", [True, False, True]),  # Starts with 'a'
            (["apple", "banana", "apricot"], "a$", [False, True, False]),  # Ends with 'a'
            (["car", "cat", "dog"], "ca.", [True, True, False]),  # Starts with 'ca' and any char
            (["spark", "flame", "flash"], ".*a.*", [True, True, True]),  # Contains 'a'
            (["123", "abc", "456"], r"\d+", [True, False, True]),  # Digits only
        ],
    )
    def test_rlike(self, values, pattern, expected):
        df = pl.DataFrame({"col": values})
        sparkle_df = DataFrame(df)

        expr = col("col").rlike(pattern).alias("result")
        result = sparkle_df.select(expr).to_native_df()["result"].to_list()

        assert result == expected

    @pytest.mark.parametrize(
        "values, pattern",
        [
            (["apple", "banana", "apricot", None], "ap"),  # basic substring
            (["car", "cat", "dog", None], "ca"),  # multiple matches
            (["Spark", "spark", "SPARK", None], "spark"),  # case sensitivity
            (["a.b", "ab", "a*b", None], "."),  # literal special char (should be literal, not regex)
            (["a.b", "ab", "a*b", None], "*"),  # another literal special char
        ],
    )
    def test_contains_matches_pyspark(self, spark, values, pattern):
        # Build Polars DF and evaluate with sparkleframe's 'contains'
        pl_df = pl.DataFrame({"idx": list(range(len(values))), "text": values})
        sf_df = DataFrame(pl_df)
        expr = col("text").contains(pattern).alias("result")
        pl_result = sf_df.select(col("idx"), expr).to_native_df()

        # Convert the Polars result to a Spark DataFrame
        spark_from_polars = spark.createDataFrame(pl_result.to_pandas())

        # Build a pure Spark DataFrame and compute expected result using PySpark's contains
        spark_input = spark.createDataFrame(list(enumerate(values)), schema=["idx", "text"])
        expected = spark_input.select("idx", F.col("text").contains(pattern).alias("result"))

        # Compare results deterministically by ordering on idx
        actual_rows = spark_from_polars.orderBy("idx").collect()
        expected_rows = expected.orderBy("idx").collect()

        assert actual_rows == expected_rows


class TestColumnComparisonCoercion:
    """
    Spark-like comparison: float cast for ordering; ``==`` / ``!=`` use numeric
    match when both sides parse as numbers, else string comparison.
    """

    def _eval(self, expr, df: pl.DataFrame) -> pl.Series:
        return df.select(expr.to_native()).to_series()

    def test_eq_int_column_to_string_literal_numeric_match(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = self._eval(col("a") == lit("1"), df)
        assert result.to_list() == [True, False, False]

    def test_eq_string_column_to_int_literal_numeric_match(self) -> None:
        df = pl.DataFrame({"s": ["1", "2", "x"]})
        result = self._eval(col("s") == lit(1), df)
        assert result.to_list() == [True, False, False]

    def test_eq_fallback_string_when_either_not_numeric(self) -> None:
        df = pl.DataFrame({"a": [1, 1], "t": ["x", "1"]})
        r = self._eval(col("t") == col("a"), df)
        assert r.to_list() == [False, True]

    def test_ne_mixed_complements_eq(self) -> None:
        df = pl.DataFrame({"a": [1, 2]})
        eq = self._eval(col("a") == lit("1"), df)
        ne = self._eval(col("a") != lit("1"), df)
        assert ne.to_list() == [not v for v in eq.to_list()]

    def test_ordering_uses_float_coercion(self) -> None:
        df = pl.DataFrame({"s": ["1.5", "2", "10"]})
        # Lexicographic would order "10" < "2"; float orders 2 < 10.
        lt = self._eval(col("s") < lit(2), df)
        assert lt.to_list() == [True, False, False]

    def test_ordering_iso_datetime_string_vs_date_sub_offer_age(self) -> None:
        """``created >= date_sub(current_date(), 30)`` must be boolean, not null."""
        today = date.today()
        recent = pl.DataFrame({"created": [f"{today.isoformat()}T12:00:00Z"]})
        assert self._eval(col("created") >= date_sub(current_date(), 30), recent).to_list() == [True]
        past = date.fromordinal(today.toordinal() - 40)
        old = pl.DataFrame({"created": [f"{past.isoformat()}T12:00:00Z"]})
        assert self._eval(col("created") >= date_sub(current_date(), 30), old).to_list() == [False]
