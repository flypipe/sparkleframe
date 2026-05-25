import itertools
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

import polars as pl
import pyspark.sql.functions as F
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

import sparkleframe.polarsdf.functions as PF
from sparkleframe.polarsdf import DataFrame, StringType
from sparkleframe.polarsdf.functions import col, lit
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
from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal, spark_rows_from_dict


def _functions_pow_optional():
    """``functions.pow`` is not on every ``main`` snapshot; skip tests until it exists."""
    import importlib

    mod = importlib.import_module("sparkleframe.polarsdf.functions")
    if not hasattr(mod, "pow"):
        pytest.skip("requires sparkleframe.polarsdf.functions.pow (merge base / later PR)")
    return mod.pow


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

    def test_pow_operator_matches_sf_pow(self, sample_df):
        """``col ** k`` matches :func:`~sparkleframe.polarsdf.functions.pow` (e.g. annuity-style ``(1+r)**-N``)."""
        sf_pow = _functions_pow_optional()
        n = 3
        r = col("a") / 10.0
        via_op = 1 - (1 + r) ** (-n)
        via_fn = 1 - sf_pow(1 + r, -n)
        assert self.evaluate_expr(via_op, sample_df).to_list() == self.evaluate_expr(via_fn, sample_df).to_list()

    def test_pow_literal_base_column_exponent(self, sample_df):
        """``pow(2, col)`` must not treat ``2`` as a column name."""
        sf_pow = _functions_pow_optional()
        result = self.evaluate_expr(sf_pow(2, col("a")), sample_df)
        expected = sample_df.select(pl.lit(2.0).pow(pl.col("a").cast(pl.Float64)).alias("result")).to_series()
        assert result.to_list() == expected.to_list()

    def test_unary_neg_column(self, sample_df):
        """Spark allows ``-col``; used e.g. as ``pow(..., -n)`` when ``n`` is a column."""
        result = self.evaluate_expr(-col("a"), sample_df)
        expected = sample_df.select((-pl.col("a")).alias("result")).to_series()
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

    def test_is_null(self):
        df = pl.DataFrame({"x": [1, None, 3, None, 5]})

        expr = col("x").isNull()
        result = df.select(expr.to_native().alias("result")).to_series()

        expected = df.select(pl.col("x").is_null().alias("result")).to_series()

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

    def test_eq_cross_type_col_col_raises(self) -> None:
        """Spark 4 raises AnalysisException for col(string) == col(int)."""
        df = pl.DataFrame({"a": [1, 1], "t": ["x", "1"]})
        with pytest.raises(TypeError, match="data type mismatch"):
            self._eval(col("t") == col("a"), df)

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
        from sparkleframe.polarsdf.functions import current_date, date_sub

        today = date.today()
        recent = pl.DataFrame({"created": [f"{today.isoformat()}T12:00:00Z"]})
        assert self._eval(col("created") >= date_sub(current_date(), 30), recent).to_list() == [True]
        past = date.fromordinal(today.toordinal() - 40)
        old = pl.DataFrame({"created": [f"{past.isoformat()}T12:00:00Z"]})
        assert self._eval(col("created") >= date_sub(current_date(), 30), old).to_list() == [False]


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
    """Return an xfail reason for known gaps, or ``None`` if the test should run.

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


class TestArithmeticParityWithSpark:
    """
    Parity tests: for every pyspark.sql.types scalar type and every arithmetic / comparison
    operator, sparkleframe must produce the same result as PySpark.  When Spark raises,
    sparkleframe must also raise.
    """

    @staticmethod
    def _make_dfs(spark, spark_type, values_a, values_b):
        schema = SparkStructType(
            [
                SparkStructField("a", spark_type),
                SparkStructField("b", spark_type),
            ]
        )
        rows = list(zip(values_a, values_b))
        spark_df = spark.createDataFrame(rows, schema)
        sparkle_df = DataFrame(pl.from_arrow(spark_df.toArrow()))
        return spark_df, sparkle_df

    @pytest.mark.parametrize(
        "dtype_label, spark_type, values_a, values_b",
        _ARITHMETIC_TYPE_FIXTURES,
    )
    @pytest.mark.parametrize("op_name, op_func", _ARITHMETIC_OPS)
    def test_arithmetic_col_col(self, spark, dtype_label, spark_type, values_a, values_b, op_name, op_func):
        spark_df, sparkle_df = self._make_dfs(spark, spark_type, values_a, values_b)

        spark_raised = False
        try:
            spark_result = spark_df.select(op_func(F.col("a"), F.col("b")).alias("result"))
            spark_result.collect()
        except Exception:
            spark_raised = True

        if spark_raised:
            with pytest.raises(Exception):
                sparkle_df.select(op_func(PF.col("a"), PF.col("b")).alias("result")).to_native_df()
        else:
            sf_result = sparkle_df.select(op_func(PF.col("a"), PF.col("b")).alias("result"))
            assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

    @pytest.mark.parametrize(
        "dtype_label, spark_type, values_a, values_b",
        _ARITHMETIC_TYPE_FIXTURES,
    )
    @pytest.mark.parametrize("op_name, op_func", _COMPARISON_OPS)
    def test_comparison_col_col(self, spark, dtype_label, spark_type, values_a, values_b, op_name, op_func):
        if (dtype_label, op_name) in _MAP_COMPARISON_GAPS:
            pytest.xfail("Polars stores maps as List(Struct) -- indistinguishable from arrays")
        if (dtype_label, op_name) in _COMPLEX_ORDERING_GAPS:
            pytest.xfail("Polars does not support <, <=, >, >= on List / Struct dtypes")
        spark_df, sparkle_df = self._make_dfs(spark, spark_type, values_a, values_b)

        spark_raised = False
        try:
            spark_result = spark_df.select(op_func(F.col("a"), F.col("b")).alias("result"))
            spark_result.collect()
        except Exception:
            spark_raised = True

        if spark_raised:
            with pytest.raises(Exception):
                sparkle_df.select(op_func(PF.col("a"), PF.col("b")).alias("result")).to_native_df()
        else:
            sf_result = sparkle_df.select(op_func(PF.col("a"), PF.col("b")).alias("result"))
            assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

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
    def test_arithmetic_with_nulls(self, spark, dtype_label, spark_type, values_a, values_b, op_name, op_func):
        schema = SparkStructType(
            [
                SparkStructField("a", spark_type, nullable=True),
                SparkStructField("b", spark_type, nullable=True),
            ]
        )
        rows = list(zip(values_a, values_b))
        spark_df = spark.createDataFrame(rows, schema)
        sparkle_df = DataFrame(pl.from_arrow(spark_df.toArrow()))

        spark_result = spark_df.select(op_func(F.col("a"), F.col("b")).alias("result"))
        sf_result = sparkle_df.select(op_func(PF.col("a"), PF.col("b")).alias("result"))

        assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

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
        "op_name, sf_op, spark_op",
        [
            ("radd", lambda v: 10 + v, lambda v: 10 + v),
            ("rsub", lambda v: 10 - v, lambda v: 10 - v),
            ("rmul", lambda v: 10 * v, lambda v: 10 * v),
            ("rtruediv", lambda v: 10 / v, lambda v: 10 / v),
        ],
    )
    def test_arithmetic_literal_col(self, spark, dtype_label, spark_type, values, op_name, sf_op, spark_op):
        schema = SparkStructType([SparkStructField("a", spark_type)])
        rows = [(v,) for v in values]
        spark_df = spark.createDataFrame(rows, schema)
        sparkle_df = DataFrame(pl.from_arrow(spark_df.toArrow()))

        spark_result = spark_df.select(spark_op(F.col("a")).alias("result"))
        sf_result = sparkle_df.select(sf_op(PF.col("a")).alias("result"))

        assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

    @staticmethod
    def _make_dfs_mixed(spark, left_spark_type, left_values, right_spark_type, right_values):
        schema = SparkStructType(
            [
                SparkStructField("a", left_spark_type),
                SparkStructField("b", right_spark_type),
            ]
        )
        rows = list(zip(left_values, right_values))
        spark_df = spark.createDataFrame(rows, schema)
        sparkle_df = DataFrame(pl.from_arrow(spark_df.toArrow()))
        return spark_df, sparkle_df

    @pytest.mark.parametrize(
        "pair_label, left_spark_type, left_values, right_spark_type, right_values",
        _MIXED_TYPE_PAIRS,
        ids=[p[0] for p in _MIXED_TYPE_PAIRS],
    )
    @pytest.mark.parametrize("op_name, op_func", _ARITHMETIC_OPS)
    def test_arithmetic_col_col_mixed_types(
        self, spark, pair_label, left_spark_type, left_values, right_spark_type, right_values, op_name, op_func
    ):
        reason = _mixed_pair_xfail_reason(pair_label, op_name)
        if reason:
            pytest.xfail(reason)

        spark_df, sparkle_df = self._make_dfs_mixed(
            spark, left_spark_type, left_values, right_spark_type, right_values
        )

        spark_raised = False
        try:
            spark_result = spark_df.select(op_func(F.col("a"), F.col("b")).alias("result"))
            spark_result.collect()
        except Exception:
            spark_raised = True

        if spark_raised:
            with pytest.raises(Exception):
                sparkle_df.select(op_func(PF.col("a"), PF.col("b")).alias("result")).to_native_df()
        else:
            sf_result = sparkle_df.select(op_func(PF.col("a"), PF.col("b")).alias("result"))
            assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

    @pytest.mark.parametrize(
        "pair_label, left_spark_type, left_values, right_spark_type, right_values",
        _MIXED_TYPE_PAIRS,
        ids=[p[0] for p in _MIXED_TYPE_PAIRS],
    )
    @pytest.mark.parametrize("op_name, op_func", _COMPARISON_OPS)
    def test_comparison_col_col_mixed_types(
        self, spark, pair_label, left_spark_type, left_values, right_spark_type, right_values, op_name, op_func
    ):
        reason = _mixed_pair_xfail_reason(pair_label, op_name)
        if reason:
            pytest.xfail(reason)

        spark_df, sparkle_df = self._make_dfs_mixed(
            spark, left_spark_type, left_values, right_spark_type, right_values
        )

        spark_raised = False
        try:
            spark_result = spark_df.select(op_func(F.col("a"), F.col("b")).alias("result"))
            spark_result.collect()
        except Exception:
            spark_raised = True

        if spark_raised:
            with pytest.raises(Exception):
                sparkle_df.select(op_func(PF.col("a"), PF.col("b")).alias("result")).to_native_df()
        else:
            sf_result = sparkle_df.select(op_func(PF.col("a"), PF.col("b")).alias("result"))
            assert_sparkle_spark_frame_are_equal(sf_result, spark_result)


class TestUnsupportedComplexComparisons:
    """Sparkleframe must raise a clear ``NotImplementedError`` -- not an opaque Polars
    error -- for compare operations it cannot replicate from Spark yet."""

    @pytest.mark.parametrize("op_name, op_func", [("<", lambda a, b: a < b), ("<=", lambda a, b: a <= b)])
    def test_ordering_on_list_raises_not_implemented(self, op_name, op_func):
        df = DataFrame(pl.DataFrame({"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]}))
        with pytest.raises(NotImplementedError, match=r"sparkleframe does not support the '.+' operator on List"):
            df.select(op_func(col("a"), col("b")).alias("r")).to_native_df()

    @pytest.mark.parametrize("op_name, op_func", [(">", lambda a, b: a > b), (">=", lambda a, b: a >= b)])
    def test_ordering_on_struct_raises_not_implemented(self, op_name, op_func):
        df = DataFrame(
            pl.DataFrame(
                {"a": [{"x": 1, "y": 2}, {"x": 3, "y": 4}], "b": [{"x": 5, "y": 6}, {"x": 7, "y": 8}]},
            )
        )
        with pytest.raises(NotImplementedError, match=r"sparkleframe does not support the '.+' operator on List"):
            df.select(op_func(col("a"), col("b")).alias("r")).to_native_df()

    @pytest.mark.parametrize("op_name, op_func", [("==", lambda a, b: a == b), ("!=", lambda a, b: a != b)])
    def test_equality_on_map_raises_not_implemented(self, op_name, op_func):
        df = DataFrame(
            pl.DataFrame(
                {
                    "a": [[{"key": "x", "value": 1}], [{"key": "y", "value": 2}]],
                    "b": [[{"key": "x", "value": 1}], [{"key": "y", "value": 2}]],
                }
            )
        )
        with pytest.raises(NotImplementedError, match=r"sparkleframe does not support the '.+' operator on MapType"):
            df.select(op_func(col("a"), col("b")).alias("r")).to_native_df()

    def test_equality_on_plain_list_still_works(self):
        """Sanity check: List equality is supported and must not raise NotImplementedError."""
        df = DataFrame(pl.DataFrame({"a": [[1, 2], [3, 4]], "b": [[1, 2], [7, 8]]}))
        result = df.select((col("a") == col("b")).alias("r")).to_native_df()
        assert result.to_series().to_list() == [True, False]


class TestCastParityWithSpark:
    """
    Parity coverage for ``Column.cast`` and ``Column.try_cast`` against real Spark 4.

    Spark 4 enables ``spark.sql.ansi.enabled`` by default, so ``cast`` of a malformed
    value raises ``CAST_INVALID_INPUT`` while ``try_cast`` returns ``NULL``;
    sparkleframe matches both behaviours.
    """

    @staticmethod
    def _make_dfs(spark, rows, spark_type):
        schema = SparkStructType([SparkStructField("s", spark_type)])
        spark_df = spark.createDataFrame(rows, schema)
        sparkle_df = DataFrame(pl.from_arrow(spark_df.toArrow()))
        return spark_df, sparkle_df

    @pytest.mark.parametrize(
        "sparkle_dtype, spark_dtype",
        [
            (IntegerType(), SparkIntegerType()),
            (LongType(), SparkLongType()),
            (DoubleType(), SparkDoubleType()),
            (FloatType(), SparkFloatType()),
        ],
    )
    def test_cast_string_to_numeric_valid(self, spark, sparkle_dtype, spark_dtype):
        rows = [("123",), ("-7",), (None,)]
        spark_df, sparkle_df = self._make_dfs(spark, rows, SparkStringType())
        spark_result = spark_df.select(F.col("s").cast(spark_dtype).alias("v"))
        sparkle_result = sparkle_df.select(col("s").cast(sparkle_dtype).alias("v"))
        assert_sparkle_spark_frame_are_equal(sparkle_result, spark_result)

    @pytest.mark.parametrize(
        "sparkle_dtype, spark_dtype",
        [
            (IntegerType(), SparkIntegerType()),
            (LongType(), SparkLongType()),
            (DoubleType(), SparkDoubleType()),
        ],
    )
    def test_try_cast_string_to_numeric_invalid_returns_null(self, spark, sparkle_dtype, spark_dtype):
        rows = [("123",), ("Bob",), (None,)]
        spark_df, sparkle_df = self._make_dfs(spark, rows, SparkStringType())
        spark_result = spark_df.select(F.col("s").try_cast(spark_dtype).alias("v"))
        sparkle_result = sparkle_df.select(col("s").try_cast(sparkle_dtype).alias("v"))
        assert_sparkle_spark_frame_are_equal(sparkle_result, spark_result)

    @pytest.mark.parametrize(
        "sparkle_dtype, spark_dtype",
        [
            (IntegerType(), SparkIntegerType()),
            (LongType(), SparkLongType()),
            (DoubleType(), SparkDoubleType()),
        ],
    )
    def test_cast_string_to_numeric_invalid_raises_like_spark_ansi(self, spark, sparkle_dtype, spark_dtype):
        """Both Spark 4 (ANSI default) and sparkleframe raise when an invalid string can't be cast."""
        rows = [("123",), ("Bob",), (None,)]
        spark_df, sparkle_df = self._make_dfs(spark, rows, SparkStringType())
        with pytest.raises(Exception):
            spark_df.select(F.col("s").cast(spark_dtype).alias("v")).collect()
        with pytest.raises(Exception):
            sparkle_df.select(col("s").cast(sparkle_dtype).alias("v")).to_native_df()

    def test_cast_string_to_boolean_invalid_raises_like_spark_ansi(self, spark):
        """Same parity for boolean casts: Spark and sparkleframe both raise on 'maybe'."""
        rows = [("true",), ("maybe",), (None,)]
        spark_df, sparkle_df = self._make_dfs(spark, rows, SparkStringType())
        with pytest.raises(Exception):
            spark_df.select(F.col("s").cast(SparkBooleanType()).alias("v")).collect()
        with pytest.raises(Exception):
            sparkle_df.select(col("s").cast(BooleanType()).alias("v")).to_native_df()

    def test_cast_int_to_boolean_parity(self, spark):
        """Spark casts nonzero int -> True, 0 -> False; no raise. Sparkleframe must match."""
        rows = [(1,), (2,), (0,), (-1,), (None,)]
        spark_df, sparkle_df = self._make_dfs(spark, rows, SparkIntegerType())
        spark_result = spark_df.select(F.col("s").cast(SparkBooleanType()).alias("v"))
        sparkle_result = sparkle_df.select(col("s").cast(BooleanType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(sparkle_result, spark_result)

    def test_cast_double_to_boolean_parity(self, spark):
        rows = [(1.5,), (0.0,), (-3.2,), (None,)]
        spark_df, sparkle_df = self._make_dfs(spark, rows, SparkDoubleType())
        spark_result = spark_df.select(F.col("s").cast(SparkBooleanType()).alias("v"))
        sparkle_result = sparkle_df.select(col("s").cast(BooleanType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(sparkle_result, spark_result)

    def test_try_cast_string_to_boolean_parity(self, spark):
        # Only test the literals Spark actually accepts for string -> boolean.
        rows = [("true",), ("TRUE",), ("false",), ("FALSE",), ("t",), ("f",), ("1",), ("0",), ("y",), ("n",), (None,)]
        spark_df, sparkle_df = self._make_dfs(spark, rows, SparkStringType())
        spark_result = spark_df.select(F.col("s").try_cast(SparkBooleanType()).alias("v"))
        sparkle_result = sparkle_df.select(col("s").try_cast(BooleanType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(sparkle_result, spark_result)

    def test_cast_string_to_boolean_valid(self, spark):
        # Valid literals only -- avoids the ANSI raise divergence above.
        rows = [("true",), ("false",), ("t",), ("f",), ("1",), ("0",), (None,)]
        spark_df, sparkle_df = self._make_dfs(spark, rows, SparkStringType())
        spark_result = spark_df.select(F.col("s").cast(SparkBooleanType()).alias("v"))
        sparkle_result = sparkle_df.select(col("s").cast(BooleanType()).alias("v"))
        assert_sparkle_spark_frame_are_equal(sparkle_result, spark_result)

    @pytest.mark.parametrize(
        "sparkle_dtype, spark_dtype",
        [
            (StringType(), SparkStringType()),
            (DoubleType(), SparkDoubleType()),
            (LongType(), SparkLongType()),
        ],
    )
    def test_cast_int_to_other_parity(self, spark, sparkle_dtype, spark_dtype):
        rows = [(1,), (-5,), (0,), (None,)]
        spark_df, sparkle_df = self._make_dfs(spark, rows, SparkIntegerType())
        spark_result = spark_df.select(F.col("s").cast(spark_dtype).alias("v"))
        sparkle_result = sparkle_df.select(col("s").cast(sparkle_dtype).alias("v"))
        assert_sparkle_spark_frame_are_equal(sparkle_result, spark_result)

    def test_try_cast_inside_when_then_against_spark(self, spark):
        """Polars evaluates all when/then branches eagerly, so strict cast fails on
        non-matching rows.  Use try_cast inside when/then as a workaround; the result
        matches PySpark's short-circuit cast behavior."""
        from pyspark.sql.functions import when as spark_when

        from sparkleframe.polarsdf.functions import when

        data = {"v": ["1", "2", "N/A"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))

        sf_result = polars_df.withColumn(
            "num",
            when(col("v").rlike(r"^\d+$"), col("v").try_cast(IntegerType())).otherwise(lit(-1)),
        )
        sp_result = spark_df.withColumn(
            "num",
            spark_when(F.col("v").rlike(r"^\d+$"), F.col("v").cast(SparkIntegerType())).otherwise(F.lit(-1)),
        )
        assert_sparkle_spark_frame_are_equal(sf_result, sp_result)
