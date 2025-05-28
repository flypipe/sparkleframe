import pytest

from sparkleframe.polarsdf import DataFrame
from sparkleframe.polarsdf.functions import col
from sparkleframe.polarsdf.types import *



@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
        "d": ["cat", "dog", "bird"]
    })

class TestColumn:
    def evaluate_expr(self, expr, df: pl.DataFrame) -> pl.Series:
        return df.select(expr.to_native()).to_series()

    @pytest.mark.parametrize("op_name,expr_func", [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        ("/", lambda a, b: a / b),
    ])
    def test_arithmetics(self, sample_df, op_name, expr_func):
        result = self.evaluate_expr(expr_func(col("a"), col("b")), sample_df)

        expected = sample_df.select((expr_func(pl.col("a"), pl.col("b"))).alias("result")).to_series()
        assert result.to_list() == expected.to_list()

    @pytest.mark.parametrize("expr_func, expected_func", [
        (lambda a: 10 + col("a"), lambda df: 10 + pl.col("a")),
        (lambda a: 10 - col("a"), lambda df: 10 - pl.col("a")),
        (lambda a: 10 * col("a"), lambda df: 10 * pl.col("a")),
        (lambda a: 10 / col("a"), lambda df: 10 / pl.col("a")),
    ])
    def test_reverse_arithmetics(self, sample_df, expr_func, expected_func):
        expr = expr_func(col("a"))
        result = self.evaluate_expr(expr, sample_df)

        expected = sample_df.select(expected_func(sample_df).alias("result")).to_series()
        assert result.to_list() == expected.to_list()

    @pytest.mark.parametrize("op_name, expr_func", [
        ("==", lambda a, b: a == b),
        ("!=", lambda a, b: a != b),
        ("<", lambda a, b: a < b),
        ("<=", lambda a, b: a <= b),
        (">", lambda a, b: a > b),
        (">=", lambda a, b: a >= b),
    ])
    def test_comparisons(self, sample_df, op_name, expr_func):
        result = self.evaluate_expr(expr_func(col("a"), col("b")), sample_df)

        expected = sample_df.select((expr_func(pl.col("a"), pl.col("b"))).alias("result")).to_series()
        assert result.to_list() == expected.to_list()


    def test_chained_expression(self, sample_df):
        expr = (col("c") - col("a") + col("b")) * col("b") / col("a")
        result = self.evaluate_expr(expr, sample_df)

        expected = sample_df.select(((pl.col("c") - pl.col("a") + pl.col("b")) * pl.col("b") / pl.col("a")).alias("result")).to_series()
        assert result.to_list() == expected.to_list()

    def test_alias(self, sample_df):
        expr = (col("a") + col("b")).alias("sum_ab")
        result = sample_df.select(expr.to_native()).to_series()

        expected = sample_df.select((pl.col("a") + pl.col("b")).alias("sum_ab")).to_series()
        assert result.to_list() == expected.to_list()

    @pytest.mark.parametrize("column_name, values", [
        ("a", [1, 2]),  # int
        ("a", [10]),  # int (no match)
        ("d", ["cat", "dog"]),  # str
        ("d", ["fish"]),  # str (no match)
        ("d", []),  # empty list
    ])
    def test_isin(self, sample_df, column_name, values):
        expr = col(column_name).isin(values)
        result = self.evaluate_expr(expr, sample_df)

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

    @pytest.mark.parametrize("data_type_class, expected_polars_dtype", [
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
    ])
    def test_cast_types(self, sample_df, data_type_class, expected_polars_dtype):
        # Cast column 'a' to the specified type
        expr = col("a").cast(data_type_class())
        result_df = DataFrame(sample_df).select(expr.alias("casted"))

        # Assert the column's dtype is as expected
        assert result_df.to_native_df().schema["casted"] == expected_polars_dtype

    def test_is_not_null(self):
        df = pl.DataFrame({
            "x": [1, None, 3, None, 5]
        })

        expr = col("x").isNotNull()
        result = df.select(expr.to_native().alias("result")).to_series()

        expected = df.select(pl.col("x").is_not_null().alias("result")).to_series()

        assert result.to_list() == expected.to_list()
