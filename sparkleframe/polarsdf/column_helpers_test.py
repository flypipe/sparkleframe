"""
Unit tests for the pure helpers in :mod:`sparkleframe.polarsdf.column_helpers`.

The arithmetic / comparison validators are exercised end-to-end by
``TestArithmeticParityWithSpark``, but the build-time branches (when the operand
dtype is already known) are not hit by those parity tests because the Spark
oracle constructs expressions outside the active schema context. These focused
unit tests pin the helper contracts in isolation so refactors can't silently
weaken them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import polars as pl
import pytest

from sparkleframe.polarsdf.column_helpers import (
    _assert_arithmetic_compatible,
    _assert_arithmetic_series,
    _assert_complex_compare_supported,
    _assert_spark_compare_compatible,
    _assert_spark_compare_compatible_eager,
    _cmp_exprs,
    _coerce_mixed_arithmetic_operands,
    _compare_series,
    _equality_comparison_expr,
    _expr_is_literal,
    _format_unsupported_complex_compare,
    _has_decimal_operand,
    _is_complex_polars_dtype,
    _is_date_polars_dtype,
    _is_integer_polars_dtype,
    _is_numeric_polars_dtype,
    _is_string_polars_dtype,
    _looks_like_map_dtype,
    _needs_date_datetime_promotion,
    _numeric_compare_operands,
    _ordering_comparison_expr,
    _parse_bool_string,
    _parse_datetime_string_safe,
    _polars_schema_for,
    _promote_date_to_datetime_pair,
    _spark_compare_group,
    _spark_decimal_div_result_type,
    _spark_numeric_widened_type,
    _string_to_bool_expr,
    _string_to_bool_expr_strict,
    _validate_and_cast_float64_lenient,
    _validate_and_cast_float64_strict,
    _validated_arithmetic_expr,
    _validated_decimal_div_expr,
    _validated_float64_expr,
    _validated_pow_expr,
)


class TestIsComplexPolarsDtype:
    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (pl.List(pl.Int64), True),
            (pl.Struct([pl.Field("x", pl.Int64)]), True),
            (pl.Array(pl.Int64, 3), True),
            (pl.Int64, False),
            (pl.Utf8, False),
            (None, False),
        ],
    )
    def test_complex_detection(self, dtype: Optional[pl.DataType], expected: bool) -> None:
        assert _is_complex_polars_dtype(dtype) is expected


class TestIsNumericPolarsDtype:
    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (pl.Int8, True),
            (pl.Int16, True),
            (pl.Int32, True),
            (pl.Int64, True),
            (pl.UInt32, True),
            (pl.Float32, True),
            (pl.Float64, True),
            (pl.Utf8, False),
            (pl.Boolean, False),
            (pl.Binary, False),
        ],
    )
    def test_numeric_detection(self, dtype: pl.DataType, expected: bool) -> None:
        assert _is_numeric_polars_dtype(dtype) is expected


class TestAssertArithmeticCompatible:
    @pytest.mark.parametrize(
        "dtype",
        [None, pl.Int32, pl.Int64, pl.Float64, pl.Duration, pl.Datetime, pl.Date, pl.Null],
    )
    def test_accepts(self, dtype: Optional[pl.DataType]) -> None:
        _assert_arithmetic_compatible(dtype)

    @pytest.mark.parametrize(
        "dtype",
        [pl.Boolean, pl.Utf8, pl.String, pl.Binary, pl.List(pl.Int64), pl.Struct([pl.Field("a", pl.Int64)])],
    )
    def test_rejects(self, dtype: pl.DataType) -> None:
        with pytest.raises(TypeError, match="Cannot resolve arithmetic operation"):
            _assert_arithmetic_compatible(dtype)


class TestAssertArithmeticSeries:
    @pytest.mark.parametrize(
        "series",
        [
            pl.Series("a", [1, 2, 3], dtype=pl.Int32),
            pl.Series("a", [1.5, 2.5], dtype=pl.Float64),
            pl.Series("a", [datetime(2024, 1, 1)], dtype=pl.Datetime),
        ],
    )
    def test_accepts(self, series: pl.Series) -> None:
        assert _assert_arithmetic_series(series).equals(series)

    @pytest.mark.parametrize(
        "series",
        [
            pl.Series("a", [True, False], dtype=pl.Boolean),
            pl.Series("a", ["x"], dtype=pl.Utf8),
            pl.Series("a", [b"\x01"], dtype=pl.Binary),
            pl.Series("a", [[1, 2]], dtype=pl.List(pl.Int64)),
        ],
    )
    def test_rejects(self, series: pl.Series) -> None:
        with pytest.raises(TypeError, match="Cannot resolve arithmetic operation"):
            _assert_arithmetic_series(series)


class TestValidateAndCastFloat64Strict:
    def test_int_casts_to_float(self) -> None:
        out = _validate_and_cast_float64_strict(pl.Series("a", [1, 2], dtype=pl.Int32))
        assert out.dtype == pl.Float64
        assert out.to_list() == [1.0, 2.0]

    @pytest.mark.parametrize(
        "series",
        [
            pl.Series("a", ["x"], dtype=pl.Utf8),
            pl.Series("a", [True], dtype=pl.Boolean),
            pl.Series("a", [b"\x01"], dtype=pl.Binary),
            pl.Series("a", [datetime(2024, 1, 1)], dtype=pl.Datetime),
            pl.Series("a", [[1, 2]], dtype=pl.List(pl.Int64)),
        ],
    )
    def test_rejects_non_numeric(self, series: pl.Series) -> None:
        with pytest.raises(TypeError, match="Cannot resolve arithmetic operation"):
            _validate_and_cast_float64_strict(series)


class TestValidateAndCastFloat64Lenient:
    def test_string_casts_to_float(self) -> None:
        out = _validate_and_cast_float64_lenient(pl.Series("a", ["1.5", "2.5"], dtype=pl.Utf8))
        assert out.dtype == pl.Float64
        assert out.to_list() == [1.5, 2.5]

    @pytest.mark.parametrize(
        "series",
        [
            pl.Series("a", [True], dtype=pl.Boolean),
            pl.Series("a", [b"\x01"], dtype=pl.Binary),
            pl.Series("a", [datetime(2024, 1, 1)], dtype=pl.Datetime),
            pl.Series("a", [[1, 2]], dtype=pl.List(pl.Int64)),
        ],
    )
    def test_rejects(self, series: pl.Series) -> None:
        with pytest.raises(TypeError, match="Cannot resolve arithmetic operation"):
            _validate_and_cast_float64_lenient(series)


class TestValidatedArithmeticExpr:
    def test_known_compatible_dtype_returns_expr_unchanged(self) -> None:
        out = _validated_arithmetic_expr(pl.col("a"), pl.Int32)
        df = pl.DataFrame({"a": [1, 2, 3]}, schema={"a": pl.Int32})
        assert df.select(out.alias("r")).to_series().dtype == pl.Int32

    def test_known_incompatible_dtype_raises_at_build_time(self) -> None:
        with pytest.raises(TypeError):
            _validated_arithmetic_expr(pl.col("a"), pl.Utf8)


class TestValidatedFloat64Expr:
    def test_known_compatible_casts_to_float64(self) -> None:
        out = _validated_float64_expr(pl.col("a"), pl.Int32)
        df = pl.DataFrame({"a": [1, 2, 3]}, schema={"a": pl.Int32})
        assert df.select(out.alias("r")).to_series().dtype == pl.Float64

    def test_known_incompatible_raises(self) -> None:
        with pytest.raises(TypeError):
            _validated_float64_expr(pl.col("a"), pl.Utf8)


class TestValidatedPowExpr:
    def test_string_known_dtype_casts_to_float64(self) -> None:
        # pow auto-casts strings: build-time path should NOT raise.
        out = _validated_pow_expr(pl.col("a"), pl.Utf8)
        df = pl.DataFrame({"a": ["1.5", "2.5"]}, schema={"a": pl.Utf8})
        assert df.select(out.alias("r")).to_series().dtype == pl.Float64

    @pytest.mark.parametrize(
        "dtype",
        # Pass dtype instances (not classes); resolved_dt comes from a Polars schema
        # collect, which returns concrete instances for parameterised dtypes.
        [pl.Binary, pl.Date(), pl.Datetime("us"), pl.Duration("us"), pl.List(pl.Int64)],
    )
    def test_rejected_known_dtypes_raise(self, dtype: pl.DataType) -> None:
        with pytest.raises(TypeError, match="Cannot resolve arithmetic operation"):
            _validated_pow_expr(pl.col("a"), dtype)


class TestSparkNumericWidenedType:
    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (pl.Int32, pl.Int32, None),  # same rank -> no cast
            (pl.Int32, pl.Int64, pl.Int64),  # widen to long
            (pl.Float32, pl.Int64, pl.Float32),  # float ranks higher than long in Spark
            (None, pl.Int32, pl.Float64),  # unknown side -> Float64
            (pl.Int32, None, pl.Float64),
            (pl.Decimal(10, 2), pl.Int32, pl.Float64),  # decimals fall back to Float64
            (pl.Utf8, pl.Int32, pl.Float64),  # non-numeric -> Float64 sentinel
        ],
    )
    def test_widening(
        self,
        left: Optional[pl.DataType],
        right: Optional[pl.DataType],
        expected: Optional[pl.DataType],
    ) -> None:
        assert _spark_numeric_widened_type(left, right) == expected


class TestLooksLikeMapDtype:
    def test_canonical_map_shape(self) -> None:
        dt = pl.List(pl.Struct([pl.Field("key", pl.Utf8), pl.Field("value", pl.Int64)]))
        assert _looks_like_map_dtype(dt) is True

    @pytest.mark.parametrize(
        "dt",
        [
            None,
            pl.List(pl.Int64),  # plain list
            pl.List(pl.Struct([pl.Field("a", pl.Int64)])),  # struct but not key/value
            pl.Struct([pl.Field("key", pl.Utf8), pl.Field("value", pl.Int64)]),  # not wrapped in List
        ],
    )
    def test_non_map(self, dt: Optional[pl.DataType]) -> None:
        assert _looks_like_map_dtype(dt) is False


class TestFormatUnsupportedComplexCompare:
    def test_map_message(self) -> None:
        dt = pl.List(pl.Struct([pl.Field("key", pl.Utf8), pl.Field("value", pl.Int64)]))
        msg = _format_unsupported_complex_compare("eq", dt, dt)
        assert "MapType" in msg
        assert "'=='" in msg

    def test_nested_message(self) -> None:
        msg = _format_unsupported_complex_compare("lt", pl.List(pl.Int64), pl.List(pl.Int64))
        assert "List / Struct / Array" in msg
        assert "'<'" in msg


class TestAssertComplexCompareSupported:
    def test_no_complex_no_raise(self) -> None:
        _assert_complex_compare_supported("lt", pl.Int32, pl.Int32)
        _assert_complex_compare_supported("eq", pl.Utf8, pl.Int32)

    def test_equality_on_plain_list_ok(self) -> None:
        # ==/!= on List/Struct/Array is OK -- Polars supports it natively.
        _assert_complex_compare_supported("eq", pl.List(pl.Int64), pl.List(pl.Int64))
        _assert_complex_compare_supported("ne", pl.List(pl.Int64), pl.List(pl.Int64))

    @pytest.mark.parametrize("op", ["lt", "le", "gt", "ge"])
    def test_ordering_on_complex_raises(self, op: str) -> None:
        with pytest.raises(NotImplementedError, match="List / Struct / Array"):
            _assert_complex_compare_supported(op, pl.List(pl.Int64), pl.List(pl.Int64))

    @pytest.mark.parametrize("op", ["eq", "ne", "lt", "le", "gt", "ge"])
    def test_any_op_on_map_raises(self, op: str) -> None:
        dt = pl.List(pl.Struct([pl.Field("key", pl.Utf8), pl.Field("value", pl.Int64)]))
        with pytest.raises(NotImplementedError, match="MapType"):
            _assert_complex_compare_supported(op, dt, dt)


class TestCmpExprs:
    @pytest.mark.parametrize(
        "op, expected",
        [
            ("lt", [True, False, False]),
            ("le", [True, True, False]),
            ("gt", [False, False, True]),
            ("ge", [False, True, True]),
        ],
    )
    def test_each_operator(self, op: str, expected: list[bool]) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [2, 2, 2]})
        result = df.select(_cmp_exprs(pl.col("a"), pl.col("b"), op).alias("r")).to_series().to_list()
        assert result == expected

    def test_unknown_op_raises(self) -> None:
        with pytest.raises(ValueError):
            _cmp_exprs(pl.col("a"), pl.col("b"), "??")


class TestParseDatetimeStringSafe:
    @pytest.mark.parametrize(
        "value",
        ["2024-01-01", "2024-01-01T08:30:00", "2024/01/01 08:30:00"],
    )
    def test_parsable_strings(self, value: str) -> None:
        parsed = _parse_datetime_string_safe(value)
        assert isinstance(parsed, datetime)
        assert parsed.tzinfo is None

    @pytest.mark.parametrize(
        "value",
        ["", "not a date", "12345"],
    )
    def test_non_parsable_strings(self, value: str) -> None:
        assert _parse_datetime_string_safe(value) is None


class TestNumericCompareOperands:
    def test_returns_four_aligned_expressions(self) -> None:
        df = pl.DataFrame({"a": ["1", "2.5", "x"], "b": ["3", "1.5", "y"]})
        left_num, right_num, left_str, right_str = _numeric_compare_operands(pl.col("a"), pl.col("b"))
        out = df.select(
            left_num.alias("ln"),
            right_num.alias("rn"),
            left_str.alias("ls"),
            right_str.alias("rs"),
        )
        assert out.schema["ln"] == pl.Float64
        assert out.schema["rn"] == pl.Float64
        assert out.schema["ls"] == pl.Utf8
        assert out.schema["rs"] == pl.Utf8
        assert out["ln"].to_list() == [1.0, 2.5, None]
        assert out["rn"].to_list() == [3.0, 1.5, None]
        assert out["ls"].to_list() == ["1", "2.5", "x"]


class TestOrderingComparisonExpr:
    def test_numeric_branch(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [2, 2, 2]}, schema={"a": pl.Int64, "b": pl.Int64})
        result = df.select(_ordering_comparison_expr(pl.col("a"), pl.col("b"), "lt").alias("r"))
        assert result["r"].to_list() == [True, False, False]

    def test_cross_type_int_vs_string_col_col_raises(self) -> None:
        # Spark 4 rejects col(int) vs col(string) column-column comparisons.
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["2", "2", "2"]}, schema={"a": pl.Int64, "b": pl.Utf8})
        with pytest.raises(TypeError, match="data type mismatch"):
            df.select(_ordering_comparison_expr(pl.col("a"), pl.col("b"), "le").alias("r"))

    def test_cross_type_int_vs_string_literal_coerces(self) -> None:
        # Spark coerces lit('2') to 2 when compared with an Int column: numeric branch wins.
        df = pl.DataFrame({"a": [1, 2, 3]}, schema={"a": pl.Int64})
        result = df.select(_ordering_comparison_expr(pl.col("a"), pl.lit("2"), "le").alias("r"))
        assert result["r"].to_list() == [True, True, False]

    def test_string_branch_when_both_string_dtype(self) -> None:
        df = pl.DataFrame({"a": ["a", "b", "c"], "b": ["b", "b", "b"]}, schema={"a": pl.Utf8, "b": pl.Utf8})
        result = df.select(_ordering_comparison_expr(pl.col("a"), pl.col("b"), "lt").alias("r"))
        assert result["r"].to_list() == [True, False, False]

    def test_ordering_on_complex_raises_at_build_time(self) -> None:
        # Known List dtype: raises BEFORE the expression is even materialised.
        df = pl.DataFrame({"a": [[1], [2]], "b": [[1], [3]]}, schema={"a": pl.List(pl.Int64), "b": pl.List(pl.Int64)})
        with pytest.raises(NotImplementedError, match="List / Struct / Array"):
            df.select(_ordering_comparison_expr(pl.col("a"), pl.col("b"), "lt").alias("r"))


class TestEqualityComparisonExpr:
    def test_numeric_equal(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 0, 3]}, schema={"a": pl.Int64, "b": pl.Int64})
        result = df.select(_equality_comparison_expr(pl.col("a"), pl.col("b"), equal=True).alias("r"))
        assert result["r"].to_list() == [True, False, True]

    def test_numeric_not_equal(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 0, 3]}, schema={"a": pl.Int64, "b": pl.Int64})
        result = df.select(_equality_comparison_expr(pl.col("a"), pl.col("b"), equal=False).alias("r"))
        assert result["r"].to_list() == [False, True, False]

    def test_cross_type_int_vs_string_col_col_raises(self) -> None:
        # Spark 4 rejects col(int) vs col(string) column-column comparisons.
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["1", "2", "4"]}, schema={"a": pl.Int64, "b": pl.Utf8})
        with pytest.raises(TypeError, match="data type mismatch"):
            df.select(_equality_comparison_expr(pl.col("a"), pl.col("b"), equal=True).alias("r"))

    def test_cross_type_int_vs_string_literal_coerces(self) -> None:
        # Spark coerces lit('1') to 1 when compared with an Int column: numeric branch wins.
        df = pl.DataFrame({"a": [1, 2, 3]}, schema={"a": pl.Int64})
        result = df.select(_equality_comparison_expr(pl.col("a"), pl.lit("1"), equal=True).alias("r"))
        assert result["r"].to_list() == [True, False, False]

    def test_complex_equal_uses_native_polars(self) -> None:
        # ==/!= on plain List is supported via native Polars; helper must NOT raise.
        df = pl.DataFrame(
            {"a": [[1, 2], [3]], "b": [[1, 2], [4]]},
            schema={"a": pl.List(pl.Int64), "b": pl.List(pl.Int64)},
        )
        result = df.select(_equality_comparison_expr(pl.col("a"), pl.col("b"), equal=True).alias("r"))
        assert result["r"].to_list() == [True, False]

    def test_map_like_equality_raises(self) -> None:
        # MapType (Polars: List(Struct([key, value]))) -> NotImplementedError.
        map_dt = pl.List(pl.Struct([pl.Field("key", pl.Utf8), pl.Field("value", pl.Int64)]))
        df = pl.DataFrame(
            {"a": [[{"key": "k", "value": 1}]], "b": [[{"key": "k", "value": 1}]]},
            schema={"a": map_dt, "b": map_dt},
        )
        with pytest.raises(NotImplementedError, match="MapType"):
            df.select(_equality_comparison_expr(pl.col("a"), pl.col("b"), equal=True).alias("r"))


class TestParseBoolString:
    @pytest.mark.parametrize(
        "value",
        ["true", "TRUE", " True ", "t", "1", "yes", "Y"],
    )
    def test_truthy(self, value: str) -> None:
        assert _parse_bool_string(value) is True

    @pytest.mark.parametrize(
        "value",
        ["false", "FALSE", "  False", "f", "0", "no", "N"],
    )
    def test_falsy(self, value: str) -> None:
        assert _parse_bool_string(value) is False

    @pytest.mark.parametrize(
        "value",
        [None, "", "maybe", "2", "yepp"],
    )
    def test_null_or_unrecognised(self, value) -> None:
        assert _parse_bool_string(value) is None

    def test_non_string_input_is_stringified(self) -> None:
        # ``str(1).lower() == "1"`` -> True; ``str(7).lower() == "7"`` -> None.
        assert _parse_bool_string(1) is True
        assert _parse_bool_string(0) is False
        assert _parse_bool_string(7) is None


class TestStringToBoolExpr:
    def test_full_table(self) -> None:
        df = pl.DataFrame(
            {"s": ["true", "FALSE", " yes", "n", "0", "maybe", None]},
            schema={"s": pl.Utf8},
        )
        result = df.select(_string_to_bool_expr(pl.col("s")).alias("v"))
        assert result.schema["v"] == pl.Boolean
        assert result["v"].to_list() == [True, False, True, False, False, None, None]

    def test_empty_frame_keeps_dtype(self) -> None:
        df = pl.DataFrame({"s": []}, schema={"s": pl.Utf8})
        result = df.select(_string_to_bool_expr(pl.col("s")).alias("v"))
        assert result.schema["v"] == pl.Boolean
        assert result.height == 0


class TestStringToBoolExprStrict:
    def test_string_valid(self) -> None:
        df = pl.DataFrame({"s": ["true", "FALSE", " yes", "n", "0", None]}, schema={"s": pl.Utf8})
        result = df.select(_string_to_bool_expr_strict(pl.col("s")).alias("v"))
        assert result.schema["v"] == pl.Boolean
        assert result["v"].to_list() == [True, False, True, False, False, None]

    def test_raises_on_unrecognised_string(self) -> None:
        df = pl.DataFrame({"s": ["true", "maybe", None]}, schema={"s": pl.Utf8})
        with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
            df.select(_string_to_bool_expr_strict(pl.col("s")).alias("v"))

    def test_int_source_uses_native_cast(self) -> None:
        # int -> bool: nonzero -> True, 0 -> False, null -> null (no raise; Spark-compatible).
        df = pl.DataFrame({"a": [1, 2, 0, None]}, schema={"a": pl.Int64})
        result = df.select(_string_to_bool_expr_strict(pl.col("a")).alias("v"))
        assert result.schema["v"] == pl.Boolean
        assert result["v"].to_list() == [True, True, False, None]

    def test_float_source_uses_native_cast(self) -> None:
        df = pl.DataFrame({"a": [1.5, 0.0, -2.0, None]}, schema={"a": pl.Float64})
        result = df.select(_string_to_bool_expr_strict(pl.col("a")).alias("v"))
        assert result.schema["v"] == pl.Boolean
        assert result["v"].to_list() == [True, False, True, None]

    def test_empty_frame_keeps_dtype(self) -> None:
        df = pl.DataFrame({"s": []}, schema={"s": pl.Utf8})
        result = df.select(_string_to_bool_expr_strict(pl.col("s")).alias("v"))
        assert result.schema["v"] == pl.Boolean
        assert result.height == 0


class TestSparkCompareGroup:
    @pytest.mark.parametrize(
        "dt, expected",
        [
            (pl.Int64, "integer"),
            (pl.UInt8, "integer"),
            (pl.Float32, "float"),
            (pl.Float64, "float"),
            (pl.Decimal(10, 2), "decimal"),
            (pl.Datetime("us"), "temporal"),
            (pl.Duration("ms"), "temporal"),
            (pl.Date, "temporal"),
            (pl.Utf8, "string"),
            (pl.String, "string"),
            (pl.Boolean, "boolean"),
            (pl.Binary, "binary"),
            (pl.List(pl.Int64), "complex"),
            (pl.Struct([pl.Field("a", pl.Int64)]), "complex"),
            (None, None),
        ],
    )
    def test_group_mapping(self, dt: Optional[pl.DataType], expected: Optional[str]) -> None:
        assert _spark_compare_group(dt) == expected


class TestExprIsLiteral:
    def test_literal_returns_true(self) -> None:
        assert _expr_is_literal(pl.lit(42)) is True

    def test_column_returns_false(self) -> None:
        assert _expr_is_literal(pl.col("a")) is False


class TestAssertSparkCompareCompatibleEager:
    def test_compatible_pair_passes(self) -> None:
        _assert_spark_compare_compatible_eager("eq", pl.Int64, pl.Float64)

    def test_incompatible_pair_raises(self) -> None:
        with pytest.raises(TypeError, match="data type mismatch"):
            _assert_spark_compare_compatible_eager("eq", pl.Int64, pl.Boolean)

    def test_unknown_group_passes(self) -> None:
        _assert_spark_compare_compatible_eager("eq", pl.Null, pl.Int64)


class TestAssertSparkCompareCompatible:
    def test_compatible_passes(self) -> None:
        _assert_spark_compare_compatible("eq", pl.col("a"), pl.col("b"), pl.Int64, pl.Float64)

    def test_incompatible_col_col_raises(self) -> None:
        with pytest.raises(TypeError, match="data type mismatch"):
            _assert_spark_compare_compatible("eq", pl.col("a"), pl.col("b"), pl.Int64, pl.Boolean)

    def test_incompatible_with_literal_passes(self) -> None:
        _assert_spark_compare_compatible("eq", pl.col("a"), pl.lit(True), pl.Int64, pl.Boolean)

    def test_unknown_dtype_passes(self) -> None:
        _assert_spark_compare_compatible("eq", pl.col("a"), pl.col("b"), None, pl.Int64)


class TestIsStringPolarsDtype:
    @pytest.mark.parametrize("dt", [pl.Utf8, pl.String])
    def test_string_types(self, dt: pl.DataType) -> None:
        assert _is_string_polars_dtype(dt) is True

    @pytest.mark.parametrize("dt", [pl.Int64, pl.Boolean, None])
    def test_non_string_types(self, dt: Optional[pl.DataType]) -> None:
        assert _is_string_polars_dtype(dt) is False


class TestIsDatePolarsDtype:
    def test_date(self) -> None:
        assert _is_date_polars_dtype(pl.Date) is True

    @pytest.mark.parametrize("dt", [pl.Datetime("us"), pl.Int64, None])
    def test_non_date(self, dt: Optional[pl.DataType]) -> None:
        assert _is_date_polars_dtype(dt) is False


class TestIsIntegerPolarsDtype:
    @pytest.mark.parametrize("dt", [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32])
    def test_integers(self, dt: pl.DataType) -> None:
        assert _is_integer_polars_dtype(dt) is True

    @pytest.mark.parametrize("dt", [pl.Float64, pl.Utf8, None])
    def test_non_integers(self, dt: Optional[pl.DataType]) -> None:
        assert _is_integer_polars_dtype(dt) is False


class TestHasDecimalOperand:
    def test_one_decimal(self) -> None:
        assert _has_decimal_operand(pl.Decimal(10, 2), pl.Int64) is True
        assert _has_decimal_operand(pl.Int64, pl.Decimal(10, 2)) is True

    def test_no_decimal(self) -> None:
        assert _has_decimal_operand(pl.Int64, pl.Float64) is False
        assert _has_decimal_operand(None, None) is False


class TestSparkDecimalDivResultType:
    def test_decimal_div_decimal(self) -> None:
        result = _spark_decimal_div_result_type(pl.Decimal(10, 2), pl.Decimal(10, 2))
        assert isinstance(result, pl.Decimal)
        assert result.scale == max(6, 2 + 10 + 1)  # 13

    def test_int_div_decimal(self) -> None:
        result = _spark_decimal_div_result_type(pl.Int64, pl.Decimal(10, 2))
        assert isinstance(result, pl.Decimal)


class TestValidatedDecimalDivExpr:
    def test_non_decimal_casts_to_decimal(self) -> None:
        expr = _validated_decimal_div_expr(pl.col("a"), pl.Int64)
        df = pl.DataFrame({"a": [10, 20]}, schema={"a": pl.Int64})
        result = df.select(expr.alias("r"))
        assert isinstance(result.schema["r"], pl.Decimal)

    def test_decimal_stays_decimal(self) -> None:
        expr = _validated_decimal_div_expr(pl.col("a"), pl.Decimal(10, 2))
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Decimal(10, 2))})
        result = df.select(expr.alias("r"))
        assert isinstance(result.schema["r"], pl.Decimal)

    def test_incompatible_dtype_raises(self) -> None:
        with pytest.raises(TypeError):
            _validated_decimal_div_expr(pl.col("a"), pl.Utf8)


class TestCoerceMixedArithmeticOperands:
    def test_returns_none_when_dtypes_unknown(self) -> None:
        assert _coerce_mixed_arithmetic_operands(pl.col("a"), pl.col("b"), None, None, "+") is None

    def test_numeric_plus_string_coerces(self) -> None:
        with _polars_schema_for(pl.Schema({"a": pl.Int64, "b": pl.Utf8})):
            result = _coerce_mixed_arithmetic_operands(pl.col("a"), pl.col("b"), pl.Int64, pl.Utf8, "+")
            assert result is not None

    def test_string_plus_numeric_coerces(self) -> None:
        result = _coerce_mixed_arithmetic_operands(pl.col("a"), pl.col("b"), pl.Utf8, pl.Float64, "+")
        assert result is not None

    def test_int_plus_date_coerces(self) -> None:
        result = _coerce_mixed_arithmetic_operands(pl.col("a"), pl.col("b"), pl.Int32, pl.Date, "+")
        assert result is not None

    def test_date_plus_int_coerces(self) -> None:
        result = _coerce_mixed_arithmetic_operands(pl.col("a"), pl.col("b"), pl.Date, pl.Int32, "+")
        assert result is not None

    def test_same_type_returns_none(self) -> None:
        assert _coerce_mixed_arithmetic_operands(pl.col("a"), pl.col("b"), pl.Int64, pl.Int64, "+") is None


class TestNeedsDateDatetimePromotion:
    def test_date_vs_datetime(self) -> None:
        assert _needs_date_datetime_promotion(pl.Date, pl.Datetime("us")) is True

    def test_datetime_vs_date(self) -> None:
        assert _needs_date_datetime_promotion(pl.Datetime("us"), pl.Date) is True

    def test_same_type(self) -> None:
        assert _needs_date_datetime_promotion(pl.Date, pl.Date) is False
        assert _needs_date_datetime_promotion(pl.Datetime("us"), pl.Datetime("us")) is False

    def test_unrelated_types(self) -> None:
        assert _needs_date_datetime_promotion(pl.Int64, pl.Utf8) is False


class TestPromoteDateToDatetimePair:
    def test_date_left_datetime_right(self) -> None:
        left = pl.Series("a", ["2024-01-01"], dtype=pl.Date)
        right = pl.Series("b", ["2024-01-01T12:00:00"], dtype=pl.Datetime("us"))
        l_out, r_out = _promote_date_to_datetime_pair(left, right)
        assert isinstance(l_out.dtype, pl.Datetime)
        assert r_out.dtype == right.dtype

    def test_datetime_left_date_right(self) -> None:
        left = pl.Series("a", ["2024-01-01T12:00:00"], dtype=pl.Datetime("us"))
        right = pl.Series("b", ["2024-01-01"], dtype=pl.Date)
        l_out, r_out = _promote_date_to_datetime_pair(left, right)
        assert l_out.dtype == left.dtype
        assert isinstance(r_out.dtype, pl.Datetime)

    def test_no_promotion_needed(self) -> None:
        left = pl.Series("a", [1, 2], dtype=pl.Int64)
        right = pl.Series("b", [3, 4], dtype=pl.Int64)
        l_out, r_out = _promote_date_to_datetime_pair(left, right)
        assert l_out.dtype == pl.Int64
        assert r_out.dtype == pl.Int64


class TestCompareSeries:
    def test_eq(self) -> None:
        a = pl.Series("a", [1, 2, 3])
        b = pl.Series("b", [1, 0, 3])
        assert _compare_series(a, b, "eq").to_list() == [True, False, True]

    def test_ne(self) -> None:
        a = pl.Series("a", [1, 2, 3])
        b = pl.Series("b", [1, 0, 3])
        assert _compare_series(a, b, "ne").to_list() == [False, True, False]

    def test_lt(self) -> None:
        a = pl.Series("a", [1, 2, 3])
        b = pl.Series("b", [2, 2, 2])
        assert _compare_series(a, b, "lt").to_list() == [True, False, False]

    def test_le(self) -> None:
        a = pl.Series("a", [1, 2, 3])
        b = pl.Series("b", [2, 2, 2])
        assert _compare_series(a, b, "le").to_list() == [True, True, False]

    def test_gt(self) -> None:
        a = pl.Series("a", [1, 2, 3])
        b = pl.Series("b", [2, 2, 2])
        assert _compare_series(a, b, "gt").to_list() == [False, False, True]

    def test_ge(self) -> None:
        a = pl.Series("a", [1, 2, 3])
        b = pl.Series("b", [2, 2, 2])
        assert _compare_series(a, b, "ge").to_list() == [False, True, True]

    def test_invalid_op_raises(self) -> None:
        a = pl.Series("a", [1])
        b = pl.Series("b", [1])
        with pytest.raises(ValueError):
            _compare_series(a, b, "??")


class TestOrderingDateDatetimePromotion:
    def test_date_lt_datetime(self) -> None:
        df = pl.DataFrame(
            {
                "a": pl.Series(["2024-01-01", "2024-06-15"], dtype=pl.Date),
                "b": pl.Series(["2024-01-01T12:00:00", "2024-06-15T00:00:00"], dtype=pl.Datetime("us")),
            }
        )
        with _polars_schema_for(df.schema):
            result = df.select(_ordering_comparison_expr(pl.col("a"), pl.col("b"), "lt").alias("r"))
        assert result["r"].to_list() == [True, False]


class TestEqualityDateDatetimePromotion:
    def test_date_eq_datetime_midnight(self) -> None:
        df = pl.DataFrame(
            {
                "a": pl.Series(["2024-01-01"], dtype=pl.Date),
                "b": pl.Series(["2024-01-01T00:00:00"], dtype=pl.Datetime("us")),
            }
        )
        with _polars_schema_for(df.schema):
            result = df.select(_equality_comparison_expr(pl.col("a"), pl.col("b"), equal=True).alias("r"))
        assert result["r"].to_list() == [True]

    def test_date_ne_datetime(self) -> None:
        df = pl.DataFrame(
            {
                "a": pl.Series(["2024-01-01"], dtype=pl.Date),
                "b": pl.Series(["2024-01-01T12:00:00"], dtype=pl.Datetime("us")),
            }
        )
        with _polars_schema_for(df.schema):
            result = df.select(_equality_comparison_expr(pl.col("a"), pl.col("b"), equal=False).alias("r"))
        assert result["r"].to_list() == [True]
