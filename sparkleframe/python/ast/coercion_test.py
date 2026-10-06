"""Unit tests for the analyze-phase coercion rules (over real PySpark types)."""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

from sparkleframe.python.ast import coercion


class TestWidenNumeric:
    def test_same_type_unchanged(self):
        assert isinstance(coercion.widen_numeric(IntegerType(), IntegerType()), IntegerType)

    def test_int_and_long_widens_to_long(self):
        assert isinstance(coercion.widen_numeric(IntegerType(), LongType()), LongType)
        assert isinstance(coercion.widen_numeric(LongType(), IntegerType()), LongType)

    def test_long_and_double_widens_to_double(self):
        assert isinstance(coercion.widen_numeric(LongType(), DoubleType()), DoubleType)


class TestInferLiteralType:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (1, IntegerType),
            (2**40, LongType),
            (3.14, DoubleType),
            ("x", StringType),
            (True, BooleanType),
            (None, NullType),
            (date(2024, 1, 1), DateType),
            (datetime(2024, 1, 1, 12), TimestampType),
            (b"ab", BinaryType),
        ],
    )
    def test_infer(self, value, expected):
        assert isinstance(coercion.infer_literal_type(value), expected)

    @pytest.mark.parametrize(
        "value, precision, scale",
        [(Decimal("1.50"), 3, 2), (Decimal("0.05"), 2, 2), (Decimal("12"), 2, 0), (Decimal("1E+2"), 3, 0)],
    )
    def test_decimal_literal_typed_by_its_digits(self, value, precision, scale):
        assert coercion.infer_literal_type(value) == DecimalType(precision, scale)

    def test_timezone_aware_datetime_not_implemented(self):
        with pytest.raises(NotImplementedError):
            coercion.infer_literal_type(datetime(2024, 1, 1, tzinfo=timezone.utc))


class TestCoerceArithmetic:
    def test_int_plus_int_no_casts(self):
        result, left_cast, right_cast = coercion.coerce_arithmetic("+", IntegerType(), IntegerType())
        assert isinstance(result, IntegerType)
        assert left_cast is None and right_cast is None

    def test_int_plus_long_casts_int_operand(self):
        result, left_cast, right_cast = coercion.coerce_arithmetic("+", IntegerType(), LongType())
        assert isinstance(result, LongType)
        assert isinstance(left_cast, LongType)
        assert right_cast is None

    def test_double_plus_string_casts_string_to_double(self):
        result, left_cast, right_cast = coercion.coerce_arithmetic("+", DoubleType(), StringType())
        assert isinstance(result, DoubleType)
        assert left_cast is None
        assert isinstance(right_cast, DoubleType)

    def test_int_plus_string_promotes_to_long(self):
        # An integral numeric + string promotes the string (and the numeric) to long, not double.
        result, left_cast, right_cast = coercion.coerce_arithmetic("+", IntegerType(), StringType())
        assert isinstance(result, LongType)
        assert isinstance(left_cast, LongType)
        assert isinstance(right_cast, LongType)

    def test_string_plus_long_promotes_to_long(self):
        # Long is already the promotion target, so the numeric operand needs no cast.
        result, left_cast, right_cast = coercion.coerce_arithmetic("+", StringType(), LongType())
        assert isinstance(result, LongType)
        assert isinstance(left_cast, LongType)
        assert right_cast is None

    def test_unsupported_operator_raises(self):
        with pytest.raises(NotImplementedError):
            coercion.coerce_arithmetic("/", IntegerType(), IntegerType())

    def test_unsupported_operand_types_raise(self):
        with pytest.raises(NotImplementedError):
            coercion.coerce_arithmetic("+", StringType(), StringType())


class TestCoercePow:
    def test_double_operands_need_no_cast(self):
        result, left_cast, right_cast = coercion.coerce_pow(DoubleType(), DoubleType())
        assert isinstance(result, DoubleType)
        assert left_cast is None and right_cast is None

    @pytest.mark.parametrize("dtype", [IntegerType(), LongType(), DecimalType(10, 2), StringType(), NullType()])
    def test_non_double_operand_is_cast_to_double(self, dtype):
        # The result is double even when both operands share a non-double type (int ** int -> double).
        result, left_cast, right_cast = coercion.coerce_pow(dtype, dtype)
        assert isinstance(result, DoubleType)
        assert isinstance(left_cast, DoubleType)
        assert isinstance(right_cast, DoubleType)

    def test_only_the_non_double_side_is_cast(self):
        _, left_cast, right_cast = coercion.coerce_pow(DoubleType(), IntegerType())
        assert left_cast is None
        assert isinstance(right_cast, DoubleType)

    @pytest.mark.parametrize("left, right", [(BooleanType(), DoubleType()), (DoubleType(), DateType())])
    def test_non_numeric_operand_raises_type_error(self, left, right):
        with pytest.raises(TypeError, match="DATATYPE_MISMATCH"):
            coercion.coerce_pow(left, right)


class TestCoerceComparison:
    @pytest.mark.parametrize(
        "left, right, target",
        [
            # numeric widening; a float/double on either side compares as double
            (ByteType(), IntegerType(), IntegerType()),
            (IntegerType(), LongType(), LongType()),
            (ShortType(), FloatType(), DoubleType()),
            (FloatType(), DoubleType(), DoubleType()),
            (DecimalType(10, 2), DoubleType(), DoubleType()),
            # integral x decimal -> the wider decimal (int is decimal(10,0))
            (IntegerType(), DecimalType(10, 2), DecimalType(12, 2)),
            (ByteType(), DecimalType(10, 2), DecimalType(10, 2)),
            # string x integral -> bigint; string x fractional/decimal -> double
            (StringType(), ShortType(), LongType()),
            (StringType(), DecimalType(10, 2), DoubleType()),
            (StringType(), FloatType(), DoubleType()),
            # string x boolean/date/timestamp/binary -> the other type
            (BooleanType(), StringType(), BooleanType()),
            (StringType(), DateType(), DateType()),
            (TimestampType(), StringType(), TimestampType()),
            (StringType(), BinaryType(), BinaryType()),
            (DateType(), TimestampType(), TimestampType()),
            # a null operand takes the other side's type
            (NullType(), DateType(), DateType()),
        ],
    )
    def test_common_type(self, left, right, target):
        result, left_cast, right_cast = coercion.coerce_comparison(left, right)
        assert isinstance(result, BooleanType)
        assert left_cast == (None if left == target else target)
        assert right_cast == (None if right == target else target)

    def test_same_type_needs_no_cast(self):
        array_int = ArrayType(IntegerType())
        assert coercion.coerce_comparison(array_int, array_int) == (BooleanType(), None, None)

    @pytest.mark.parametrize(
        "left, right",
        [(IntegerType(), BooleanType()), (DateType(), DoubleType()), (BinaryType(), IntegerType())],
    )
    def test_unrelated_types_raise_type_error(self, left, right):
        with pytest.raises(TypeError, match="BINARY_OP_DIFF_TYPES"):
            coercion.coerce_comparison(left, right)

    def test_maps_are_not_orderable_even_when_nested(self):
        nested = StructType([StructField("m", MapType(StringType(), IntegerType()))])
        with pytest.raises(TypeError, match="INVALID_ORDERING_TYPE"):
            coercion.coerce_comparison(nested, nested)

    def test_differing_complex_types_not_implemented(self):
        # Spark widens array<int> vs array<bigint> element-wise; the engine does not yet.
        with pytest.raises(NotImplementedError):
            coercion.coerce_comparison(ArrayType(IntegerType()), ArrayType(LongType()))


class TestCoerceLogical:
    def test_boolean_operands_need_no_cast(self):
        assert coercion.coerce_logical("and", BooleanType(), BooleanType()) == (BooleanType(), None, None)

    def test_string_and_null_operands_cast_to_boolean(self):
        assert coercion.coerce_logical("or", StringType(), NullType()) == (BooleanType(), BooleanType(), BooleanType())
        assert coercion.coerce_not(StringType()) == (BooleanType(), BooleanType())

    @pytest.mark.parametrize("dtype", [IntegerType(), DoubleType(), DateType()])
    def test_non_boolean_operand_raises_type_error(self, dtype):
        with pytest.raises(TypeError, match="DATATYPE_MISMATCH"):
            coercion.coerce_logical("and", BooleanType(), dtype)
        with pytest.raises(TypeError, match="DATATYPE_MISMATCH"):
            coercion.coerce_not(dtype)

    def test_types_outside_the_modeled_set_not_implemented(self):
        with pytest.raises(NotImplementedError):
            coercion.coerce_comparison(TimestampNTZType(), TimestampNTZType())
