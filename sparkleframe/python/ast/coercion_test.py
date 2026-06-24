"""Unit tests for the analyze-phase coercion rules (over real PySpark types)."""

import pytest
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    NullType,
    StringType,
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
        ],
    )
    def test_infer(self, value, expected):
        assert isinstance(coercion.infer_literal_type(value), expected)


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
