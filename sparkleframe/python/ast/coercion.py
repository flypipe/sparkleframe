"""Spark type-coercion rules for the **analyze** phase, written over the engine's types.

These are pure functions: given the resolved ``data_type`` of an operand (a
:mod:`sparkleframe.python.types` ``DataType``), they decide the result type of an operation and where Spark would insert an
implicit cast. They are a clean rewrite of the coercion behavior Spark documents — they do
**not** port ``polarsdf`` internals (per #104 Q4).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional, Tuple

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DayTimeIntervalType,
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
    TimestampType,
)


def _numeric_rank(dtype: Any) -> int:
    """Spark numeric widening rank: ``Byte < Short < Int < Long < Float < Double``."""
    order = (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType)
    for rank, cls in enumerate(order):
        if isinstance(dtype, cls):
            return rank
    raise ValueError(f"{type(dtype).__name__} is not a numeric type")


def is_numeric(dtype: Any) -> bool:
    """Whether ``dtype`` is one of the numeric types this slice widens over."""
    return isinstance(dtype, (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType))


def is_string(dtype: Any) -> bool:
    return isinstance(dtype, StringType)


def widen_numeric(left: Any, right: Any) -> Any:
    """Return the higher-ranked of two numeric types (the common widened type).

    Same type in → same type out (``int + int -> int``).
    """
    return left if _numeric_rank(left) >= _numeric_rank(right) else right


def _is_integral(dtype: Any) -> bool:
    """Whether ``dtype`` is one of Spark's integral types (Byte/Short/Int/Long)."""
    return isinstance(dtype, (ByteType, ShortType, IntegerType, LongType))


def _string_arith_promote_type(numeric_dt: Any) -> Any:
    """Type the string operand (and numeric operand) is cast to in ``numeric <op> string``.

    Spark 4 ANSI promotes the string to ``long`` when the numeric operand is integral
    (``byte/short/int/long`` all widen to ``long``) and to ``double`` when it is floating
    (``float`` widens to ``double``).
    """
    return LongType() if _is_integral(numeric_dt) else DoubleType()


def infer_literal_type(value: Any) -> Any:
    """Infer the PySpark type of a Python literal (minimal set this slice needs)."""
    if value is None:
        return NullType()
    # ``bool`` is a subclass of ``int`` — check it first.
    if isinstance(value, bool):
        return BooleanType()
    if isinstance(value, int):
        # Spark widens an out-of-int32-range literal to bigint.
        return IntegerType() if -(2**31) <= value < 2**31 else LongType()
    if isinstance(value, float):
        return DoubleType()
    if isinstance(value, str):
        return StringType()
    if isinstance(value, Decimal):
        return _decimal_literal_type(value)
    # ``datetime`` is a subclass of ``date`` — check it first. A naive datetime is a session-local
    # timestamp; an aware one needs time-zone conversion the engine does not model yet.
    if isinstance(value, datetime) and value.tzinfo is None:
        return TimestampType()
    if isinstance(value, date) and not isinstance(value, datetime):
        return DateType()
    if isinstance(value, (bytes, bytearray)):
        return BinaryType()
    not_implemented_yet(f"literal type inference for {type(value).__name__}")


def _decimal_literal_type(value: Decimal) -> Any:
    """Spark types a ``Decimal`` literal by its own digits: ``1.50 -> decimal(3,2)``, ``0.05 -> decimal(2,2)``."""
    digits, exponent = len(value.as_tuple().digits), value.as_tuple().exponent
    if exponent > 0:  # e.g. Decimal("1E+2"): Spark rescales a negative scale to 0.
        return DecimalType(digits + exponent, 0)
    scale = -exponent
    return DecimalType(max(digits, scale), scale)


def coerce_arithmetic(op: str, left_dt: Any, right_dt: Any) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Resolve the result type (and any operand casts) of ``left <op> right`` for ``+ - * /``.

    Returns ``(result_dt, left_cast_dt, right_cast_dt)`` where a ``*_cast_dt`` is the type the
    corresponding operand must be cast to (or ``None`` if no cast is needed). ``**`` resolves
    through :func:`coerce_pow`.

    - ``/`` is never integral: integral / integral and anything with a float/double/string is
      ``double``; with a decimal operand it follows Spark's decimal division rule.
    - Decimal operands keep their own type (the evaluator computes exactly, then rounds to the
      result type); an integral operand is cast to its narrowest decimal (``int -> decimal(10,0)``).
    - A ``null`` operand is cast to the other operand's type.
    """
    if op not in ("+", "-", "*", "/"):
        not_implemented_yet(f"arithmetic result type for operator {op!r}")

    if isinstance(left_dt, NullType) and isinstance(right_dt, NullType):
        result = DoubleType() if op == "/" else NullType()
        return result, None, None
    if isinstance(left_dt, NullType):
        result, left_cast, right_cast = coerce_arithmetic(op, right_dt, right_dt)
        return result, left_cast or right_dt, right_cast
    if isinstance(right_dt, NullType):
        result, left_cast, right_cast = coerce_arithmetic(op, left_dt, left_dt)
        return result, left_cast, right_cast or left_dt

    numeric = (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType)
    if isinstance(left_dt, numeric) and isinstance(right_dt, numeric):
        if isinstance(left_dt, (FloatType, DoubleType)) or isinstance(right_dt, (FloatType, DoubleType)):
            if op == "/" or isinstance(left_dt, DecimalType) or isinstance(right_dt, DecimalType):
                result = DoubleType()
            else:
                result = widen_numeric(left_dt, right_dt)
            return result, _cast_unless(left_dt, result), _cast_unless(right_dt, result)
        if isinstance(left_dt, DecimalType) or isinstance(right_dt, DecimalType):
            left_dec, right_dec = _as_decimal(left_dt), _as_decimal(right_dt)
            result = decimal_arithmetic_type(op, left_dec, right_dec)
            return result, _cast_unless(left_dt, left_dec), _cast_unless(right_dt, right_dec)
        if op == "/":
            return DoubleType(), DoubleType(), DoubleType()
        result = widen_numeric(left_dt, right_dt)
        return result, _cast_unless(left_dt, result), _cast_unless(right_dt, result)

    # numeric + string: Spark promotes the string to the numeric operand's arithmetic type —
    # ``long`` when the numeric is integral (``int + "3" -> long``) and ``double`` when it is
    # floating (``double + "3.14" -> double``). The numeric operand widens to that type too, and
    # ``/`` then divides those as doubles (so ``int / "1.5"`` fails casting "1.5" to long).
    if is_string(left_dt) and isinstance(right_dt, numeric):
        promoted = _string_arith_promote_type(right_dt)
        result = DoubleType() if op == "/" else promoted
        return result, promoted, _cast_unless(right_dt, promoted)
    if isinstance(left_dt, numeric) and is_string(right_dt):
        promoted = _string_arith_promote_type(left_dt)
        result = DoubleType() if op == "/" else promoted
        return result, _cast_unless(left_dt, promoted), promoted

    datetime_result = _coerce_datetime_arithmetic(op, left_dt, right_dt)
    if datetime_result is not None:
        return datetime_result

    not_implemented_yet(f"arithmetic coercion for {type(left_dt).__name__} {op} {type(right_dt).__name__}")


_DATE_OFFSET_TYPES = (ByteType, ShortType, IntegerType)


def _coerce_datetime_arithmetic(
    op: str, left_dt: Any, right_dt: Any
) -> Optional[Tuple[Any, Optional[Any], Optional[Any]]]:
    """Spark's date/timestamp arithmetic, or ``None`` when ``left <op> right`` is not one of its forms.

    - ``date + n`` / ``n + date`` / ``date - n`` (``n`` a byte/short/int) is ``date_add`` / ``date_sub``.
    - ``date - date`` is an ``interval day``.
    - ``timestamp - timestamp`` is an ``interval day to second``; a date operand widens to a timestamp.
    """
    if op == "+" and (
        (isinstance(left_dt, DateType) and isinstance(right_dt, _DATE_OFFSET_TYPES))
        or (isinstance(left_dt, _DATE_OFFSET_TYPES) and isinstance(right_dt, DateType))
    ):
        return DateType(), None, None
    if op != "-":
        return None
    if isinstance(left_dt, DateType) and isinstance(right_dt, _DATE_OFFSET_TYPES):
        return DateType(), None, None
    if isinstance(left_dt, DateType) and isinstance(right_dt, DateType):
        return DayTimeIntervalType(DayTimeIntervalType.DAY, DayTimeIntervalType.DAY), None, None
    if isinstance(left_dt, (DateType, TimestampType)) and isinstance(right_dt, (DateType, TimestampType)):
        result = DayTimeIntervalType(DayTimeIntervalType.DAY, DayTimeIntervalType.SECOND)
        return result, _cast_unless(left_dt, TimestampType()), _cast_unless(right_dt, TimestampType())
    return None


def _cast_unless(dtype: Any, target: Any) -> Optional[Any]:
    """``target`` if ``dtype`` must be cast to reach it, else ``None``."""
    return None if dtype == target else target


MAX_DECIMAL_PRECISION = 38
_MIN_ADJUSTED_SCALE = 6


def _bounded_decimal(precision: int, scale: int) -> Any:
    """Spark's ``adjustPrecisionScale`` (precision loss allowed): keep the integral digits, give up scale."""
    if precision <= MAX_DECIMAL_PRECISION:
        return DecimalType(precision, scale)
    int_digits = precision - scale
    min_scale = min(scale, _MIN_ADJUSTED_SCALE)
    adjusted_scale = max(MAX_DECIMAL_PRECISION - int_digits, min_scale)
    return DecimalType(MAX_DECIMAL_PRECISION, adjusted_scale)


def decimal_arithmetic_type(op: str, left: Any, right: Any) -> Any:
    """Spark's ``DecimalPrecision`` result type of ``decimal(p1,s1) <op> decimal(p2,s2)``."""
    p1, s1, p2, s2 = left.precision, left.scale, right.precision, right.scale
    if op in ("+", "-"):
        scale = max(s1, s2)
        return _bounded_decimal(max(p1 - s1, p2 - s2) + scale + 1, scale)
    if op == "*":
        return _bounded_decimal(p1 + p2 + 1, s1 + s2)
    scale = max(_MIN_ADJUSTED_SCALE, s1 + p2 + 1)
    return _bounded_decimal(p1 - s1 + s2 + scale, scale)


def literal_decimal_type(value: int) -> Any:
    """The decimal Spark narrows an integral literal to inside decimal arithmetic (``100 -> decimal(3,0)``)."""
    return DecimalType(max(len(str(abs(value))), 1), 0)


def find_wider_type(left: Any, right: Any) -> Optional[Any]:
    """Spark's wider common type of two types (``union``, ``coalesce``, ``when`` branches, ``greatest``...).

    ``None`` when the types have no common type (callers raise Spark's ``DATATYPE_MISMATCH``).
    """
    if left == right:
        return left
    if isinstance(left, NullType):
        return right
    if isinstance(right, NullType):
        return left
    numeric = (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType)
    if isinstance(left, numeric) and isinstance(right, numeric):
        if isinstance(left, (FloatType, DoubleType)) or isinstance(right, (FloatType, DoubleType)):
            return widen_numeric(*(DoubleType() if isinstance(t, DecimalType) else t for t in (left, right)))
        if isinstance(left, DecimalType) or isinstance(right, DecimalType):
            return _wider_decimal(_as_decimal(left), _as_decimal(right))
        return widen_numeric(left, right)
    if {type(left), type(right)} == {DateType, TimestampType}:
        return TimestampType()
    if is_string(left) != is_string(right):
        return _wider_type_for_string(right if is_string(left) else left)
    if isinstance(left, ArrayType) and isinstance(right, ArrayType):
        element = find_wider_type(left.elementType, right.elementType)
        return None if element is None else ArrayType(element, left.containsNull or right.containsNull)
    if isinstance(left, MapType) and isinstance(right, MapType):
        key = find_wider_type(left.keyType, right.keyType)
        value = find_wider_type(left.valueType, right.valueType)
        if key is None or value is None:
            return None
        return MapType(key, value, left.valueContainsNull or right.valueContainsNull)
    if isinstance(left, StructType) and isinstance(right, StructType):
        if [f.name for f in left.fields] != [f.name for f in right.fields]:
            return None
        fields = []
        for lf, rf in zip(left.fields, right.fields):
            wider = find_wider_type(lf.dataType, rf.dataType)
            if wider is None:
                return None
            fields.append(StructField(lf.name, wider, lf.nullable or rf.nullable))
        return StructType(fields)
    return None


def _wider_type_for_string(other: Any) -> Optional[Any]:
    """Spark 4 ANSI ``findWiderTypeForString``: the type a string widens to next to ``other``.

    ``bigint`` next to an integral, ``double`` next to a fractional (decimal included), the other
    type next to any other atomic type; complex types have no common type with a string. The
    string side is then cast strictly, so a malformed value raises ``CAST_INVALID_INPUT``.
    """
    if _is_integral(other):
        return LongType()
    if isinstance(other, (FloatType, DoubleType, DecimalType)):
        return DoubleType()
    if isinstance(other, (BooleanType, BinaryType, DateType, TimestampType)):
        return other
    return None


def find_wider_common_type(types: list) -> Optional[Any]:
    """Fold :func:`find_wider_type` over ``types`` (``None`` if any pair has no common type)."""
    result: Any = NullType()
    for dtype in types:
        result = find_wider_type(result, dtype)
        if result is None:
            return None
    return result


def coerce_pow(left_dt: Any, right_dt: Any) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Resolve the result type (and operand casts) of ``pow(left, right)`` / ``left ** right``.

    Spark's ``Pow`` declares ``(double, double) -> double`` input types, so each operand is
    implicitly cast to ``double``: numerics, decimals and nulls always, strings via an ANSI cast
    (malformed strings raise at evaluate time). Any other type (boolean, date, binary, nested)
    is a ``DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`` analysis error in Spark, raised here as
    :class:`TypeError`.

    Returns ``(DoubleType(), left_cast_dt, right_cast_dt)`` where a ``*_cast_dt`` is ``None`` if
    the operand is already ``double``.
    """
    casts = []
    for position, dtype in (("first", left_dt), ("second", right_dt)):
        if not isinstance(
            dtype,
            (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType, StringType, NullType),
        ):
            raise TypeError(
                f"[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve POWER: the {position} parameter "
                f'requires the "DOUBLE" type, however it has the type "{dtype.simpleString().upper()}".'
            )
        casts.append(None if isinstance(dtype, DoubleType) else DoubleType())
    return DoubleType(), casts[0], casts[1]


_INTEGRAL_DECIMAL_PRECISION = {"ByteType": 3, "ShortType": 5, "IntegerType": 10, "LongType": 20}


def is_orderable(dtype: Any) -> bool:
    """Whether Spark can compare two values of ``dtype`` (maps never; arrays/structs if their children can)."""
    if isinstance(dtype, ArrayType):
        return is_orderable(dtype.elementType)
    if isinstance(dtype, StructType):
        return all(is_orderable(field.dataType) for field in dtype.fields)
    return isinstance(
        dtype,
        (
            ByteType,
            ShortType,
            IntegerType,
            LongType,
            FloatType,
            DoubleType,
            DecimalType,
            StringType,
            BooleanType,
            DateType,
            TimestampType,
            BinaryType,
            NullType,
        ),
    )


def contains_map(dtype: Any) -> bool:
    if isinstance(dtype, MapType):
        return True
    if isinstance(dtype, ArrayType):
        return contains_map(dtype.elementType)
    if isinstance(dtype, StructType):
        return any(contains_map(field.dataType) for field in dtype.fields)
    return False


def _as_decimal(dtype: Any) -> Any:
    """An integral type as the narrowest decimal holding it (``int -> decimal(10,0)``)."""
    if isinstance(dtype, DecimalType):
        return dtype
    return DecimalType(_INTEGRAL_DECIMAL_PRECISION[type(dtype).__name__], 0)


def _wider_decimal(left: Any, right: Any) -> Any:
    """Spark's ``widerDecimalType``: keep the larger integer range and the larger scale (max 38)."""
    scale = max(left.scale, right.scale)
    integer_range = max(left.precision - left.scale, right.precision - right.scale)
    return DecimalType(min(integer_range + scale, 38), scale)


def _comparison_target(left_dt: Any, right_dt: Any) -> Optional[Any]:
    """The common type Spark 4 (ANSI) coerces two *different* comparison operand types to, or ``None``.

    - numeric × numeric: a float/double on either side → ``double``; integral × integral widens;
      integral × decimal → the wider decimal.
    - string × integral → ``bigint``; string × float/double/decimal → ``double``; string × boolean /
      date / timestamp / binary → the string is cast to that type (strictly, at evaluate time).
    - date × timestamp → ``timestamp``.
    """
    numeric = (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType)
    if isinstance(left_dt, numeric) and isinstance(right_dt, numeric):
        if isinstance(left_dt, (FloatType, DoubleType)) or isinstance(right_dt, (FloatType, DoubleType)):
            return DoubleType()
        if isinstance(left_dt, DecimalType) or isinstance(right_dt, DecimalType):
            return _wider_decimal(_as_decimal(left_dt), _as_decimal(right_dt))
        return widen_numeric(left_dt, right_dt)
    for string_dt, other_dt in ((left_dt, right_dt), (right_dt, left_dt)):
        if isinstance(string_dt, StringType):
            if _is_integral(other_dt):
                return LongType()
            if isinstance(other_dt, (FloatType, DoubleType, DecimalType)):
                return DoubleType()
            if isinstance(other_dt, (BooleanType, DateType, TimestampType, BinaryType)):
                return other_dt
    if {type(left_dt), type(right_dt)} == {DateType, TimestampType}:
        return TimestampType()
    return None


def coerce_comparison(left_dt: Any, right_dt: Any) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Resolve ``left <cmp> right`` (``== != < <= > >=``) to ``(BooleanType(), left_cast, right_cast)``.

    Operands are coerced to one common type (see :func:`_comparison_target`); a ``null`` operand is
    cast to the other side's type. Spark rejects maps (``INVALID_ORDERING_TYPE``) and unrelated types
    (``BINARY_OP_DIFF_TYPES``) at analysis time, raised here as :class:`TypeError`.
    """
    if contains_map(left_dt) or contains_map(right_dt):
        raise TypeError(
            f"[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] Cannot compare values of type "
            f'"{left_dt.simpleString().upper()}" and "{right_dt.simpleString().upper()}".'
        )
    if not (is_orderable(left_dt) and is_orderable(right_dt)):
        not_implemented_yet(f"comparison of {left_dt.simpleString()} and {right_dt.simpleString()}")
    if left_dt == right_dt:
        return BooleanType(), None, None
    if isinstance(left_dt, NullType):
        return BooleanType(), right_dt, None
    if isinstance(right_dt, NullType):
        return BooleanType(), None, left_dt

    target = _comparison_target(left_dt, right_dt)
    if target is None:
        if isinstance(left_dt, (ArrayType, StructType)) and isinstance(right_dt, type(left_dt)):
            # Spark widens e.g. array<int> vs array<bigint> element-wise; not modeled yet.
            not_implemented_yet(f"comparison of {left_dt.simpleString()} and {right_dt.simpleString()}")
        raise TypeError(
            f"[DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES] Cannot compare values of differing types "
            f'"{left_dt.simpleString().upper()}" and "{right_dt.simpleString().upper()}".'
        )
    return (
        BooleanType(),
        None if left_dt == target else target,
        None if right_dt == target else target,
    )


def _boolean_operand_cast(dtype: Any, operator: str) -> Optional[Any]:
    """The cast (if any) an ``and`` / ``or`` / ``not`` operand needs: Spark requires ``boolean``.

    A string is cast to boolean (strictly, at evaluate time) and a ``null`` is typed as boolean;
    any other type is a ``DATATYPE_MISMATCH`` analysis error, raised here as :class:`TypeError`.
    """
    if isinstance(dtype, BooleanType):
        return None
    if isinstance(dtype, (StringType, NullType)):
        return BooleanType()
    raise TypeError(
        f"[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve {operator.upper()}: it requires the "
        f'"BOOLEAN" type, however an operand has the type "{dtype.simpleString().upper()}".'
    )


def coerce_logical(op: str, left_dt: Any, right_dt: Any) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Resolve ``left and|or right`` to ``(BooleanType(), left_cast, right_cast)``."""
    return BooleanType(), _boolean_operand_cast(left_dt, op), _boolean_operand_cast(right_dt, op)


def coerce_not(dtype: Any) -> Tuple[Any, Optional[Any]]:
    """Resolve ``not child`` to ``(BooleanType(), child_cast)``."""
    return BooleanType(), _boolean_operand_cast(dtype, "not")
