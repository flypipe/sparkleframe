"""Spark type-coercion rules for the **analyze** phase, written over PySpark types.

These are pure functions: given the resolved ``data_type`` of an operand (a PySpark
``DataType``), they decide the result type of an operation and where Spark would insert an
implicit cast. They are a clean rewrite of the coercion behavior Spark documents — they do
**not** port ``polarsdf`` internals (per #104 Q4).

The ``pyspark.sql.types`` import is guarded exactly like
:mod:`sparkleframe.python.types`: under ``activate()`` the real ``pyspark`` package is replaced
by a mock that has no real types, but these functions only run at analyze time (under real
pyspark), so referencing the type classes inside them is safe.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional, Tuple

from sparkleframe.python._errors import not_implemented_yet

try:  # pragma: no cover - exercised only with the real pyspark installed
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
        StructType,
        TimestampType,
    )
except Exception:  # pragma: no cover - mock pyspark (under activate) has no real types
    pass


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
    """Resolve the result type (and any operand casts) of ``left <op> right``.

    Returns ``(result_dt, left_cast_dt, right_cast_dt)`` where a ``*_cast_dt`` is the type the
    corresponding operand must be cast to (or ``None`` if no cast is needed). Only ``+ - *`` are
    handled here — ``**`` resolves through :func:`coerce_pow`, and ``/`` is out of scope and raises.
    """
    if op not in ("+", "-", "*"):
        not_implemented_yet(f"arithmetic result type for operator {op!r}")

    if is_numeric(left_dt) and is_numeric(right_dt):
        result = widen_numeric(left_dt, right_dt)
        left_cast = None if type(left_dt) is type(result) else result
        right_cast = None if type(right_dt) is type(result) else result
        return result, left_cast, right_cast

    # numeric + string: Spark promotes the string to the numeric operand's arithmetic type —
    # ``long`` when the numeric is integral (``int + "3" -> long``) and ``double`` when it is
    # floating (``double + "3.14" -> double``). The numeric operand widens to that type too.
    if is_string(left_dt) and is_numeric(right_dt):
        result = _string_arith_promote_type(right_dt)
        return result, result, (None if type(right_dt) is type(result) else result)
    if is_numeric(left_dt) and is_string(right_dt):
        result = _string_arith_promote_type(left_dt)
        return result, (None if type(left_dt) is type(result) else result), result

    not_implemented_yet(f"arithmetic coercion for {type(left_dt).__name__} {op} {type(right_dt).__name__}")


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


def _is_orderable(dtype: Any) -> bool:
    """Whether Spark can compare two values of ``dtype`` (maps never; arrays/structs if their children can)."""
    if isinstance(dtype, ArrayType):
        return _is_orderable(dtype.elementType)
    if isinstance(dtype, StructType):
        return all(_is_orderable(field.dataType) for field in dtype.fields)
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


def _contains_map(dtype: Any) -> bool:
    if isinstance(dtype, MapType):
        return True
    if isinstance(dtype, ArrayType):
        return _contains_map(dtype.elementType)
    if isinstance(dtype, StructType):
        return any(_contains_map(field.dataType) for field in dtype.fields)
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
    if _contains_map(left_dt) or _contains_map(right_dt):
        raise TypeError(
            f"[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] Cannot compare values of type "
            f'"{left_dt.simpleString().upper()}" and "{right_dt.simpleString().upper()}".'
        )
    if not (_is_orderable(left_dt) and _is_orderable(right_dt)):
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
