"""Shared, private helpers for ``column.py`` (pure-Python engine).

Per the repo convention (CLAUDE.md), shared/private logic for the public ``column`` surface
lives here rather than in ``column.py``. These are the per-value routines the evaluate phase
applies for Column operators:

- :func:`cast_value` — Spark 4 casts with a ``strict`` flag (ANSI ``cast`` raises
  ``CAST_INVALID_INPUT`` on malformed input, ``try_cast`` returns null), sharing one parser per
  target type so the strict API never delegates to its lenient sibling.
- :func:`compare_values` — Spark's comparison operators, including its total ordering over
  NaN and nulls nested inside arrays/structs.
- :func:`arithmetic` — ``+ - * /`` with ANSI overflow / divide-by-zero errors and exact decimal
  math rounded to the analyzer's result type.
- :func:`to_spark_string` — values formatted the way ``cast(... as string)`` does.
- :func:`logical_and` / :func:`logical_or` / :func:`logical_not` — SQL three-valued logic.
"""

from __future__ import annotations

import math
import re
import struct
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy

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
    Row,
    ShortType,
    StringType,
    StructType,
    TimestampType,
)

# --------------------------------------------------------------------------- #
# Casting
# --------------------------------------------------------------------------- #


class _Malformed(Exception):
    """Internal signal from a parser: the input is malformed for the target type."""


def cast_value(value: Any, target_type: Any, strict: bool, source_type: Any = None) -> Any:
    """Cast one Python value to ``target_type`` (an engine ``DataType``) with Spark 4 semantics.

    A null stays null. Malformed input raises ``CAST_INVALID_INPUT`` and an out-of-range value
    raises ``CAST_OVERFLOW`` / ``NUMERIC_VALUE_OUT_OF_RANGE`` when ``strict`` (ANSI ``cast``); both
    become null otherwise (``try_cast``). ``source_type`` is the value's resolved type, needed where
    the Python value alone is ambiguous (e.g. formatting a ``float`` column as a string).
    """
    if value is None:
        return None
    try:
        return _cast(value, target_type, source_type)
    except _Malformed:
        if not strict:
            return None
        source = source_type.simpleString().upper() if source_type is not None else "STRING"
        raise ValueError(
            f"[CAST_INVALID_INPUT] The value '{value}' of the type \"{source}\" cannot be cast to "
            f'"{target_type.simpleString().upper()}" because it is malformed. Correct the value as per '
            f"the syntax, or change its target type. Use `try_cast` to tolerate malformed input and "
            f"return NULL instead."
        ) from None
    except _Overflow:
        if not strict:
            return None
        if isinstance(target_type, DecimalType):
            raise ArithmeticError(
                f"[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION] {value} cannot be represented as "
                f"Decimal({target_type.precision}, {target_type.scale}). If necessary set "
                f'"spark.sql.ansi.enabled" to "false" to bypass this error, and return NULL instead.'
            ) from None
        raise ArithmeticError(
            f'[CAST_OVERFLOW] The value {value} cannot be cast to "{target_type.simpleString().upper()}" '
            f"due to an overflow. Use `try_cast` to tolerate overflow and return NULL instead."
        ) from None


class _Overflow(Exception):
    """Internal signal: the value does not fit the target type's range."""


def _cast(value: Any, target_type: Any, source_type: Any) -> Any:
    if isinstance(target_type, StringType):
        return to_spark_string(value, source_type)
    if isinstance(value, str):
        return _cast_string(value, target_type)
    if isinstance(value, datetime) and isinstance(target_type, _NUMERIC_TYPES):
        return _timestamp_to_number(value, target_type)
    if isinstance(target_type, TimestampType) and isinstance(value, (int, float, Decimal)):
        return _seconds_to_timestamp(value)
    if isinstance(target_type, BooleanType):
        return value if isinstance(value, bool) else value != 0
    if isinstance(target_type, (ByteType, ShortType, IntegerType, LongType)):
        return _to_integral(value, target_type)
    if isinstance(target_type, (FloatType, DoubleType)):
        result = float(value)
        return _round_to_float32(result) if isinstance(target_type, FloatType) else result
    if isinstance(target_type, DecimalType):
        if isinstance(value, float) and not math.isfinite(value):
            return None  # Spark's castToDecimal turns the NumberFormatException into null, ANSI or not
        return to_decimal(value, target_type)
    if isinstance(target_type, DateType) and isinstance(value, date):
        return value.date() if isinstance(value, datetime) else value
    if isinstance(target_type, TimestampType) and isinstance(value, date):
        return value if isinstance(value, datetime) else datetime.combine(value, time())
    if isinstance(target_type, ArrayType) and isinstance(value, list):
        element_source = getattr(source_type, "elementType", None)
        return [None if v is None else _cast(v, target_type.elementType, element_source) for v in value]
    if isinstance(target_type, MapType) and isinstance(value, dict):
        key_source, value_source = getattr(source_type, "keyType", None), getattr(source_type, "valueType", None)
        return {
            _cast(k, target_type.keyType, key_source): (
                None if v is None else _cast(v, target_type.valueType, value_source)
            )
            for k, v in value.items()
        }
    if isinstance(target_type, StructType) and isinstance(value, tuple):
        sources = (
            [f.dataType for f in source_type.fields] if isinstance(source_type, StructType) else [None] * len(value)
        )
        return Row.from_fields(
            [f.name for f in target_type.fields],
            [
                None if v is None else _cast(v, f.dataType, src)
                for v, f, src in zip(value, target_type.fields, sources)
            ],
        )
    not_implemented_yet(f"casting {type(value).__name__} to {type(target_type).__name__}")


_NUMERIC_TYPES = (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICROS = 1_000_000


def _seconds_to_timestamp(seconds: Any) -> datetime:
    """Spark's numeric -> timestamp: seconds since the epoch (fraction truncated to micros), session-local."""
    if isinstance(seconds, float) and not math.isfinite(seconds):
        raise _Malformed
    micros = int(Decimal(repr(seconds) if isinstance(seconds, float) else seconds) * _MICROS)
    try:
        return (_EPOCH + timedelta(microseconds=micros)).astimezone().replace(tzinfo=None)
    except OverflowError:
        raise _Overflow from None


def _timestamp_to_number(value: datetime, target_type: Any) -> Any:
    """Spark's timestamp -> numeric: whole seconds (floored) for integrals, fractional seconds otherwise."""
    delta = value.astimezone() - _EPOCH
    micros = (delta.days * 86_400 + delta.seconds) * _MICROS + delta.microseconds
    if isinstance(target_type, (ByteType, ShortType, IntegerType, LongType)):
        return _to_integral(micros // _MICROS, target_type)
    seconds = micros / _MICROS
    if isinstance(target_type, DecimalType):
        return to_decimal(seconds, target_type)  # Spark goes through the double, too
    return _round_to_float32(seconds) if isinstance(target_type, FloatType) else seconds


def _to_integral(value: Any, target_type: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise _Overflow
    result = int(value)  # truncates toward zero for float and Decimal, like Spark
    low, high = _INTEGRAL_RANGE[type(target_type)]
    if not low <= result <= high:
        raise _Overflow
    return result


_DECIMAL_CONTEXT = Context(prec=120, rounding=ROUND_HALF_UP)


def to_decimal(value: Any, target_type: Any) -> Decimal:
    """Round ``value`` HALF_UP to ``target_type``'s scale; raise ``_Overflow`` if it exceeds the precision."""
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise _Overflow
        value = Decimal(repr(value))
    exact = value if isinstance(value, Decimal) else Decimal(value)
    quantum = Decimal(1).scaleb(-target_type.scale)
    result = exact.quantize(quantum, context=_DECIMAL_CONTEXT)
    if abs(result) >= Decimal(10) ** (target_type.precision - target_type.scale):
        raise _Overflow
    return result.copy_abs() if result.is_zero() else result  # Java's BigDecimal has no negative zero


def _round_to_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0] if math.isfinite(value) else value


def _cast_string(value: str, target_type: Any) -> Any:
    if isinstance(target_type, (ByteType, ShortType, IntegerType, LongType)):
        return _parse_integral(value, target_type)
    if isinstance(target_type, (FloatType, DoubleType)):
        result = _parse_double(value)
        return _round_to_float32(result) if isinstance(target_type, FloatType) else result
    if isinstance(target_type, DecimalType):
        return to_decimal(_parse_decimal(value), target_type)
    if isinstance(target_type, BooleanType):
        return _parse_boolean(value)
    if isinstance(target_type, DateType):
        return _parse_date(value)
    if isinstance(target_type, TimestampType):
        return _parse_timestamp(value)
    if isinstance(target_type, BinaryType):
        return value.encode("utf-8")
    not_implemented_yet(f"casting string to {type(target_type).__name__}")


_DECIMAL_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _parse_decimal(value: str) -> Decimal:
    """Java ``BigDecimal`` syntax (optional sign, fraction, exponent) after Spark's whitespace trim."""
    text = _trim(value)
    if not _DECIMAL_RE.fullmatch(text):
        raise _Malformed
    return Decimal(text)


# --------------------------------------------------------------------------- #
# Values as Spark strings
# --------------------------------------------------------------------------- #


def to_spark_string(value: Any, source_type: Any = None) -> str:
    """Format a non-null value the way Spark's ``cast(... as string)`` does."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return java_float_string(value, single=isinstance(source_type, FloatType))
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return format_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, list):
        element_type = getattr(source_type, "elementType", None)
        return "[" + ", ".join(_nested_string(v, element_type) for v in value) + "]"
    if isinstance(value, dict):
        key_type, value_type = getattr(source_type, "keyType", None), getattr(source_type, "valueType", None)
        items = (f"{_nested_string(k, key_type)} -> {_nested_string(v, value_type)}" for k, v in value.items())
        return "{" + ", ".join(items) + "}"
    if isinstance(value, tuple):
        field_types = [f.dataType for f in source_type.fields] if isinstance(source_type, StructType) else []
        field_types += [None] * (len(value) - len(field_types))
        return "{" + ", ".join(_nested_string(v, t) for v, t in zip(value, field_types)) + "}"
    not_implemented_yet(f"casting {type(value).__name__} to string")


def _nested_string(value: Any, dtype: Any) -> str:
    return "null" if value is None else to_spark_string(value, dtype)


def format_timestamp(value: datetime) -> str:
    """``yyyy-MM-dd HH:mm:ss`` plus the fraction with trailing zeros dropped (Spark's timestamp string)."""
    text = f"{value.year:04d}-" + value.strftime("%m-%d %H:%M:%S")  # strftime doesn't pad years < 1000
    if value.microsecond:
        text += "." + f"{value.microsecond:06d}".rstrip("0")
    return text


def java_float_string(value: float, single: bool = False) -> str:
    """Java ``Double.toString`` / ``Float.toString``: plain in ``[1e-3, 1e7)``, else ``d.dddE±n``."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    shortest = str(numpy.float32(value)) if single else repr(value)
    sign, digits, exponent = Decimal(shortest).normalize().as_tuple()
    text_digits = "".join(map(str, digits))
    sci_exponent = exponent + len(text_digits) - 1
    prefix = "-" if sign else ""
    if 1e-3 <= abs(value) < 1e7:
        if exponent >= 0:
            return f"{prefix}{text_digits}{'0' * exponent}.0"
        point = len(text_digits) + exponent
        if point > 0:
            return f"{prefix}{text_digits[:point]}.{text_digits[point:]}"
        return f"{prefix}0.{'0' * -point}{text_digits}"
    return f"{prefix}{text_digits[0]}.{text_digits[1:] or '0'}E{sci_exponent}"


_SPARK_WHITESPACE = "".join(chr(c) for c in range(33))


def _trim(value: str) -> str:
    """Spark trims ASCII whitespace and control characters (``<= ' '``) before parsing."""
    return value.strip(_SPARK_WHITESPACE)


_INTEGRAL_RANGE = {
    ByteType: (-(2**7), 2**7 - 1),
    ShortType: (-(2**15), 2**15 - 1),
    IntegerType: (-(2**31), 2**31 - 1),
    LongType: (-(2**63), 2**63 - 1),
}
_INTEGRAL_RE = re.compile(r"[+-]?[0-9]+")


def _parse_integral(value: str, target_type: Any) -> int:
    """Plain (optionally signed) ASCII digits within the target range — no ``1.0``, ``1e2`` or ``1_0``."""
    text = _trim(value)
    if not _INTEGRAL_RE.fullmatch(text):
        raise _Malformed
    result = int(text)
    low, high = _INTEGRAL_RANGE[type(target_type)]
    if not low <= result <= high:
        raise _Malformed
    return result


_POS_INFINITY = {"inf", "+inf", "infinity", "+infinity"}
_NEG_INFINITY = {"-inf", "-infinity"}
_DECIMAL_FLOAT_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[dDfF]?")
_HEX_FLOAT_RE = re.compile(r"[+-]?0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)[pP][+-]?[0-9]+[dDfF]?")


def _parse_double(value: str) -> float:
    """Java ``Double.parseDouble`` syntax (incl. ``1d``/``1f`` suffixes, hex) plus Spark's special literals."""
    text = _trim(value)
    lowered = text.lower()
    if lowered in _POS_INFINITY:
        return math.inf
    if lowered in _NEG_INFINITY:
        return -math.inf
    if lowered == "nan":
        return math.nan
    if _DECIMAL_FLOAT_RE.fullmatch(text):
        return float(text.rstrip("dDfF"))
    if _HEX_FLOAT_RE.fullmatch(text):
        return float.fromhex(text.rstrip("dDfF"))
    raise _Malformed


_TRUE_STRINGS = {"t", "true", "y", "yes", "1"}
_FALSE_STRINGS = {"f", "false", "n", "no", "0"}


def _parse_boolean(value: str) -> bool:
    lowered = _trim(value).lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    raise _Malformed


# ``yyyy[-[m]m[-[d]d]]`` with an optional sign and 4+ year digits; for a date anything may
# follow the day after a ' ' or 'T' separator.
_DATE_RE = re.compile(r"(?P<y>[+-]?[0-9]{4,})(?:-(?P<m>[0-9]{1,2})(?:-(?P<d>[0-9]{1,2})(?:[ T].*)?)?)?", re.S)
_TIMESTAMP_RE = re.compile(
    r"(?P<y>[+-]?[0-9]{4,})(?:-(?P<m>[0-9]{1,2})(?:-(?P<d>[0-9]{1,2})"
    r"(?:[ T](?P<H>[0-9]{1,2})(?::(?P<M>[0-9]{1,2})(?::(?P<S>[0-9]{1,2})(?:\.(?P<f>[0-9]+))?)?)?)?)?)?"
    r"(?P<zone>.*)",
    re.S,
)
_TIME_ONLY_RE = re.compile(r"T?[0-9]{1,2}:[0-9]{1,2}.*", re.S)
_OFFSET_ZONE_RE = re.compile(
    r"Z|(?:UTC|GMT|UT)?(?P<sign>[+-])(?P<h>[0-9]{1,2})(?::?(?P<m>[0-9]{2})(?::?(?P<s>[0-9]{2}))?)?"
)


def _build_date(year: str, month: Optional[str], day: Optional[str]) -> date:
    try:
        return date(int(year), int(month or 1), int(day or 1))
    except (ValueError, OverflowError):
        raise _Malformed from None


def _parse_date(value: str) -> date:
    match = _DATE_RE.fullmatch(_trim(value))
    if match is None:
        raise _Malformed
    return _build_date(match["y"], match["m"], match["d"])


def _parse_timestamp(value: str) -> datetime:
    """``yyyy[-mm[-dd[ hh[:mm[:ss[.ffffff]]]]]]`` plus an optional zone; fractions beyond micros are truncated.

    Timestamps are held as naive datetimes in the session time zone, which (as Spark's default, the
    JVM zone) is the process-local zone. A zone suffix (``Z``, ``+02:00``, ``UTC``, a region id) is
    converted into it. A time-only string (which takes the current date) is not modelled yet.
    """
    text = _trim(value)
    if _TIME_ONLY_RE.fullmatch(text):
        not_implemented_yet("casting a time-only string to timestamp")
    match = _TIMESTAMP_RE.fullmatch(text)
    if match is None:
        raise _Malformed
    zone_text = match["zone"].strip()
    zone = _zone(zone_text) if zone_text else None
    if zone_text and zone is None:
        raise _Malformed
    day = _build_date(match["y"], match["m"], match["d"])
    hour, minute, second = int(match["H"] or 0), int(match["M"] or 0), int(match["S"] or 0)
    if hour > 23 or minute > 59 or second > 59:
        raise _Malformed
    micros = int((match["f"] or "0")[:6].ljust(6, "0"))
    parsed = datetime(day.year, day.month, day.day, hour, minute, second, micros)
    if zone is None:
        return parsed
    return parsed.replace(tzinfo=zone).astimezone().replace(tzinfo=None)


def _zone(text: str) -> Optional[tzinfo]:
    """A zone suffix as ``tzinfo``: ``Z``, ``[UTC|GMT|UT]±h[h][[:]mm[[:]ss]]``, or a region id; else ``None``."""
    match = _OFFSET_ZONE_RE.fullmatch(text)
    if match is not None:
        if match["sign"] is None:
            return timezone.utc
        hours, minutes, seconds = int(match["h"]), int(match["m"] or 0), int(match["s"] or 0)
        if hours > 18 or minutes > 59 or seconds > 59:
            return None
        offset = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return timezone(-offset if match["sign"] == "-" else offset)
    if text in ("UTC", "GMT", "UT"):
        return timezone.utc
    try:
        return ZoneInfo(text)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def arithmetic(op: str, left: Any, right: Any, result_type: Any, strict: bool = True) -> Any:
    """Apply ``+ - * /`` to two operands already cast by the analyzer; any null operand yields null.

    Decimal results are computed exactly then rounded HALF_UP to ``result_type``. Under ANSI
    (``strict``) a zero divisor raises ``DIVIDE_BY_ZERO`` and an out-of-range result raises
    ``ARITHMETIC_OVERFLOW`` / ``NUMERIC_VALUE_OUT_OF_RANGE``; the lenient form (``try_divide``)
    returns null for both.
    """
    if left is None or right is None:
        return None
    if op == "/" and right == 0:
        if not strict:
            return None
        raise ZeroDivisionError(
            "[DIVIDE_BY_ZERO] Division by zero. Use `try_divide` to tolerate divisor being 0 and return NULL "
            'instead. If necessary set "spark.sql.ansi.enabled" to "false" to bypass this error.'
        )
    if isinstance(result_type, DateType):
        days = timedelta(days=right if isinstance(left, date) else left)
        start = left if isinstance(left, date) else right
        return start + days if op == "+" else start - days
    if isinstance(result_type, DayTimeIntervalType):
        return left - right
    if isinstance(result_type, DecimalType):
        exact = _DECIMAL_OPS[op](Decimal(left), Decimal(right), _DECIMAL_CONTEXT)
        try:
            return to_decimal(exact, result_type)
        except _Overflow:
            if not strict:
                return None
            raise ArithmeticError(
                f"[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION] {exact} cannot be represented as "
                f"Decimal({result_type.precision}, {result_type.scale})."
            ) from None
    result = _FLOAT_OPS[op](left, right)
    if isinstance(result_type, (ByteType, ShortType, IntegerType, LongType)):
        low, high = _INTEGRAL_RANGE[type(result_type)]
        if not low <= result <= high:
            if not strict:
                return None
            raise ArithmeticError(
                f"[ARITHMETIC_OVERFLOW] {result_type.simpleString()} overflow. Use 'try_{_OP_NAMES[op]}' to "
                f"tolerate overflow and return NULL instead."
            )
    if isinstance(result_type, FloatType):
        return _round_to_float32(result)
    return result


_DECIMAL_OPS = {
    "+": lambda a, b, ctx: ctx.add(a, b),
    "-": lambda a, b, ctx: ctx.subtract(a, b),
    "*": lambda a, b, ctx: ctx.multiply(a, b),
    "/": lambda a, b, ctx: ctx.divide(a, b),
}
_FLOAT_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}
_OP_NAMES = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def compare_values(op: str, left: Any, right: Any, data_type: Any) -> Optional[bool]:
    """Apply comparison ``op`` to two values already coerced to the common ``data_type``.

    A null operand yields null. Otherwise Spark's ordering applies: NaN equals NaN and sorts
    above every other number, and arrays/structs compare element-wise with null elements first.
    """
    if left is None or right is None:
        return None
    order = spark_order(left, right, data_type)
    if op == "==":
        return order == 0
    if op == "!=":
        return order != 0
    if op == "<":
        return order < 0
    if op == "<=":
        return order <= 0
    if op == ">":
        return order > 0
    return order >= 0


def spark_order(left: Any, right: Any, data_type: Any) -> int:
    """Three-way compare of two non-null values of ``data_type`` (-1, 0, 1)."""
    if isinstance(data_type, (FloatType, DoubleType)):
        left_nan, right_nan = math.isnan(left), math.isnan(right)
        if left_nan or right_nan:
            return int(left_nan) - int(right_nan)
    if isinstance(data_type, ArrayType):
        return _sequence_order(list(left), list(right), [data_type.elementType] * max(len(left), len(right)))
    if isinstance(data_type, StructType):
        field_types = [field.dataType for field in data_type.fields]
        return _sequence_order(_struct_values(left, data_type), _struct_values(right, data_type), field_types)
    return (left > right) - (left < right)


def _sequence_order(left: list, right: list, element_types: list) -> int:
    """Element-wise order (null elements sort first), then the shorter sequence first."""
    for left_item, right_item, element_type in zip(left, right, element_types):
        if left_item is None or right_item is None:
            if left_item is None and right_item is None:
                continue
            return -1 if left_item is None else 1
        order = spark_order(left_item, right_item, element_type)
        if order != 0:
            return order
    return (len(left) > len(right)) - (len(left) < len(right))


def _struct_values(value: Any, data_type: Any) -> list:
    """A struct cell as its field values in schema order (accepts a dict or a tuple/Row)."""
    if isinstance(value, dict):
        return [value.get(field.name) for field in data_type.fields]
    return list(value)


# --------------------------------------------------------------------------- #
# Three-valued logic
# --------------------------------------------------------------------------- #


def logical_and(left: Optional[bool], right: Optional[bool]) -> Optional[bool]:
    """SQL ``AND``: false wins over null (``null AND false`` is false)."""
    if left is False or right is False:
        return False
    if left is None or right is None:
        return None
    return True


def logical_or(left: Optional[bool], right: Optional[bool]) -> Optional[bool]:
    """SQL ``OR``: true wins over null (``null OR true`` is true)."""
    if left is True or right is True:
        return True
    if left is None or right is None:
        return None
    return False


def logical_not(value: Optional[bool]) -> Optional[bool]:
    return None if value is None else not value
