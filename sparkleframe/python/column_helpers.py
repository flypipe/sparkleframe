"""Shared, private helpers for ``column.py`` (pure-Python engine).

Per the repo convention (CLAUDE.md), shared/private logic for the public ``column`` surface
lives here rather than in ``column.py``. These are the per-value routines the evaluate phase
applies for Column operators:

- :func:`cast_value` — Spark 4 casts with a ``strict`` flag (ANSI ``cast`` raises
  ``CAST_INVALID_INPUT`` on malformed input, ``try_cast`` returns null), sharing one parser per
  target type so the strict API never delegates to its lenient sibling.
- :func:`compare_values` — Spark's comparison operators, including its total ordering over
  NaN and nulls nested inside arrays/structs.
- :func:`logical_and` / :func:`logical_or` / :func:`logical_not` — SQL three-valued logic.

Only the casts the analyzer's implicit coercions insert are implemented; any other cast raises
``not_implemented_yet``.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Optional
from zoneinfo import ZoneInfo

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
        ShortType,
        StructType,
        TimestampType,
    )
except Exception:  # pragma: no cover - mock pyspark (under activate) has no real types
    pass


# --------------------------------------------------------------------------- #
# Casting
# --------------------------------------------------------------------------- #


class _Malformed(Exception):
    """Internal signal from a parser: the input is malformed for the target type."""


def cast_value(value: Any, target_type: Any, strict: bool) -> Any:
    """Cast one Python value to ``target_type`` (a PySpark ``DataType``) with Spark 4 semantics.

    A null stays null. Malformed input raises ``CAST_INVALID_INPUT`` when ``strict`` (ANSI
    ``cast``) and becomes null otherwise (``try_cast``).
    """
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return _cast_string(value, target_type)
        return _cast_non_string(value, target_type)
    except _Malformed:
        if not strict:
            return None
        raise ValueError(
            f"[CAST_INVALID_INPUT] The value '{value}' of the type \"STRING\" cannot be cast to "
            f'"{target_type.simpleString().upper()}" because it is malformed. Correct the value as per '
            f"the syntax, or change its target type. Use `try_cast` to tolerate malformed input and "
            f"return NULL instead."
        ) from None


def _cast_non_string(value: Any, target_type: Any) -> Any:
    if isinstance(target_type, (FloatType, DoubleType)):
        return float(value)
    if isinstance(target_type, (ByteType, ShortType, IntegerType, LongType)):
        return int(value)
    if isinstance(target_type, DecimalType):
        return Decimal(value)
    if isinstance(target_type, TimestampType) and isinstance(value, date) and not isinstance(value, datetime):
        return datetime.combine(value, time())
    not_implemented_yet(f"casting {type(value).__name__} to {type(target_type).__name__}")


def _cast_string(value: str, target_type: Any) -> Any:
    if isinstance(target_type, (ByteType, ShortType, IntegerType, LongType)):
        return _parse_integral(value, target_type)
    if isinstance(target_type, (FloatType, DoubleType)):
        return _parse_double(value)
    if isinstance(target_type, BooleanType):
        return _parse_boolean(value)
    if isinstance(target_type, DateType):
        return _parse_date(value)
    if isinstance(target_type, TimestampType):
        return _parse_timestamp(value)
    if isinstance(target_type, BinaryType):
        return value.encode("utf-8")
    not_implemented_yet(f"casting string to {type(target_type).__name__}")


_SPARK_WHITESPACE = "".join(chr(c) for c in range(33))


def _trim(value: str) -> str:
    """Spark trims ASCII whitespace and control characters (``<= ' '``) before parsing."""
    return value.strip(_SPARK_WHITESPACE)


# Keyed by class name: the type classes are not importable under the mock pyspark.
_INTEGRAL_RANGE = {
    "ByteType": (-(2**7), 2**7 - 1),
    "ShortType": (-(2**15), 2**15 - 1),
    "IntegerType": (-(2**31), 2**31 - 1),
    "LongType": (-(2**63), 2**63 - 1),
}
_INTEGRAL_RE = re.compile(r"[+-]?[0-9]+")


def _parse_integral(value: str, target_type: Any) -> int:
    """Plain (optionally signed) ASCII digits within the target range — no ``1.0``, ``1e2`` or ``1_0``."""
    text = _trim(value)
    if not _INTEGRAL_RE.fullmatch(text):
        raise _Malformed
    result = int(text)
    low, high = _INTEGRAL_RANGE[type(target_type).__name__]
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
_OFFSET_ZONE_RE = re.compile(r"(?:Z|(?:UTC|GMT|UT)?[+-][0-9]{1,2}(?::?[0-9]{2}(?::?[0-9]{2})?)?)")


def _build_date(year: str, month: Optional[str], day: Optional[str]) -> date:
    try:
        return date(int(year), int(month or 1), int(day or 1))
    except ValueError:
        raise _Malformed from None


def _parse_date(value: str) -> date:
    match = _DATE_RE.fullmatch(_trim(value))
    if match is None:
        raise _Malformed
    return _build_date(match["y"], match["m"], match["d"])


def _parse_timestamp(value: str) -> datetime:
    """Zone-less ``yyyy[-mm[-dd[ hh[:mm[:ss[.ffffff]]]]]]``; fractions beyond micros are truncated.

    A zone suffix (``Z``, ``+02:00``, ``UTC``, a region id) and a time-only string (which takes
    the current date) both depend on the session time zone, which the Python engine does not
    model yet — those raise ``not_implemented_yet`` instead of guessing.
    """
    text = _trim(value)
    if _TIME_ONLY_RE.fullmatch(text):
        not_implemented_yet("casting a time-only string to timestamp")
    match = _TIMESTAMP_RE.fullmatch(text)
    if match is None:
        raise _Malformed
    zone = match["zone"].strip()
    if zone:
        if _is_zone_id(zone):
            not_implemented_yet("casting a string with a time zone to timestamp")
        raise _Malformed
    day = _build_date(match["y"], match["m"], match["d"])
    hour, minute, second = int(match["H"] or 0), int(match["M"] or 0), int(match["S"] or 0)
    if hour > 23 or minute > 59 or second > 59:
        raise _Malformed
    micros = int((match["f"] or "0")[:6].ljust(6, "0"))
    return datetime(day.year, day.month, day.day, hour, minute, second, micros)


def _is_zone_id(text: str) -> bool:
    if _OFFSET_ZONE_RE.fullmatch(text):
        return True
    try:
        ZoneInfo(text)
    except Exception:
        return False
    return True


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
    order = _spark_order(left, right, data_type)
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


def _spark_order(left: Any, right: Any, data_type: Any) -> int:
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
        order = _spark_order(left_item, right_item, element_type)
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
