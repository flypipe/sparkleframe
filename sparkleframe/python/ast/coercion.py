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

from typing import Any, Optional, Tuple

from sparkleframe.python._errors import not_implemented_yet

try:  # pragma: no cover - exercised only with the real pyspark installed
    from pyspark.sql.types import (
        BooleanType,
        ByteType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        NullType,
        ShortType,
        StringType,
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
    not_implemented_yet(f"literal type inference for {type(value).__name__}")


def coerce_arithmetic(op: str, left_dt: Any, right_dt: Any) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Resolve the result type (and any operand casts) of ``left <op> right``.

    Returns ``(result_dt, left_cast_dt, right_cast_dt)`` where a ``*_cast_dt`` is the type the
    corresponding operand must be cast to (or ``None`` if no cast is needed). Only ``+ - *`` are
    supported in this slice — ``/`` and ``**`` result types are out of scope and raise.
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
