"""Shared, private helpers for ``functions.py`` (pure-Python engine).

Per the repo convention (CLAUDE.md), shared/private logic for the public ``functions`` surface
lives here rather than in ``functions.py``. These are the per-value routines the evaluate phase
applies; date/time parsing, JSON/struct, and array/map helpers will land here as the
corresponding functions are implemented.
"""

from __future__ import annotations

import math
from typing import Optional


def spark_pow(base: Optional[float], exponent: Optional[float]) -> Optional[float]:
    """``base ** exponent`` with Spark's (Java ``StrictMath.pow``) semantics; null in → null out.

    Python's ``math.pow`` follows C99 and *raises* where Java returns a special value, and it
    also differs on two NaN cases, so each divergence is mapped explicitly:

    - ``x ** ±0`` is ``1.0`` for every ``x``, NaN included (same as C99).
    - A NaN operand otherwise yields NaN — including ``1 ** NaN`` (C99 says ``1.0``).
    - ``(±1) ** ±inf`` is NaN (C99 says ``1.0``).
    - Overflow is ``±inf`` (Python raises ``OverflowError``).
    - ``±0 ** negative`` is ``±inf`` (Python raises ``ValueError``).
    - A negative finite base to a non-integer power is NaN (Python raises ``ValueError``).
    """
    if base is None or exponent is None:
        return None
    if exponent == 0.0:
        return 1.0
    if math.isnan(base) or math.isnan(exponent):
        return math.nan
    if math.isinf(exponent) and abs(base) == 1.0:
        return math.nan
    try:
        return math.pow(base, exponent)
    except OverflowError:
        return -math.inf if base < 0 and _is_odd_integer(exponent) else math.inf
    except ValueError:
        if base == 0.0:
            # Only -0.0 raised to a negative odd integer keeps its sign.
            return -math.inf if math.copysign(1.0, base) < 0 and _is_odd_integer(exponent) else math.inf
        return math.nan


def _is_odd_integer(value: float) -> bool:
    """Whether ``value`` is a finite, odd integral double (decides the sign of ``pow``'s infinities)."""
    return math.isfinite(value) and value.is_integer() and value % 2 == 1
