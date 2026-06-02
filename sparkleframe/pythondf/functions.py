"""PySpark-compatible functions for pythondf. Pure stdlib."""
from __future__ import annotations

import builtins
import datetime as _dt
import hashlib
import json as _json
import math
import random as _random
import re
import uuid as _uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Optional, Union

from sparkleframe.pythondf.column import Column, _row_count, _to_column
from sparkleframe.pythondf.types import (
    ArrayType,
    DataType,
    MapType,
    StructField,
    StructType,
    spark_type_name_to_type,
)


def col(name: str) -> Column:
    return Column(lambda d, n=name: list(d[n]), name=name)


def lit(value: Any) -> Column:
    return Column(lambda d, v=value: [v] * _row_count(d))


def _as_column(x: Union[str, Column, Any]) -> Column:
    if isinstance(x, Column):
        return x
    if isinstance(x, str):
        return col(x)
    return lit(x)


def coalesce(*cols: Union[str, Column]) -> Column:
    if not cols:
        raise ValueError("coalesce requires at least one column")
    evals = [_as_column(c)._eval for c in cols]

    def _ev(d):
        n = _row_count(d)
        cols_vals = [e(d) for e in evals]
        out: list = [None] * n
        for i in range(n):
            for vals in cols_vals:
                if vals[i] is not None:
                    out[i] = vals[i]
                    break
        return out
    return Column(_ev)


def _horizontal_reduce(cols, op_name: str, reducer):
    if len(cols) < 2:
        raise ValueError(f"{op_name} requires at least two arguments")
    evals = [_as_column(c)._eval for c in cols]

    def _ev(d):
        n = _row_count(d)
        per_col = [e(d) for e in evals]
        out: list = []
        for i in range(n):
            chosen: Any = None
            for vals in per_col:
                v = vals[i]
                if v is None:
                    continue
                if chosen is None or reducer(v, chosen):
                    chosen = v
            out.append(chosen)
        return out
    return Column(_ev)


def least(*cols: Any) -> Column:
    return _horizontal_reduce(cols, "least", lambda v, chosen: v < chosen)


def greatest(*cols: Any) -> Column:
    return _horizontal_reduce(cols, "greatest", lambda v, chosen: v > chosen)


class WhenColumn(Column):
    """A :class:`Column` produced by ``when(...)`` that also supports chaining
    further ``.when(...)`` clauses and a final ``.otherwise(...)``.

    PySpark allows using ``F.when(cond, val)`` directly as a Column (NULL where
    no branch matches), so we model the no-otherwise result as a real Column
    while still exposing ``.when()`` / ``.otherwise()`` for chaining.
    """

    def __init__(self, branches: list[tuple[Column, Column]]):
        self._branches: list[tuple[Column, Column]] = list(branches)

        def _ev_no_otherwise(d, _branches=self._branches):
            n = _row_count(d)
            cond_vals = [c.eval(d) for c, _ in _branches]
            val_vals = [v.eval(d) for _, v in _branches]
            out: list = [None] * n
            for i in range(n):
                for k, cv in enumerate(cond_vals):
                    if cv[i] is True:
                        out[i] = val_vals[k][i]
                        break
            return out

        super().__init__(_ev_no_otherwise)

    def when(self, condition: Any, value: Any) -> "WhenColumn":
        cond_col = condition if isinstance(condition, Column) else _to_column(condition)
        return WhenColumn([*self._branches, (cond_col, _to_column(value))])

    def otherwise(self, value: Any) -> Column:
        else_col = _to_column(value)
        branches = list(self._branches)

        def _ev(d):
            n = _row_count(d)
            cond_vals = [c.eval(d) for c, _ in branches]
            val_vals = [v.eval(d) for _, v in branches]
            else_vals = else_col.eval(d)
            out: list = [None] * n
            for i in range(n):
                matched = False
                for k, cv in enumerate(cond_vals):
                    if cv[i] is True:
                        out[i] = val_vals[k][i]
                        matched = True
                        break
                if not matched:
                    # Match Spark: if all conditions false/null, use otherwise
                    out[i] = else_vals[i]
            return out
        return Column(_ev)


# Backwards-compatible alias for code that imported the old type name.
WhenBuilder = WhenColumn


def when(condition: Any, value: Any) -> WhenColumn:
    cond_col = condition if isinstance(condition, Column) else _to_column(condition)
    return WhenColumn([(cond_col, _to_column(value))])


# ---------------------------------------------------------------------------
# Sort helpers (module-level mirrors of Column.asc/.desc/... )
# ---------------------------------------------------------------------------


def _as_sort_column(column: Union[str, Column]) -> Column:
    """Resolve a str column name or pass through a Column for sort helpers."""
    if isinstance(column, Column):
        return column
    if isinstance(column, str):
        return col(column)
    raise TypeError(
        f"sort helper expects a column name (str) or Column, got {type(column).__name__}"
    )


def asc(column: Union[str, Column]) -> Column:
    """Mimics pyspark.sql.functions.asc."""
    return _as_sort_column(column).asc()


def asc_nulls_first(column: Union[str, Column]) -> Column:
    """Mimics pyspark.sql.functions.asc_nulls_first."""
    return _as_sort_column(column).asc_nulls_first()


def asc_nulls_last(column: Union[str, Column]) -> Column:
    """Mimics pyspark.sql.functions.asc_nulls_last."""
    return _as_sort_column(column).asc_nulls_last()


def desc(column: Union[str, Column]) -> Column:
    """Mimics pyspark.sql.functions.desc."""
    return _as_sort_column(column).desc()


def desc_nulls_first(column: Union[str, Column]) -> Column:
    """Mimics pyspark.sql.functions.desc_nulls_first."""
    return _as_sort_column(column).desc_nulls_first()


def desc_nulls_last(column: Union[str, Column]) -> Column:
    """Mimics pyspark.sql.functions.desc_nulls_last."""
    return _as_sort_column(column).desc_nulls_last()


# ---------------------------------------------------------------------------
# Aggregate functions
# ---------------------------------------------------------------------------


def _agg_source_info(col_name):
    """Resolve an agg arg into (source_name, source_eval, display_name).

    - ``str``: source is the named column; display name is the bare name.
    - ``Column`` with a ``.name``: treat like a bare column reference.
    - ``Column`` without a name: arbitrary expression; produce values via ``_eval``.
    """
    if isinstance(col_name, str):
        c = col(col_name)
        return col_name, c._eval, col_name
    if isinstance(col_name, Column):
        # Even if the Column has a custom expression, use its eval to produce
        # the per-row values. Display name falls back to its `.name` or "expr".
        return col_name.name, col_name._eval, col_name.name or "expr"
    raise TypeError(f"aggregate expects str or Column, got {type(col_name).__name__}")


def _make_agg_column(
    source_arg,
    agg_fn,
    fn_label: str,
    *,
    source_name_override: str | None = None,
    display_label: str | None = None,
) -> Column:
    """Construct an aggregate Column carrying ``_agg_fn`` metadata.

    The Column's ``_eval`` for standalone use produces a 1-element list
    containing the reduced scalar (so ``df.select(F.sum(...))`` gives one row).
    """
    source_name, source_eval, display = _agg_source_info(source_arg) if source_arg is not None else (None, None, "")

    if source_name_override is not None:
        source_name = source_name_override
    if display_label is not None:
        display = display_label

    pretty_name = f"{fn_label}({display})"

    def _ev(d):
        if source_name == "*":
            n = _row_count(d)
            return [agg_fn([1] * n)]
        if source_eval is None:
            return [agg_fn([])]
        vals = source_eval(d)
        return [agg_fn(vals)]

    out = Column(_ev, name=pretty_name)
    out._agg_fn = agg_fn
    out._agg_source_name = source_name
    out._agg_source_eval = source_eval
    return out


def _sum_ignore_nulls(values):
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None
    return builtins.sum(non_null)


def _count_ignore_nulls(values):
    # PySpark count returns long.
    return builtins.sum(1 for v in values if v is not None)


def _count_all(values):
    return len(values)


def _mean_ignore_nulls(values):
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None
    return float(builtins.sum(non_null)) / len(non_null)


def _min_ignore_nulls(values):
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None
    return builtins.min(non_null)


def _max_ignore_nulls(values):
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None
    return builtins.max(non_null)


def sum(col_name):  # noqa: A001 — shadow builtin to match PySpark API
    return _make_agg_column(col_name, _sum_ignore_nulls, "sum")


def count(col_name):
    if isinstance(col_name, str) and col_name == "*":
        # COUNT(*) — count rows, not values.
        out_name = "count(1)"

        def _ev(d):
            return [_row_count(d)]

        out = Column(_ev, name=out_name)
        out._agg_fn = _count_all
        out._agg_source_name = "*"
        out._agg_source_eval = None
        return out
    return _make_agg_column(col_name, _count_ignore_nulls, "count")


def mean(col_name):
    return _make_agg_column(col_name, _mean_ignore_nulls, "avg")


# PySpark alias.
avg = mean


def min(col_name):  # noqa: A001
    return _make_agg_column(col_name, _min_ignore_nulls, "min")


def max(col_name):  # noqa: A001
    return _make_agg_column(col_name, _max_ignore_nulls, "max")


def first(col_name, ignorenulls: bool = False) -> Column:
    """First non-null (if ``ignorenulls=True``) or first value of the column."""
    if ignorenulls:
        def _agg(values):
            for v in values:
                if v is not None:
                    return v
            return None
    else:
        def _agg(values):
            if not values:
                return None
            return values[0]

    return _make_agg_column(col_name, _agg, "first")


def collect_list(col_name) -> Column:
    """Collect non-null values into a list (preserves input row order)."""
    def _agg(values):
        return [v for v in values if v is not None]
    return _make_agg_column(col_name, _agg, "collect_list")


# ---------------------------------------------------------------------------
# Math functions
# ---------------------------------------------------------------------------


def abs(col_name: Union[str, Column]) -> Column:  # noqa: A001 — shadow builtin to match PySpark
    """Absolute value. Null in → null out."""
    src = _as_column(col_name)._eval

    def _ev(d):
        return [None if v is None else builtins.abs(v) for v in src(d)]

    return Column(_ev)


def floor(col_name: Union[str, Column]) -> Column:
    """Floor toward -infinity; returns int (Spark LongType). Null in → null out."""
    src = _as_column(col_name)._eval

    def _ev(d):
        out = []
        for v in src(d):
            if v is None:
                out.append(None)
            else:
                try:
                    out.append(int(math.floor(float(v))))
                except (TypeError, ValueError):
                    out.append(None)
        return out

    return Column(_ev)


def _round_half_up(value: float, scale: int) -> float:
    """Spark-compatible HALF_UP rounding using Decimal.

    Python's built-in ``round`` uses banker's rounding (HALF_EVEN), which
    diverges from Spark's ``round`` (which uses HALF_UP). Spark's ``bround``
    uses HALF_EVEN; we keep that distinction by routing through Decimal here.
    """
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value
    try:
        d = Decimal(str(value))
    except Exception:
        return None
    if scale >= 0:
        q = Decimal(1).scaleb(-scale)  # 10^-scale
    else:
        q = Decimal(1).scaleb(-scale)  # 10^|scale|
    quantized = d.quantize(q, rounding=ROUND_HALF_UP)
    # Always return a float — Spark's round() preserves the input numeric type
    # (Double in → Double out, Long in → Long out), but our values are already
    # native Python and the parity comparator normalises floats vs ints.
    return float(quantized)


def round(col_name: Union[str, Column], scale: int = 0) -> Column:  # noqa: A001
    """Round to ``scale`` decimal places using HALF_UP (Spark's ``round``).

    Use ``bround`` (not implemented here) for banker's rounding.
    """
    src = _as_column(col_name)._eval

    def _ev(d):
        out = []
        for v in src(d):
            if v is None:
                out.append(None)
                continue
            try:
                out.append(_round_half_up(v, scale))
            except Exception:
                out.append(None)
        return out

    return Column(_ev)


def pow(base: Any, exponent: Any) -> Column:  # noqa: A001
    """``base ** exponent`` — always returns Double. Null if either side is null."""
    base_col = _as_column(base)._eval
    exp_col = _as_column(exponent)._eval

    def _ev(d):
        b_vals = base_col(d)
        e_vals = exp_col(d)
        out = []
        for b, e in zip(b_vals, e_vals):
            if b is None or e is None:
                out.append(None)
            else:
                try:
                    out.append(float(b) ** float(e))
                except (TypeError, ValueError):
                    out.append(None)
        return out

    return Column(_ev)


def isnan(col_name: Union[str, Column]) -> Column:
    """True for IEEE 754 NaN. Null input → null output (Spark behaviour)."""
    src = _as_column(col_name)._eval

    def _ev(d):
        out = []
        for v in src(d):
            if v is None:
                # Spark returns False for null on isnan; polarsdf does same.
                out.append(False)
            elif isinstance(v, float):
                out.append(math.isnan(v))
            else:
                out.append(False)
        return out

    return Column(_ev)


def try_divide(left: Union[str, Column], right: Union[str, Column]) -> Column:
    """Return ``left / right`` as float; null on divide-by-zero or null operands."""
    l_col = _as_column(left)._eval
    r_col = _as_column(right)._eval

    def _ev(d):
        l_vals = l_col(d)
        r_vals = r_col(d)
        out = []
        for a, b in zip(l_vals, r_vals):
            if a is None or b is None:
                out.append(None)
                continue
            try:
                bf = float(b)
                if bf == 0.0:
                    out.append(None)
                else:
                    out.append(float(a) / bf)
            except (TypeError, ValueError, ZeroDivisionError):
                out.append(None)
        return out

    return Column(_ev)


def nullif(e1: Union[str, Column], e2: Union[str, Column]) -> Column:
    """Return null when e1 == e2 (both non-null); otherwise return e1.

    Matches Spark's ``CASE WHEN e1 = e2 THEN NULL ELSE e1 END``. If either
    operand is null, equality is unknown so we return e1.
    """
    a_col = _as_column(e1)._eval
    b_col = _as_column(e2)._eval

    def _ev(d):
        a_vals = a_col(d)
        b_vals = b_col(d)
        out = []
        for a, b in zip(a_vals, b_vals):
            if a is None or b is None:
                out.append(a)
            elif a == b:
                out.append(None)
            else:
                out.append(a)
        return out

    return Column(_ev)


def collect_set(col_name) -> Column:
    """Collect distinct non-null values into a list (order undefined per Spark)."""
    def _agg(values):
        seen: set = set()
        out: list = []
        for v in values:
            if v is None:
                continue
            try:
                if v in seen:
                    continue
                seen.add(v)
            except TypeError:
                # Unhashable — fall back to linear membership check.
                if v in out:
                    continue
            out.append(v)
        return out
    return _make_agg_column(col_name, _agg, "collect_set")


# ---------------------------------------------------------------------------
# String functions
# ---------------------------------------------------------------------------


def _as_str_or_none(v: Any) -> Any:
    """Return ``v`` as a string if non-null; else None.

    Spark's string functions coerce numeric inputs to strings (e.g. ``concat``
    accepts integer columns). We mirror that by calling ``str()`` on non-None
    non-str values.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def upper(col_name: Union[str, Column]) -> Column:
    """Upper-case each string. Null in → null out."""
    src = _as_column(col_name)._eval

    def _ev(d):
        return [None if v is None else _as_str_or_none(v).upper() for v in src(d)]

    return Column(_ev)


def lower(col_name: Union[str, Column]) -> Column:
    """Lower-case each string. Null in → null out."""
    src = _as_column(col_name)._eval

    def _ev(d):
        return [None if v is None else _as_str_or_none(v).lower() for v in src(d)]

    return Column(_ev)


def _initcap_one(s: str) -> str:
    """Spark ``initcap``: split on whitespace, capitalize first char of each word,
    lower-case the rest. Whitespace runs are collapsed/preserved per Spark — Spark
    actually preserves the *original* whitespace; we emulate by walking the string.
    """
    if not s:
        return s
    out_chars: list[str] = []
    new_word = True
    for ch in s:
        if ch.isspace():
            out_chars.append(ch)
            new_word = True
        else:
            if new_word:
                out_chars.append(ch.upper())
                new_word = False
            else:
                out_chars.append(ch.lower())
    return "".join(out_chars)


def initcap(col_name: Union[str, Column]) -> Column:
    """Capitalize the first letter of each whitespace-separated word; lower-case
    the rest. Null in → null out.
    """
    src = _as_column(col_name)._eval

    def _ev(d):
        return [None if v is None else _initcap_one(_as_str_or_none(v)) for v in src(d)]

    return Column(_ev)


def trim(col_name: Union[str, Column]) -> Column:
    """Strip leading/trailing ASCII space (U+0020) only, matching Spark's ``trim``."""
    src = _as_column(col_name)._eval

    def _ev(d):
        out: list = []
        for v in src(d):
            if v is None:
                out.append(None)
            else:
                out.append(_as_str_or_none(v).strip(" "))
        return out

    return Column(_ev)


def length(col_name: Union[str, Column]) -> Column:
    """Number of characters. Spark returns IntegerType. Null in → null out."""
    src = _as_column(col_name)._eval

    def _ev(d):
        out: list = []
        for v in src(d):
            if v is None:
                out.append(None)
            elif isinstance(v, (bytes, bytearray)):
                out.append(len(v))
            else:
                out.append(len(_as_str_or_none(v)))
        return out

    return Column(_ev)


def _substring_one(value: Any, pos: int, ln: int) -> Any:
    """Spark substring: 1-based; pos<0 counts from end; length<=0 -> empty string."""
    if value is None:
        return None
    if ln <= 0:
        return ""
    s = value if isinstance(value, str) else str(value)
    n = len(s)
    if pos > 0:
        start = pos - 1
    elif pos < 0:
        start = n + pos
    else:
        start = 0
    if start < 0:
        start = 0
    if start >= n:
        return ""
    end = start + ln
    if end > n:
        end = n
    return s[start:end]


def substring(col_name: Union[str, Column], pos: int, length: int) -> Column:  # noqa: A002 — match PySpark
    """Spark substring (1-indexed; negative ``pos`` counts from end)."""
    src = _as_column(col_name)._eval
    p, ln = pos, length

    def _ev(d):
        return [_substring_one(v, p, ln) for v in src(d)]

    return Column(_ev)


def concat(*cols: Union[str, Column]) -> Column:
    """Concatenate string columns. If any operand is null → result null (Spark default)."""
    if not cols:
        raise ValueError("concat requires at least one column")
    evals = [_as_column(c)._eval for c in cols]

    def _ev(d):
        n = _row_count(d)
        per_col = [e(d) for e in evals]
        out: list = []
        for i in range(n):
            row_parts: list[str] = []
            null_seen = False
            for vals in per_col:
                v = vals[i]
                if v is None:
                    null_seen = True
                    break
                row_parts.append(_as_str_or_none(v))
            out.append(None if null_seen else "".join(row_parts))
        return out

    return Column(_ev)


def _split_one(value: Any, pattern: str, limit: int) -> Any:
    """Spark split limit semantics:
    - limit <= 0: split as many times as possible (no cap).
    - limit == 1: return the whole string as a single-element list.
    - limit > 1: at most ``limit`` pieces (maxsplit = limit - 1).
    """
    if value is None:
        return None
    s = value if isinstance(value, str) else str(value)
    if limit <= 0:
        return re.split(pattern, s)
    if limit == 1:
        return [s]
    return re.split(pattern, s, maxsplit=limit - 1)


def split(col_name: Union[str, Column], pattern: str, limit: int = -1) -> Column:
    """Split each string on a regex pattern; returns an array column."""
    src = _as_column(col_name)._eval
    pat, lim = pattern, limit

    def _ev(d):
        return [_split_one(v, pat, lim) for v in src(d)]

    return Column(_ev)


def regexp_replace(col_name: Union[str, Column], pattern: str, replacement: str) -> Column:
    """Replace all regex matches in each string. Null in → null out."""
    src = _as_column(col_name)._eval
    pat, repl = pattern, replacement

    def _ev(d):
        out: list = []
        for v in src(d):
            if v is None:
                out.append(None)
            else:
                s = v if isinstance(v, str) else str(v)
                out.append(re.sub(pat, repl, s))
        return out

    return Column(_ev)


def md5(col_name: Union[str, Column]) -> Column:
    """MD5 digest as 32-char lowercase hex string (Spark: UTF-8 for strings, raw bytes for binary)."""
    src = _as_column(col_name)._eval

    def _ev(d):
        out: list = []
        for v in src(d):
            if v is None:
                out.append(None)
            elif isinstance(v, (bytes, bytearray, memoryview)):
                out.append(hashlib.md5(bytes(v), usedforsecurity=False).hexdigest())
            else:
                s = v if isinstance(v, str) else str(v)
                out.append(hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest())
        return out

    return Column(_ev)


# ---------------------------------------------------------------------------
# Date/time functions
# ---------------------------------------------------------------------------


# Spark datetime-pattern → strftime substitutions, ordered longest-first so
# multi-char tokens like ``yyyy`` match before single-char prefixes inside them.
_SPARK_FMT_SUBSTITUTIONS: list[tuple[str, str]] = [
    ("yyyy", "%Y"),
    ("yy", "%y"),
    ("MM", "%m"),
    ("dd", "%d"),
    ("HH", "%H"),
    ("mm", "%M"),
    ("ss", "%S"),
    ("SSSSSS", "%f"),
    ("SSS", "%f"),
]


def _spark_fmt_to_strftime(fmt: str) -> str:
    """Translate a Spark datetime pattern (subset) to a Python ``strftime`` format.

    Supported tokens: ``yyyy`` / ``yy`` / ``MM`` / ``dd`` / ``HH`` / ``mm``
    / ``ss`` / ``SSS`` / ``SSSSSS``. Anything else is left as-is.

    Caveats:
      * Spark's ``mm`` always means minutes (we honor that); the calendar uses
        ``MM`` for month-of-year — same as Spark.
      * Microseconds are emitted as ``%f`` (6 digits) which differs from Spark's
        millisecond-precision (3 digits) when the format requests ``.SSS``; for
        our parity tests we round-trip via ``strptime``/``strftime`` so this
        mismatch only surfaces for direct ``date_format`` output.
    """
    out = fmt
    for spark_tok, py_tok in _SPARK_FMT_SUBSTITUTIONS:
        out = out.replace(spark_tok, py_tok)
    return out


def _coerce_to_date(value: Any) -> Optional[_dt.date]:
    """Best-effort coerce ``value`` to a ``datetime.date``; ``None`` on failure.

    Accepts ``date``, ``datetime``, and ISO-style strings (``YYYY-MM-DD`` with
    optional time suffix).
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        s = value.strip()
        # Strict ISO-8601 date prefix.
        try:
            return _dt.date.fromisoformat(s[:10])
        except ValueError:
            return None
    return None


def _coerce_to_datetime(value: Any) -> Optional[_dt.datetime]:
    """Best-effort coerce ``value`` to a naive ``datetime``; ``None`` on failure."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        # Drop tzinfo to match Spark TimestampType (which we model as naive).
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        s = value.strip()
        # Try ISO 8601 (with optional T or Z).
        candidate = s.replace("Z", "+00:00") if s.endswith("Z") else s
        try:
            parsed = _dt.datetime.fromisoformat(candidate)
            if parsed.tzinfo is not None:
                parsed = parsed.replace(tzinfo=None)
            return parsed
        except ValueError:
            pass
        # Try common space-separated form.
        try:
            return _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def now() -> Column:
    """Single wall-clock timestamp broadcast to all rows.

    Uses naive local time to match Spark's session-TZ semantics for ``current_timestamp``
    (PySpark returns ``datetime`` in the JVM's session TZ on ``.collect()``).
    """
    ts = _dt.datetime.now()

    def _ev(d):
        return [ts] * _row_count(d)

    return Column(_ev)


def current_timestamp() -> Column:
    """Alias for :func:`now` (same Spark semantics)."""
    return now()


def current_date() -> Column:
    """Today's date (local), broadcast to all rows.

    Spark's ``current_date`` returns the JVM session-TZ date; using local matches it
    on the same host.
    """
    today = _dt.date.today()

    def _ev(d):
        return [today] * _row_count(d)

    return Column(_ev)


def _parse_with_fmt(value: Any, fmt: Optional[str], *, want_date: bool) -> Any:
    """Parse ``value`` per ``fmt`` (Spark pattern) into date/datetime; return sentinel-free.

    Returns the parsed object, ``None`` (for null inputs), or raises ``ValueError``
    on malformed input. Caller decides whether to swallow.
    """
    if value is None:
        return None
    if fmt is None:
        # ISO 8601 default.
        if want_date:
            parsed = _coerce_to_date(value)
        else:
            parsed = _coerce_to_datetime(value)
        if parsed is None:
            raise ValueError(f"Cannot parse {value!r} with default format")
        return parsed
    # Format provided — strict strptime via translated pattern.
    if not isinstance(value, str):
        # Spark coerces to string first; cast through str().
        value = str(value)
    py_fmt = _spark_fmt_to_strftime(fmt)
    parsed = _dt.datetime.strptime(value, py_fmt)
    return parsed.date() if want_date else parsed


def to_date(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    """Parse strings to ``date``. Spark 4 ANSI: malformed inputs raise."""
    src = _as_column(col_name)._eval
    f = fmt

    def _ev(d):
        out: list = []
        for v in src(d):
            if v is None:
                out.append(None)
                continue
            try:
                out.append(_parse_with_fmt(v, f, want_date=True))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"[CAST_INVALID_INPUT] Cannot cast {v!r} to date with format {f!r}"
                ) from exc
        return out

    return Column(_ev)


def try_to_date(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    """Lenient :func:`to_date`: malformed inputs become NULL."""
    src = _as_column(col_name)._eval
    f = fmt

    def _ev(d):
        out: list = []
        for v in src(d):
            if v is None:
                out.append(None)
                continue
            try:
                out.append(_parse_with_fmt(v, f, want_date=True))
            except (ValueError, TypeError):
                out.append(None)
        return out

    return Column(_ev)


def to_timestamp(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    """Parse strings to ``datetime``. Spark 4 ANSI: malformed inputs raise."""
    src = _as_column(col_name)._eval
    f = fmt

    def _ev(d):
        out: list = []
        for v in src(d):
            if v is None:
                out.append(None)
                continue
            try:
                out.append(_parse_with_fmt(v, f, want_date=False))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"[CAST_INVALID_INPUT] Cannot cast {v!r} to timestamp with format {f!r}"
                ) from exc
        return out

    return Column(_ev)


def try_to_timestamp(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    """Lenient :func:`to_timestamp`: malformed inputs become NULL."""
    src = _as_column(col_name)._eval
    f = fmt

    def _ev(d):
        out: list = []
        for v in src(d):
            if v is None:
                out.append(None)
                continue
            try:
                out.append(_parse_with_fmt(v, f, want_date=False))
            except (ValueError, TypeError):
                out.append(None)
        return out

    return Column(_ev)


def date_format(col_name: Union[str, Column], fmt: str) -> Column:
    """Format a date/timestamp column as a string using a Spark datetime pattern.

    Inputs are first coerced to ``datetime`` (date → midnight). A null input
    produces a null output.
    """
    src = _as_column(col_name)._eval
    py_fmt = _spark_fmt_to_strftime(fmt)

    def _ev(d):
        out: list = []
        for v in src(d):
            ts = _coerce_to_datetime(v)
            if ts is None:
                out.append(None)
            else:
                out.append(ts.strftime(py_fmt))
        return out

    return Column(_ev)


def date_sub(col_name: Union[str, Column], days: int) -> Column:
    """Subtract ``days`` calendar days. Returns a ``date``; null in → null out."""
    src = _as_column(col_name)._eval
    n = int(days)

    def _ev(d):
        out: list = []
        for v in src(d):
            base = _coerce_to_date(v)
            if base is None:
                out.append(None)
            else:
                out.append(base - _dt.timedelta(days=n))
        return out

    return Column(_ev)


def datediff(end: Union[str, Column], start: Union[str, Column]) -> Column:
    """``end - start`` in whole calendar days (int). Null in either side → null."""
    e_eval = _as_column(end)._eval
    s_eval = _as_column(start)._eval

    def _ev(d):
        e_vals = e_eval(d)
        s_vals = s_eval(d)
        out: list = []
        for ev, sv in zip(e_vals, s_vals):
            ed = _coerce_to_date(ev)
            sd = _coerce_to_date(sv)
            if ed is None or sd is None:
                out.append(None)
            else:
                out.append((ed - sd).days)
        return out

    return Column(_ev)


def months_between(end: Union[str, Column], start: Union[str, Column]) -> Column:
    """Months between ``end`` and ``start`` as a float using Spark's day/31 rule.

    Approximation: ``(year_diff*12 + month_diff) + (end_day - start_day) / 31.0``.
    When both day-of-month components match, the result is exactly an integer.
    Null in either side → null.
    """
    e_eval = _as_column(end)._eval
    s_eval = _as_column(start)._eval

    def _ev(d):
        e_vals = e_eval(d)
        s_vals = s_eval(d)
        out: list = []
        for ev, sv in zip(e_vals, s_vals):
            ed = _coerce_to_date(ev)
            sd = _coerce_to_date(sv)
            if ed is None or sd is None:
                out.append(None)
                continue
            whole = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if ed.day == sd.day:
                out.append(float(whole))
            else:
                # Spark rounds the fractional result to 8 decimal places.
                out.append(builtins.round(float(whole) + (ed.day - sd.day) / 31.0, 8))
        return out

    return Column(_ev)


# ---------------------------------------------------------------------------
# Array functions
# ---------------------------------------------------------------------------


# Magic per-element data key used by transform/filter to feed the inner
# expression. Element-bound column references this when the lambda body is
# applied to each array element.
_ELEMENT_KEY = "__pythondf_element__"


def array(*cols: Any) -> Column:
    """Pack columns/literals into a list per row.

    Empty ``array()`` yields ``[]`` for every row (Spark behaviour).
    Matches PySpark's ``F.array``: each per-row list contains one entry per
    argument, in argument order.
    """
    if not cols:
        def _ev_empty(d):
            return [[] for _ in range(_row_count(d))]
        return Column(_ev_empty)

    evals = [_as_column(c)._eval for c in cols]

    def _ev(d):
        n = _row_count(d)
        per_col = [e(d) for e in evals]
        return [[per_col[k][i] for k in range(len(per_col))] for i in range(n)]

    return Column(_ev)


def array_contains(col_name: Union[str, Column], value: Any) -> Column:
    """Spark ``array_contains``.

    Semantics: null array → null; null value → null; otherwise True/False.
    """
    src = _as_column(col_name)._eval
    val_col = _as_column(value)
    val_eval = val_col._eval

    def _ev(d):
        arrs = src(d)
        vals = val_eval(d)
        out: list = []
        for arr, v in zip(arrs, vals):
            if arr is None:
                out.append(None)
            elif v is None:
                out.append(None)
            else:
                try:
                    out.append(v in arr)
                except TypeError:
                    out.append(False)
        return out

    return Column(_ev)


def size(col_name: Union[str, Column]) -> Column:
    """Length of array/map. Spark's ``size`` returns -1 for null when
    ``spark.sql.legacy.sizeOfNull=true`` (default in older Spark) and ``null``
    when false. Modern Spark (3.0+) defaults to returning ``null`` for null;
    we match that since Spark 4 is the parity target.
    """
    src = _as_column(col_name)._eval

    def _ev(d):
        out: list = []
        for v in src(d):
            if v is None:
                out.append(None)
            elif isinstance(v, (list, tuple, dict)):
                out.append(len(v))
            else:
                out.append(None)
        return out

    return Column(_ev)


def _element_at_one(arr: Any, index: Any, *, strict: bool) -> Any:
    """1-indexed element access for arrays; ``.get`` for dicts.

    On out-of-bounds with ``strict=True``, raises (Spark ANSI). Otherwise
    returns ``None``.
    """
    if arr is None:
        return None
    if index is None:
        return None
    if isinstance(arr, dict):
        return arr.get(index)
    if isinstance(arr, (list, tuple)):
        if not isinstance(index, int) or isinstance(index, bool):
            if strict:
                raise ValueError(f"element_at index must be int for arrays, got {type(index).__name__}")
            return None
        if index == 0:
            if strict:
                raise ValueError("SQL array indices start at 1")
            return None
        n = len(arr)
        if index > 0:
            i = index - 1
        else:
            i = n + index  # negative: from end
        if i < 0 or i >= n:
            if strict:
                raise IndexError(f"element_at index {index} out of bounds for array of length {n}")
            return None
        return arr[i]
    return None


def element_at(col_name: Union[str, Column], extraction: Any) -> Column:
    """Spark ``element_at`` (ANSI strict for arrays): 1-indexed, negative from end.

    For maps, ``extraction`` is a literal key.
    """
    src = _as_column(col_name)._eval
    ext_col = _as_column(extraction)
    ext_eval = ext_col._eval

    def _ev(d):
        arrs = src(d)
        idxs = ext_eval(d)
        out: list = []
        for arr, idx in zip(arrs, idxs):
            out.append(_element_at_one(arr, idx, strict=True))
        return out

    return Column(_ev)


def try_element_at(col_name: Union[str, Column], extraction: Any) -> Column:
    """Lenient :func:`element_at`: out-of-bounds → null instead of raising."""
    src = _as_column(col_name)._eval
    ext_col = _as_column(extraction)
    ext_eval = ext_col._eval

    def _ev(d):
        arrs = src(d)
        idxs = ext_eval(d)
        out: list = []
        for arr, idx in zip(arrs, idxs):
            out.append(_element_at_one(arr, idx, strict=False))
        return out

    return Column(_ev)


class _SortKey:
    """Composite sort key that places nulls first (asc) regardless of value type."""

    __slots__ = ("is_null", "value", "desc")

    def __init__(self, value: Any, desc: bool):
        self.is_null = value is None
        self.value = value
        self.desc = desc

    def __lt__(self, other: "_SortKey") -> bool:
        # In Spark, ascending sort places nulls first; descending places them last.
        if self.is_null and other.is_null:
            return False
        if self.desc:
            # Descending: nulls last
            if self.is_null:
                return False
            if other.is_null:
                return True
            return other.value < self.value
        # Ascending: nulls first
        if self.is_null:
            return True
        if other.is_null:
            return False
        return self.value < other.value


def sort_array(col_name: Union[str, Column], asc: bool = True) -> Column:
    """Sort each array. ``asc=True`` → ascending, nulls first; ``asc=False`` →
    descending, nulls last (Spark semantics)."""
    src = _as_column(col_name)._eval
    descending = not asc

    def _ev(d):
        out: list = []
        for arr in src(d):
            if arr is None:
                out.append(None)
            elif isinstance(arr, (list, tuple)):
                keys = [_SortKey(v, descending) for v in arr]
                indices = sorted(range(len(arr)), key=lambda i, k=keys: k[i])
                out.append([arr[i] for i in indices])
            else:
                out.append(None)
        return out

    return Column(_ev)


# ---- transform / filter --------------------------------------------------


def _element_column() -> Column:
    """Build the placeholder Column referenced by ``func`` inside transform/filter."""
    return Column(lambda d: list(d[_ELEMENT_KEY]), name="element")


def _apply_inner_to_elements(elements: list, func: Callable[[Column], Any]) -> list:
    """Evaluate ``func(element_col)`` against a list of array elements.

    Builds a tiny per-element ``Data`` dict that satisfies any references to the
    element placeholder, then calls the lambda once with the element column and
    evaluates the resulting Column against the dict to produce one result per
    element.
    """
    if not elements:
        return []
    element_col = _element_column()
    result = func(element_col)
    if not isinstance(result, Column):
        # Lambda returned a literal — broadcast it to each element.
        return [result] * len(elements)
    return result.eval({_ELEMENT_KEY: list(elements)})


def transform(col_name: Union[str, Column], func: Callable[[Column], Any]) -> Column:
    """Spark ``transform``: apply ``func`` to each element of each array.

    The lambda receives a :class:`Column` representing the inner element and
    returns a transformed Column expression. Null arrays stay null; empty
    arrays stay empty.
    """
    src = _as_column(col_name)._eval

    def _ev(d):
        out: list = []
        for arr in src(d):
            if arr is None:
                out.append(None)
            elif isinstance(arr, (list, tuple)):
                out.append(_apply_inner_to_elements(list(arr), func))
            else:
                out.append(None)
        return out

    return Column(_ev)


def filter(col_name: Union[str, Column], func: Callable[[Column], Any]) -> Column:  # noqa: A001
    """Spark ``filter`` (higher-order on arrays): keep elements where
    ``func(element)`` evaluates to ``True``.

    Note: shadows :func:`builtins.filter` to match PySpark API; module-internal
    code uses ``builtins.filter`` when needed.
    """
    src = _as_column(col_name)._eval

    def _ev(d):
        out: list = []
        for arr in src(d):
            if arr is None:
                out.append(None)
            elif isinstance(arr, (list, tuple)):
                elems = list(arr)
                mask = _apply_inner_to_elements(elems, func)
                kept = [e for e, keep in zip(elems, mask) if keep is True]
                out.append(kept)
            else:
                out.append(None)
        return out

    return Column(_ev)


# ---- explode --------------------------------------------------------------


# ---------------------------------------------------------------------------
# struct / map functions
# ---------------------------------------------------------------------------


def _struct_field_name(arg: Union[str, Column], index: int) -> str:
    """Spark ``CreateStruct`` naming rules:

    - string column reference → use the (last segment of the) name
    - Column with an explicit ``.alias()`` or a plain ``col(name)`` → use ``_name``
    - literals / anonymous expressions → ``col1``, ``col2``, ...
    """
    if isinstance(arg, str):
        return arg.split(".")[-1]
    if isinstance(arg, Column) and arg._name:
        return arg._name.split(".")[-1]
    return f"col{index + 1}"


def struct(*cols: Any) -> Column:
    """Pack columns/literals into a dict (struct) per row.

    Field-naming mirrors PySpark's ``CreateStruct``: string args / plain
    ``F.col(name)`` keep their name; ``F.lit(v).alias("k")`` uses the alias;
    unaliased literals become ``col1``, ``col2``, ....

    A single ``list`` / ``set`` arg is expanded (PySpark 3.4+ behaviour).
    """
    if len(cols) == 1 and isinstance(cols[0], (list, set)):
        cols = tuple(cols[0])
    if not cols:
        raise ValueError("struct requires at least one column")

    names = [_struct_field_name(c, i) for i, c in enumerate(cols)]
    evals = [_as_column(c)._eval for c in cols]

    def _ev(d):
        n = _row_count(d)
        per_col = [e(d) for e in evals]
        return [
            {names[k]: per_col[k][i] for k in range(len(per_col))}
            for i in range(n)
        ]

    return Column(_ev)


def create_map(*cols: Any) -> Column:
    """Build a ``dict`` per row from alternating ``k1, v1, k2, v2, ...`` args.

    A null key is preserved (Spark behaviour: maps may contain a null key);
    duplicate keys overwrite earlier entries with the later value, matching
    Spark's ``map`` constructor.
    """
    if len(cols) % 2 != 0:
        raise ValueError("create_map requires an even number of arguments (key, value pairs)")
    if not cols:
        def _ev_empty(d):
            return [{} for _ in range(_row_count(d))]
        return Column(_ev_empty)

    key_evals = [_as_column(cols[i])._eval for i in range(0, len(cols), 2)]
    val_evals = [_as_column(cols[i + 1])._eval for i in range(0, len(cols), 2)]

    def _ev(d):
        n = _row_count(d)
        keys_per_pair = [e(d) for e in key_evals]
        vals_per_pair = [e(d) for e in val_evals]
        out = []
        for i in range(n):
            row: dict = {}
            for p in range(len(key_evals)):
                row[keys_per_pair[p][i]] = vals_per_pair[p][i]
            out.append(row)
        return out

    return Column(_ev)


def map_keys(col_name: Union[str, Column]) -> Column:
    """Return the list of keys from a map column. Null map → null."""
    src = _as_column(col_name)._eval

    def _ev(d):
        out = []
        for v in src(d):
            if v is None:
                out.append(None)
            elif isinstance(v, dict):
                out.append(list(v.keys()))
            elif isinstance(v, list):
                # ``list[{"key": ..., "value": ...}, ...]`` legacy encoding.
                ks = []
                for entry in v:
                    if isinstance(entry, dict) and "key" in entry:
                        ks.append(entry["key"])
                    elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
                        ks.append(entry[0])
                out.append(ks)
            else:
                out.append(None)
        return out

    return Column(_ev)


def map_from_entries(col_name: Union[str, Column]) -> Column:
    """Convert an array of two-field structs into a map (``dict``).

    Each input element should be a ``dict`` with ``key`` / ``value`` keys, or a
    two-element tuple ``(key, value)``. Empty array → ``{}``. Null array → null.
    """
    src = _as_column(col_name)._eval

    def _ev(d):
        out = []
        for v in src(d):
            if v is None:
                out.append(None)
            elif isinstance(v, list):
                m: dict = {}
                for entry in v:
                    if isinstance(entry, dict) and "key" in entry:
                        m[entry["key"]] = entry.get("value")
                    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        m[entry[0]] = entry[1]
                out.append(m)
            else:
                out.append(None)
        return out

    return Column(_ev)


def explode(col_name: Union[str, Column]) -> Column:
    """Spark ``explode`` for array columns. **Row-multiplying**.

    Returns a Column carrying ``_is_explode`` metadata; the DataFrame's
    ``withColumn`` / ``select`` detects this and expands the row count.

    Note: this implementation matches Polars' ``explode``, which yields one
    null row per empty / null source array (i.e. Spark's ``explode_outer``
    semantics). The polarsdf backend behaves identically; the strict Spark
    ``explode`` (which drops empties) is intentionally **not** matched here so
    the two backends remain in lockstep.
    """
    src_col = _as_column(col_name)
    src_eval = src_col._eval

    def _ev(d):
        # Standalone evaluation (e.g. inside another expression) returns the
        # raw per-row arrays unchanged — only DataFrame.select/withColumn knows
        # how to expand the row count.
        return src_eval(d)

    out = Column(_ev)
    out._is_explode = True
    out._explode_source_eval = src_eval
    # Carry the source name when available so DataFrame.withColumn can rewrite
    # the column in place rather than appending a new exploded column.
    if isinstance(col_name, str):
        out._explode_source_name = col_name
    elif isinstance(col_name, Column) and col_name.name:
        out._explode_source_name = col_name.name
    else:
        out._explode_source_name = None
    return out


# ---------------------------------------------------------------------------
# JSON functions
# ---------------------------------------------------------------------------

# Matches a path token: either ``.name`` / ``name``, ``[N]`` (integer index),
# or ``['key']`` / ``["key"]`` (bracketed string key). Used by
# ``_parse_json_path`` to tokenize a Spark-style JSONPath into a sequence of
# field names (str) and array indices (int).
_PATH_TOKEN_RE = re.compile(
    r"""
    \.(?P<dot_name>[A-Za-z_][A-Za-z0-9_]*)   # .name
    | \[\s*(?P<idx>-?\d+)\s*\]               # [N]
    | \[\s*'(?P<sq_key>[^']*)'\s*\]          # ['key']
    | \[\s*"(?P<dq_key>[^"]*)"\s*\]          # ["key"]
    """,
    re.VERBOSE,
)


def _parse_json_path(path: str) -> Optional[list]:
    """Tokenize a Spark-style JSONPath into a list of (str | int) accessors.

    Returns ``None`` when the path is not a valid Spark JSONPath. Spark accepts
    a strict subset:

    - Must start with ``$``.
    - ``$.name`` — field access.
    - ``$['name']`` — bracketed field access (supports keys with non-identifier chars).
    - ``$[N]`` — array index.
    - Chained: ``$.a.b[0]['k'].c``.
    """
    if not isinstance(path, str) or not path.startswith("$"):
        return None
    rest = path[1:]
    tokens: list = []
    pos = 0
    while pos < len(rest):
        m = _PATH_TOKEN_RE.match(rest, pos)
        if m is None:
            return None
        if m.group("dot_name") is not None:
            tokens.append(m.group("dot_name"))
        elif m.group("idx") is not None:
            tokens.append(int(m.group("idx")))
        elif m.group("sq_key") is not None:
            tokens.append(m.group("sq_key"))
        else:
            tokens.append(m.group("dq_key"))
        pos = m.end()
    return tokens


def _atom_to_json_string(value: Any) -> Optional[str]:
    """Render a JSON atom (or container) as a Spark ``get_json_object`` result.

    Spark returns the original substring for atomic values (numbers/booleans
    serialize back without quotes; strings unquoted) and the JSON form for
    containers.
    """
    if value is None:
        # Spark renders the JSON null as the string "null" when found at the
        # target path (vs. returning SQL NULL for a missing path).
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # JSON's repr matches Spark's textual form for typical integers/floats.
        return _json.dumps(value)
    if isinstance(value, str):
        return value
    # dict / list — return canonical JSON.
    return _json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def get_json_object(col_name: Union[str, Column], path: str) -> Column:
    """Spark ``get_json_object``: extract a value from a JSON string via a path.

    Returns the extracted value as a string (or null if the path is missing,
    the input is null, or the JSON is malformed). Mirrors Spark's behaviour:
    atomic non-string values are returned as their JSON textual form (e.g. the
    number ``42`` becomes ``"42"``); objects and arrays are returned as
    canonical JSON strings.

    Path syntax (subset of JSONPath): ``$``, ``.name``, ``[N]``, ``['name']``.
    """
    tokens = _parse_json_path(path)
    src = _as_column(col_name)._eval

    def _one(raw: Any) -> Optional[str]:
        if raw is None or tokens is None:
            return None
        if isinstance(raw, str):
            try:
                value: Any = _json.loads(raw)
            except (ValueError, TypeError):
                return None
        elif isinstance(raw, (dict, list)):
            # Already-parsed JSON values are tolerated (Spark would reject these,
            # but matching the polarsdf behaviour keeps the two backends in lockstep).
            value = raw
        else:
            return None

        for tok in tokens:
            if isinstance(tok, int):
                if not isinstance(value, list):
                    return None
                idx = tok if tok >= 0 else len(value) + tok
                if idx < 0 or idx >= len(value):
                    return None
                value = value[idx]
            else:
                if not isinstance(value, dict) or tok not in value:
                    return None
                value = value[tok]

        return _atom_to_json_string(value)

    return Column(lambda d: [_one(v) for v in src(d)])


def _ddl_split_top_level(s: str, sep: str) -> list:
    """Split ``s`` on ``sep`` chars that are at angle-bracket depth 0.

    Used to split struct field declarations (``"a INT, b STRING"``) and
    map key/value types (``"map<string, int>"``) without splitting inside
    nested ``array<...>`` / ``map<...>`` / ``struct<...>``.
    """
    parts: list = []
    depth = 0
    buf: list = []
    for ch in s:
        if ch == "<":
            depth += 1
            buf.append(ch)
        elif ch == ">":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _ddl_to_datatype(schema: str) -> DataType:
    """Parse a Spark DDL-style schema string into a ``DataType``.

    Supported grammar:

    - Primitive type names (``int``, ``string``, ``decimal(10,2)``, ...).
    - ``array<inner>``.
    - ``map<key, value>``.
    - Struct DDL: ``"a INT, b STRING"`` (PySpark's StructType.fromDDL form).
    - Explicit struct syntax: ``struct<a:int, b:string>``.
    """
    normalized = schema.strip()
    lowered = normalized.lower()

    if lowered.startswith("array<") and normalized.endswith(">"):
        inner = normalized[6:-1].strip()
        return ArrayType(_ddl_to_datatype(inner))

    if lowered.startswith("map<") and normalized.endswith(">"):
        inner = normalized[4:-1].strip()
        kv = _ddl_split_top_level(inner, ",")
        if len(kv) != 2:
            raise ValueError(f"Invalid map schema: {schema!r}")
        return MapType(_ddl_to_datatype(kv[0]), _ddl_to_datatype(kv[1]))

    if lowered.startswith("struct<") and normalized.endswith(">"):
        inner = normalized[7:-1].strip()
        fields = []
        for piece in _ddl_split_top_level(inner, ","):
            name, _, type_str = piece.partition(":")
            if not type_str:
                raise ValueError(f"Invalid struct field: {piece!r}")
            fields.append(StructField(name.strip(), _ddl_to_datatype(type_str.strip())))
        return StructType(fields)

    # Plain struct DDL form ("a INT, b STRING") — comma-separated `name TYPE` pairs.
    pieces = _ddl_split_top_level(normalized, ",")
    if len(pieces) > 1 or (len(pieces) == 1 and " " in pieces[0]):
        fields = []
        for piece in pieces:
            tokens = piece.strip().split(None, 1)
            if len(tokens) != 2:
                # Single token: fall through to primitive lookup below.
                if len(pieces) == 1:
                    break
                raise ValueError(f"Invalid struct field declaration: {piece!r}")
            field_name, field_type = tokens
            fields.append(StructField(field_name, _ddl_to_datatype(field_type)))
        if fields:
            return StructType(fields)

    return spark_type_name_to_type(normalized)


def from_json(col_name: Union[str, Column], schema: Union[DataType, str]) -> Column:
    """Spark ``from_json``: parse a JSON string column to a struct/array/map.

    On a parse error (or null input) the result is null. The output is
    coerced through ``schema._cast(value, strict=False)`` so individual fields
    that don't match the schema become null rather than raising.
    """
    if isinstance(schema, str):
        parsed_schema: DataType = _ddl_to_datatype(schema)
    elif isinstance(schema, DataType):
        parsed_schema = schema
    else:
        raise TypeError(f"from_json schema must be DataType or str, got {type(schema).__name__}")

    src = _as_column(col_name)._eval

    def _one(raw: Any) -> Any:
        if raw is None:
            return None
        if isinstance(raw, (dict, list)):
            value = raw
        elif isinstance(raw, str):
            try:
                value = _json.loads(raw)
            except (ValueError, TypeError):
                return None
        else:
            return None
        try:
            return parsed_schema._cast(value, strict=False)
        except Exception:
            return None

    return Column(lambda d: [_one(v) for v in src(d)])


def _json_default(value: Any) -> Any:
    """``json.dumps`` ``default=`` hook: stringify dates / timestamps / decimals.

    Spark's ``to_json`` emits dates as ``"yyyy-MM-dd"`` and timestamps as
    ``"yyyy-MM-dd HH:mm:ss[.SSS]"``; ``Decimal`` is rendered as its plain
    string representation (no scientific notation).
    """
    if isinstance(value, _dt.datetime):
        if value.microsecond:
            return value.strftime("%Y-%m-%d %H:%M:%S") + f".{value.microsecond // 1000:03d}"
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _strip_nulls(value: Any) -> Any:
    """Recursively drop keys whose value is None (Spark default ``ignoreNullFields=True``).

    Lists/tuples are walked but their None elements are preserved (Spark keeps
    null array elements, only struct/map nulls get dropped).
    """
    if isinstance(value, dict):
        return {k: _strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_strip_nulls(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Window ranking factories (must be used with .over(WindowSpec))
# ---------------------------------------------------------------------------


class _RankColumn(Column):
    """Placeholder Column for a ranking function.

    Calling ``.over(spec)`` on the returned object dispatches into
    :meth:`Column.over` which sees the ``_window_rank_kind`` marker and computes
    the appropriate ranking. The bare Column has no meaningful standalone
    evaluation — using it outside ``.over(...)`` raises.
    """

    def __init__(self, kind: str):
        def _ev(d):
            raise ValueError(
                f"{kind}() must be used inside .over(window) — it has no standalone value"
            )

        super().__init__(_ev, name=f"{kind}()")
        self._window_rank_kind = kind


def rank() -> Column:
    """Spark ``rank``: 1-based rank with gaps after ties (1, 2, 2, 4)."""
    return _RankColumn("rank")


def dense_rank() -> Column:
    """Spark ``dense_rank``: 1-based rank, no gaps (1, 2, 2, 3)."""
    return _RankColumn("dense_rank")


def row_number() -> Column:
    """Spark ``row_number``: 1-based positional index within the window partition."""
    return _RankColumn("row_number")


def to_json(col_name: Union[str, Column], options: Any = None) -> Column:
    """Spark ``to_json``: serialize a struct / array / map column to a JSON string.

    Only ``options={"ignoreNullFields": False}`` is supported from Spark's
    options map (matches the polarsdf backend). Per Spark's default,
    null struct/map fields are dropped from the output.
    """
    if options is not None and options != {"ignoreNullFields": False}:
        raise ValueError('pythondf to_json only supports options={"ignoreNullFields": False}')
    drop_nulls = options is None  # default ignoreNullFields=True

    src = _as_column(col_name)._eval

    def _one(value: Any) -> Optional[str]:
        if value is None:
            return None
        v = _strip_nulls(value) if drop_nulls else value
        return _json.dumps(v, separators=(",", ":"), ensure_ascii=False, default=_json_default)

    return Column(lambda d: [_one(v) for v in src(d)])


def uuid() -> Column:
    """Mimics :func:`pyspark.sql.functions.uuid`.

    One random canonical UUID v4 string per row via :func:`uuid.uuid4`.
    """

    def _ev(d):
        n = _row_count(d)
        return [str(_uuid.uuid4()) for _ in builtins.range(n)]

    return Column(_ev)


def rand(seed: Optional[int] = None) -> Column:
    """Mimics :func:`pyspark.sql.functions.rand`: uniform random in ``[0.0, 1.0)`` per row.

    With ``seed``, draws are deterministic per row index within a process. Values do
    not match Spark's JVM PRNG for the same seed.
    """

    def _ev(d):
        n = _row_count(d)
        rng = _random.Random(seed) if seed is not None else _random.Random()
        return [rng.random() for _ in builtins.range(n)]

    return Column(_ev)


def monotonically_increasing_id() -> Column:
    """Mimics :func:`pyspark.sql.functions.monotonically_increasing_id` for a single
    in-memory partition: yields 0, 1, 2, … in current row order as Int64.
    """

    def _ev(d):
        n = _row_count(d)
        return list(builtins.range(n))

    return Column(_ev)


def broadcast(df: Any) -> Any:
    """Mimics :func:`pyspark.sql.functions.broadcast`. No-op for in-process backends."""
    return df
