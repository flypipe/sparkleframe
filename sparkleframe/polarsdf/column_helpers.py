from __future__ import annotations

import contextvars
import hashlib
import re
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Generator, Optional, Union

import polars as pl

_polars_schema_ctx: contextvars.ContextVar[Optional[pl.Schema]] = contextvars.ContextVar(
    "sparkleframe_polars_schema", default=None
)


@contextmanager
def _polars_schema_for(schema: pl.Schema) -> Generator[None, None, None]:
    """Set the active Polars frame schema so Column.getItem can resolve struct/list types."""
    token = _polars_schema_ctx.set(schema)
    try:
        yield
    finally:
        _polars_schema_ctx.reset(token)


def _output_dtype_of_expr(expr: pl.Expr, schema: pl.Schema) -> Optional[pl.DataType]:
    try:
        return pl.LazyFrame(schema=schema).select(expr.alias("_x")).collect_schema()["_x"]
    except Exception:
        return None


def _resolve_expr_output_dtype(expr: pl.Expr) -> Optional[pl.DataType]:
    sch = _polars_schema_ctx.get()
    if sch is not None:
        d = _output_dtype_of_expr(expr, sch)
        if d is not None:
            return d
    try:
        meta = expr.meta
        if hasattr(meta, "output_dtype"):
            return meta.output_dtype()  # type: ignore[no-any-return]
    except Exception:
        pass
    return None


_NUMERIC_ORDER_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
)


def _is_numeric_polars_dtype(dt: Optional[pl.DataType]) -> bool:
    if dt is None:
        return False
    if isinstance(dt, pl.Decimal):
        return True
    return dt in _NUMERIC_ORDER_DTYPES


_SPARK_NUMERIC_RANK: dict[pl.DataType, int] = {
    pl.Int8: 0,
    pl.Int16: 1,
    pl.Int32: 2,
    pl.Int64: 3,
    pl.Float32: 4,
    pl.Float64: 5,
}


_SPARK_INTEGER_DTYPES = frozenset({pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64})
_SPARK_FLOAT_DTYPES = frozenset({pl.Float32, pl.Float64})


def _spark_compare_group(dt: Optional[pl.DataType]) -> Optional[str]:
    """Return the Spark type-compatibility group for comparisons, or ``None`` if unknown.

    Uses fine-grained groups so the cross-group allowlist can distinguish
    integer-string (Spark raises) from float-string (Spark coerces).
    """
    if dt is None:
        return None
    if dt in _SPARK_INTEGER_DTYPES:
        return "integer"
    if dt in _SPARK_FLOAT_DTYPES:
        return "float"
    if isinstance(dt, pl.Decimal):
        return "decimal"
    if isinstance(dt, (pl.Datetime, pl.Duration)):
        return "temporal"
    if dt == pl.Date:
        return "temporal"
    if dt in (pl.Utf8, pl.String):
        return "string"
    if dt == pl.Boolean:
        return "boolean"
    if dt == pl.Binary:
        return "binary"
    if _is_complex_polars_dtype(dt):
        return "complex"
    return None


def _expr_is_literal(e: pl.Expr) -> bool:
    """Return ``True`` when the expression has no column references (is a literal / constant)."""
    try:
        return len(e.meta.root_names()) == 0
    except Exception:
        return False


_SPARK_COMPARE_COMPATIBLE_PAIRS = frozenset(
    {
        ("integer", "integer"),
        ("integer", "float"),
        ("integer", "decimal"),
        ("float", "integer"),
        ("float", "float"),
        ("float", "decimal"),
        ("float", "string"),
        ("decimal", "integer"),
        ("decimal", "float"),
        ("decimal", "decimal"),
        ("decimal", "string"),
        ("temporal", "temporal"),
        ("string", "float"),
        ("string", "decimal"),
        ("string", "string"),
        ("string", "binary"),
        ("boolean", "boolean"),
        ("binary", "string"),
        ("binary", "binary"),
        ("complex", "complex"),
    }
)


def _assert_spark_compare_compatible_eager(
    op: str,
    left_dt: pl.DataType,
    right_dt: pl.DataType,
) -> None:
    """Raise ``TypeError`` when the two concrete dtypes are not comparable in Spark."""
    lg = _spark_compare_group(left_dt)
    rg = _spark_compare_group(right_dt)
    if lg is None or rg is None:
        return
    if (lg, rg) in _SPARK_COMPARE_COMPATIBLE_PAIRS:
        return
    display = _OP_DISPLAY.get(op, op)
    raise TypeError(
        f'Cannot resolve "{display}" due to data type mismatch: '
        f"differing types in '{display}' ({left_dt!r} and {right_dt!r})"
    )


def _assert_spark_compare_compatible(
    op: str,
    left: pl.Expr,
    right: pl.Expr,
    left_dt: Optional[pl.DataType],
    right_dt: Optional[pl.DataType],
) -> None:
    """Raise ``TypeError`` when the two operands' dtypes belong to different Spark comparison groups.

    Spark 4 raises ``AnalysisException`` for cross-group column-column comparisons
    (e.g. ``col(int) == col(boolean)``, ``col(float) < col(date)``).  We mirror that.

    When one side is a literal (no column references), Spark applies implicit cast
    rules, so we skip the check — matching Spark's ``col(int) == lit("1")`` behaviour.

    When either dtype is ``None`` (unknown at build time) we skip — the deferred
    runtime check via :func:`_spark_compare_guard_expr` will handle it.
    """
    lg = _spark_compare_group(left_dt)
    rg = _spark_compare_group(right_dt)
    if lg is None or rg is None:
        return
    if (lg, rg) in _SPARK_COMPARE_COMPATIBLE_PAIRS:
        return
    if _expr_is_literal(left) or _expr_is_literal(right):
        return
    display = _OP_DISPLAY.get(op, op)
    raise TypeError(
        f'Cannot resolve "{display}" due to data type mismatch: '
        f"differing types in '{display}' ({left_dt!r} and {right_dt!r})"
    )


def _promote_date_to_datetime_pair(left_s: pl.Series, right_s: pl.Series) -> tuple[pl.Series, pl.Series]:
    """If one side is Date and the other Datetime, cast Date to Datetime (midnight).

    Preserves the timezone of the Datetime operand so the two sides are comparable.
    """
    ld, rd = left_s.dtype, right_s.dtype
    if ld == pl.Date and isinstance(rd, pl.Datetime):
        left_s = left_s.cast(pl.Datetime(rd.time_unit or "us"), strict=False)
        if rd.time_zone:
            left_s = left_s.dt.replace_time_zone(rd.time_zone)
    elif isinstance(ld, pl.Datetime) and rd == pl.Date:
        right_s = right_s.cast(pl.Datetime(ld.time_unit or "us"), strict=False)
        if ld.time_zone:
            right_s = right_s.dt.replace_time_zone(ld.time_zone)
    return left_s, right_s


def _compare_series(left_s: pl.Series, right_s: pl.Series, op: str) -> pl.Series:
    """Apply a comparison operator to two Series."""
    if op in ("eq", "=="):
        return left_s == right_s
    if op in ("ne", "!="):
        return left_s != right_s
    if op in ("lt", "<"):
        return left_s < right_s
    if op in ("le", "<="):
        return left_s <= right_s
    if op in ("gt", ">"):
        return left_s > right_s
    if op in ("ge", ">="):
        return left_s >= right_s
    raise ValueError(op)


def _needs_date_datetime_promotion(left_dt: pl.DataType, right_dt: pl.DataType) -> bool:
    left_date = left_dt == pl.Date
    right_date = right_dt == pl.Date
    left_datetime = isinstance(left_dt, pl.Datetime)
    right_datetime = isinstance(right_dt, pl.Datetime)
    return (left_date and right_datetime) or (left_datetime and right_date)


def _make_compare_type_guard(op: str):
    """Return a ``map_batches`` callback that validates dtype compatibility at runtime.

    When one operand is Date and the other Datetime, promotes Date to Datetime
    (midnight) and re-computes the comparison — matching Spark's implicit promotion.
    """

    def _apply_guard(s: pl.Series) -> pl.Series:
        struct_df = s.struct.unnest()
        left_s = struct_df.get_column("_cmp_left")
        right_s = struct_df.get_column("_cmp_right")
        result_s = struct_df.get_column("_cmp_result")
        _assert_spark_compare_compatible_eager(op, left_s.dtype, right_s.dtype)
        if _needs_date_datetime_promotion(left_s.dtype, right_s.dtype):
            left_s, right_s = _promote_date_to_datetime_pair(left_s, right_s)
            return _compare_series(left_s, right_s, op)
        return result_s

    return _apply_guard


def _spark_compare_guard_expr(
    left: pl.Expr,
    right: pl.Expr,
    op: str,
    result_expr: pl.Expr,
    left_dt: Optional[pl.DataType],
    right_dt: Optional[pl.DataType],
) -> pl.Expr:
    """Wrap *result_expr* with a runtime dtype-compatibility check.

    When both dtypes are already resolved at build time, the static check has already
    run and we return the result as-is.  Otherwise, we prepend a ``map_batches`` guard
    that inspects the actual dtypes at evaluation time and raises ``TypeError`` for
    cross-group comparisons (mirroring Spark's ``AnalysisException``).

    Literals (no column roots) are always allowed through — Spark's implicit-cast rules
    accept them.
    """
    if left_dt is not None and right_dt is not None:
        return result_expr
    if _expr_is_literal(left) or _expr_is_literal(right):
        return result_expr

    return pl.struct(
        left.alias("_cmp_left"),
        right.alias("_cmp_right"),
        result_expr.alias("_cmp_result"),
    ).map_batches(_make_compare_type_guard(op), return_dtype=pl.Boolean)


def _is_complex_polars_dtype(dt: Optional[pl.DataType]) -> bool:
    if dt is None:
        return False
    return isinstance(dt, (pl.List, pl.Struct, pl.Array))


def _is_string_polars_dtype(dt: Optional[pl.DataType]) -> bool:
    if dt is None:
        return False
    return dt in (pl.Utf8, pl.String)


def _is_date_polars_dtype(dt: Optional[pl.DataType]) -> bool:
    if dt is None:
        return False
    return dt == pl.Date


def _is_integer_polars_dtype(dt: Optional[pl.DataType]) -> bool:
    if dt is None:
        return False
    return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def _coerce_mixed_arithmetic_operands(
    left: pl.Expr,
    right: pl.Expr,
    ld: Optional[pl.DataType],
    rd: Optional[pl.DataType],
    op: str,
) -> Optional[tuple[pl.Expr, pl.Expr]]:
    """Pre-cast operands for Spark-compatible mixed-type arithmetic.

    Returns ``(left, right)`` with appropriate casts when cross-type coercion is
    needed, or ``None`` when the standard per-operand validation should run as-is.

    Handles:
    - ``numeric op string`` and ``string op numeric``: cast string to Float64 (Spark
      auto-promotes numeric strings for ``+``, ``-``, ``*``, ``/``).
    - ``int op date`` for ``+`` and ``-``: Spark adds/subtracts days (produces Date).

    .. note::

       This function bails out when either dtype is ``None`` (unresolvable at
       expression build time).  For bare ``pl.col("name")`` references the dtype
       is only known once a DataFrame schema is available — which happens later,
       inside ``DataFrame.select()`` etc.  This is a known gap; see
       ``docs/known_gaps.md`` for details and possible fix paths.
    """
    if ld is None or rd is None:
        return None

    left_numeric = _is_numeric_polars_dtype(ld)
    right_numeric = _is_numeric_polars_dtype(rd)
    left_string = _is_string_polars_dtype(ld)
    right_string = _is_string_polars_dtype(rd)

    if left_numeric and right_string:
        return left.cast(pl.Float64, strict=False), right.cast(pl.Float64, strict=False)
    if left_string and right_numeric:
        return left.cast(pl.Float64, strict=False), right.cast(pl.Float64, strict=False)

    if op in ("+", "-"):
        left_int = _is_integer_polars_dtype(ld)
        right_int = _is_integer_polars_dtype(rd)
        left_date = _is_date_polars_dtype(ld)
        right_date = _is_date_polars_dtype(rd)
        if left_int and right_date:
            days = left.cast(pl.Int64, strict=False).cast(pl.Duration("ms"), strict=False) * 86_400_000
            return days, right.cast(pl.Datetime("ms"), strict=False)
        if left_date and right_int:
            days = right.cast(pl.Int64, strict=False).cast(pl.Duration("ms"), strict=False) * 86_400_000
            return left.cast(pl.Datetime("ms"), strict=False), days

    return None


_NON_ARITHMETIC_DTYPES = frozenset({pl.Boolean, pl.Utf8, pl.Binary, pl.String})


def _assert_arithmetic_compatible(dt: Optional[pl.DataType]) -> None:
    """Raise TypeError for non-numeric Polars dtypes, matching Spark's AnalysisException."""
    if dt is None:
        return
    if _is_numeric_polars_dtype(dt):
        return
    if isinstance(dt, (pl.Duration, pl.Datetime)):
        return
    if dt in (pl.Date, pl.Null):
        return
    if dt in _NON_ARITHMETIC_DTYPES:
        raise TypeError(f"Cannot resolve arithmetic operation with {dt} type")
    if _is_complex_polars_dtype(dt):
        raise TypeError(f"Cannot resolve arithmetic operation with {dt} type")


def _assert_arithmetic_series(s: pl.Series) -> pl.Series:
    """
    Runtime guard for +, -, *: raise for non-arithmetic dtypes (boolean, string,
    binary, complex) to mirror Spark's ``AnalysisException``. Returning the
    series unchanged preserves the natural Polars output type (e.g. ``Int32 +
    Int32`` stays ``Int32``).
    """
    if s.dtype in _NON_ARITHMETIC_DTYPES:
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    if _is_complex_polars_dtype(s.dtype):
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    return s


def _validate_and_cast_float64_strict(s: pl.Series) -> pl.Series:
    """
    Runtime guard for ``/``: Spark rejects string, boolean, binary, date /
    timestamp and complex operands; we mirror that and only allow numerics
    (cast to Float64).
    """
    if s.dtype in _NON_ARITHMETIC_DTYPES:
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    if s.dtype == pl.Binary:
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    if isinstance(s.dtype, (pl.Date, pl.Datetime, pl.Duration)):
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    if _is_complex_polars_dtype(s.dtype):
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    return s.cast(pl.Float64, strict=False)


def _validate_and_cast_float64_lenient(s: pl.Series) -> pl.Series:
    """
    Runtime guard for ``**``: Spark's ``pow`` auto-promotes strings to double, so
    we allow string operands (cast best-effort to Float64). Reject types Spark
    itself refuses (boolean, binary, date / timestamp, complex).
    """
    if s.dtype == pl.Boolean:
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    if s.dtype == pl.Binary:
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    if isinstance(s.dtype, (pl.Date, pl.Datetime, pl.Duration)):
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    if _is_complex_polars_dtype(s.dtype):
        raise TypeError(f"Cannot resolve arithmetic operation with {s.dtype} type")
    return s.cast(pl.Float64, strict=False)


def _validated_arithmetic_expr(expr: pl.Expr, resolved_dt: Optional[pl.DataType]) -> pl.Expr:
    """
    For +, -, *: validate if dtype is known at build time; otherwise embed a
    runtime check via ``map_batches`` (without ``return_dtype`` so Polars keeps
    the natural output type, e.g. ``Int32 + Int32`` stays ``Int32``).
    """
    if resolved_dt is not None:
        _assert_arithmetic_compatible(resolved_dt)
        return expr
    return expr.map_batches(_assert_arithmetic_series)


def _has_decimal_operand(left_dt: Optional[pl.DataType], right_dt: Optional[pl.DataType]) -> bool:
    """Return ``True`` when at least one resolved dtype is Decimal."""
    return isinstance(left_dt, pl.Decimal) or isinstance(right_dt, pl.Decimal)


def _spark_decimal_div_result_type(left_dt: pl.DataType, right_dt: pl.DataType) -> pl.Decimal:
    """Compute the Spark result Decimal type for ``left / right``.

    Spark's formula (``DecimalType.divideResultType``):
      result precision = p1 - s1 + s2 + max(6, s1 + p2 + 1)
      result scale     = max(6, s1 + p2 + 1)
    with the result capped at precision 38.

    For non-Decimal operands promoted to Decimal we use (20, 0) as the implicit
    type (matches Spark's promotion of Long to Decimal(20, 0)).
    """
    if isinstance(left_dt, pl.Decimal):
        p1, s1 = left_dt.precision or 38, left_dt.scale or 0
    else:
        p1, s1 = 20, 0
    if isinstance(right_dt, pl.Decimal):
        p2, s2 = right_dt.precision or 38, right_dt.scale or 0
    else:
        p2, s2 = 20, 0
    result_scale = max(6, s1 + p2 + 1)
    result_precision = min(38, p1 - s1 + s2 + result_scale)
    return pl.Decimal(precision=result_precision, scale=result_scale)


def _validated_decimal_div_expr(expr: pl.Expr, resolved_dt: Optional[pl.DataType]) -> pl.Expr:
    """For ``/`` with Decimal operand: validate and cast non-Decimal side to Decimal."""
    if resolved_dt is not None:
        _assert_arithmetic_compatible(resolved_dt)
        if not isinstance(resolved_dt, pl.Decimal):
            return expr.cast(pl.Decimal(precision=20, scale=0), strict=False)
        return expr
    return expr.map_batches(_validate_and_cast_float64_strict, return_dtype=pl.Float64)


def _validated_float64_expr(expr: pl.Expr, resolved_dt: Optional[pl.DataType]) -> pl.Expr:
    """
    For ``/``: validate strictly (reject string / boolean) and cast to Float64.
    Spark rejects string / boolean operands for ``/`` even though ``pow`` accepts
    them, so we keep the strict semantics here.
    """
    if resolved_dt is not None:
        _assert_arithmetic_compatible(resolved_dt)
        return expr.cast(pl.Float64, strict=False)
    return expr.map_batches(_validate_and_cast_float64_strict, return_dtype=pl.Float64)


def _validated_pow_expr(expr: pl.Expr, resolved_dt: Optional[pl.DataType]) -> pl.Expr:
    """
    For ``**``: Spark's ``pow`` auto-promotes string operands; only reject types
    Spark itself rejects (binary, date / timestamp, complex).
    """
    if resolved_dt is not None and resolved_dt == pl.Binary:
        raise TypeError(f"Cannot resolve arithmetic operation with {resolved_dt} type")
    if resolved_dt is not None and isinstance(resolved_dt, (pl.Date, pl.Datetime, pl.Duration)):
        raise TypeError(f"Cannot resolve arithmetic operation with {resolved_dt} type")
    if resolved_dt is not None and _is_complex_polars_dtype(resolved_dt):
        raise TypeError(f"Cannot resolve arithmetic operation with {resolved_dt} type")
    if resolved_dt is not None:
        return expr.cast(pl.Float64, strict=False)
    return expr.map_batches(_validate_and_cast_float64_lenient, return_dtype=pl.Float64)


def _spark_numeric_widened_type(
    left_dt: Optional[pl.DataType], right_dt: Optional[pl.DataType]
) -> Optional[pl.DataType]:
    """
    Return the Spark-like widened numeric type for two operands.
    Returns None if no cast is needed (same type or unknown).
    """
    if left_dt is None or right_dt is None:
        return pl.Float64
    if isinstance(left_dt, pl.Decimal) or isinstance(right_dt, pl.Decimal):
        return pl.Float64
    left_rank = _SPARK_NUMERIC_RANK.get(left_dt)
    right_rank = _SPARK_NUMERIC_RANK.get(right_dt)
    if left_rank is None or right_rank is None:
        return pl.Float64
    if left_rank == right_rank:
        return None
    return left_dt if left_rank > right_rank else right_dt


def _cmp_exprs(left: pl.Expr, right: pl.Expr, op: str) -> pl.Expr:
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    raise ValueError(op)


_DT_LIKE_STRING = re.compile(r"\d{4}.*[-/:T]")


def _parse_datetime_string_safe(v: str) -> Optional[datetime]:
    if not _DT_LIKE_STRING.search(v):
        return None
    try:
        s = pl.Series("_v", [v], dtype=pl.Utf8)
        parsed = s.str.to_datetime(strict=False)
        if parsed.is_null().all():
            return None
        ts = parsed.dt.replace_time_zone(None)[0]
        return ts  # type: ignore[no-any-return]
    except Exception:
        return None


_ORDERING_OPS = frozenset({"lt", "le", "gt", "ge"})

_OP_DISPLAY = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}


def _looks_like_map_dtype(dt: Optional[pl.DataType]) -> bool:
    """
    PySpark ``MapType`` round-trips to Polars as ``List(Struct([key, value]))``. We can't
    distinguish it from a literal list-of-struct, but a struct with exactly the fields
    ``key`` / ``value`` is the canonical map shape -- treat it as map-like so we can give
    a more specific error message.
    """
    if not isinstance(dt, pl.List):
        return False
    inner = dt.inner
    if not isinstance(inner, pl.Struct):
        return False
    field_names = {f.name for f in inner.fields}
    return field_names == {"key", "value"}


def _format_unsupported_complex_compare(op: str, left_dt: Any, right_dt: Any) -> str:
    operator = _OP_DISPLAY.get(op, op)
    if _looks_like_map_dtype(left_dt) or _looks_like_map_dtype(right_dt):
        return (
            f"sparkleframe does not support the '{operator}' operator on MapType columns yet "
            f"(Polars stores maps as List(Struct([key, value])), which prevents native dispatch). "
            f"Got operands of dtype {left_dt!r} and {right_dt!r}."
        )
    return (
        f"sparkleframe does not support the '{operator}' operator on List / Struct / Array "
        f"columns yet (Polars does not implement lexicographic ordering for nested dtypes). "
        f"Got operands of dtype {left_dt!r} and {right_dt!r}."
    )


def _assert_complex_compare_supported(
    op: str, left_dt: Optional[pl.DataType], right_dt: Optional[pl.DataType]
) -> None:
    """
    Raise :class:`NotImplementedError` for combinations sparkleframe can't replicate from
    Spark today: ordering on List / Struct / Array, and any comparison on Map-like dtypes.
    Equality / inequality on plain List / Struct / Array is supported via Polars natives.
    """
    left_complex = _is_complex_polars_dtype(left_dt)
    right_complex = _is_complex_polars_dtype(right_dt)
    if not (left_complex or right_complex):
        return
    is_map = _looks_like_map_dtype(left_dt) or _looks_like_map_dtype(right_dt)
    is_ordering = op in _ORDERING_OPS
    if is_map or is_ordering:
        raise NotImplementedError(_format_unsupported_complex_compare(op, left_dt, right_dt))


def _series_is_complex(s: pl.Series) -> pl.Series:
    """Per-batch boolean indicator: every row is True iff the input dtype is List / Struct / Array."""
    flag = _is_complex_polars_dtype(s.dtype)
    return pl.Series(s.name, [flag] * s.len(), dtype=pl.Boolean)


def _expr_is_complex_at_runtime(e: pl.Expr) -> pl.Expr:
    """Lift :func:`_series_is_complex` into an expression so it can drive ``when().then()``."""
    return e.map_batches(_series_is_complex, return_dtype=pl.Boolean)


def _native_compare_when_complex(left: pl.Expr, right: pl.Expr, op: str, fallback_expr: pl.Expr) -> pl.Expr:
    """
    Return ``fallback_expr`` except when ``left`` / ``right`` are List / Struct / Array
    at runtime, in which case use Polars' native compare operator. Implemented via
    ``map_batches`` over a packed struct so the native path is only evaluated for the
    complex case (``pl.when`` evaluates both branches eagerly and would crash on
    cross-type fallback expressions).

    Raises :class:`NotImplementedError` with a clear message for operator / dtype combos
    sparkleframe cannot match against Spark yet (ordering on nested types, any comparison
    on Map-like dtypes).
    """

    def _dispatch(s: pl.Series) -> pl.Series:
        if s.len() == 0:
            return pl.Series(s.name, [], dtype=pl.Boolean)
        left_s = s.struct.field("_l")
        right_s = s.struct.field("_r")
        fallback_s = s.struct.field("_f")
        if _is_complex_polars_dtype(left_s.dtype) or _is_complex_polars_dtype(right_s.dtype):
            _assert_complex_compare_supported(op, left_s.dtype, right_s.dtype)
            if op == "eq":
                return left_s == right_s
            if op == "ne":
                return left_s != right_s
            raise ValueError(op)
        return fallback_s

    return pl.struct([left.alias("_l"), right.alias("_r"), fallback_expr.alias("_f")]).map_batches(
        _dispatch, return_dtype=pl.Boolean
    )


def _series_to_string_compare(s: pl.Series) -> pl.Series:
    """Utf8 cast for comparisons; Object columns (e.g. ``getItem``) become ``str`` per cell.

    Complex / nested types (List / Struct / Array) can't safely string-cast; we return
    an all-null Utf8 series so the caller's ``numeric_ok`` / ``temporal_ok`` guards collapse
    to ``False`` and the runtime dispatcher can route to native Polars comparison instead.
    """
    if _is_complex_polars_dtype(s.dtype):
        return pl.Series(s.name, [None] * s.len(), dtype=pl.Utf8)
    if s.dtype == pl.Object:
        out: list[Any] = []
        for x in s.to_list():
            out.append(None if x is None else str(x))
        return pl.Series(s.name, out, dtype=pl.Utf8)
    return s.cast(pl.Utf8, strict=False)


def _expr_as_string_for_compare(e: pl.Expr) -> pl.Expr:
    return e.map_batches(_series_to_string_compare, return_dtype=pl.Utf8)


def _series_coerce_order_datetime(s: pl.Series) -> pl.Series:
    """Per-row datetime coercion for ordering (avoids Polars raising on ``str.to_datetime`` in ``when``)."""
    if s.len() == 0:
        return pl.Series(s.name, [], dtype=pl.Datetime("us"))
    out: list[Any] = []
    for v in s.to_list():
        if v is None:
            out.append(None)
        elif isinstance(v, datetime):
            ts = v
            if ts.tzinfo is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            out.append(ts)
        elif isinstance(v, date) and not isinstance(v, datetime):
            out.append(datetime(v.year, v.month, v.day))
        elif isinstance(v, str):
            out.append(_parse_datetime_string_safe(v))
        else:
            out.append(None)
    return pl.Series(s.name, out, dtype=pl.Datetime("us"))


def _coerce_expr_order_datetime(e: pl.Expr) -> pl.Expr:
    """Coerce to naive microsecond datetimes for Spark-like ordering (date / timestamp / ISO strings)."""
    return e.map_batches(_series_coerce_order_datetime, return_dtype=pl.Datetime("us"))


_BOOL_TRUE_STRINGS = frozenset({"true", "t", "1", "yes", "y"})
_BOOL_FALSE_STRINGS = frozenset({"false", "f", "0", "no", "n"})


def _parse_bool_string(value: Any) -> Optional[bool]:
    """Per-row Spark-like string -> bool: ``None`` for null / unrecognised values."""
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in _BOOL_TRUE_STRINGS:
        return True
    if lowered in _BOOL_FALSE_STRINGS:
        return False
    return None


def _string_to_bool_expr(expr: pl.Expr) -> pl.Expr:
    """
    Spark-like ``string -> Boolean`` coercion as a Polars expression: nulls stay null,
    common truthy / falsy literals (``true/false``, ``t/f``, ``1/0``, ``yes/no``,
    ``y/n``; case-insensitive, trimmed) map to ``True`` / ``False``, anything else
    becomes ``None``. Polars has no built-in strict ``Utf8 -> Boolean`` cast, so we
    parse manually with a ``when().then()`` ladder.

    This is the *lenient* helper used by :meth:`Column.try_cast`. For Spark 4 ANSI
    ``cast`` semantics (raise on malformed input), use :func:`_string_to_bool_expr_strict`.
    """
    string_expr = expr.cast(pl.String, strict=False).str.strip_chars().str.to_lowercase()
    return (
        pl.when(expr.is_null())
        .then(pl.lit(None, dtype=pl.Boolean))
        .when(string_expr.is_in(list(_BOOL_TRUE_STRINGS)))
        .then(pl.lit(True))
        .when(string_expr.is_in(list(_BOOL_FALSE_STRINGS)))
        .then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
    )


def _series_to_bool_strict(s: pl.Series) -> pl.Series:
    """
    Per-batch Spark-like strict ``-> Boolean`` cast. Strings must be one of the
    recognised literals (case-insensitive, trimmed) or the batch raises
    ``CAST_INVALID_INPUT``. Numeric / boolean inputs use Polars' native cast
    (nonzero -> true), matching Spark 4 ANSI behaviour for those source types.
    """
    if s.dtype not in (pl.Utf8, pl.String):
        return s.cast(pl.Boolean, strict=True)
    out: list[Optional[bool]] = []
    for v in s.to_list():
        if v is None:
            out.append(None)
            continue
        lowered = str(v).strip().lower()
        if lowered in _BOOL_TRUE_STRINGS:
            out.append(True)
        elif lowered in _BOOL_FALSE_STRINGS:
            out.append(False)
        else:
            raise ValueError(
                f'[CAST_INVALID_INPUT] The value {v!r} of the type "STRING" cannot be cast to '
                f'"BOOLEAN" because it is malformed. Use try_cast to tolerate malformed input and '
                f"return NULL instead."
            )
    return pl.Series(s.name, out, dtype=pl.Boolean)


def _string_to_bool_expr_strict(expr: pl.Expr) -> pl.Expr:
    """
    Spark 4 ANSI-style strict ``-> Boolean`` cast: raises ``CAST_INVALID_INPUT`` for
    any unrecognised string literal (mirroring Spark's default behaviour). Numeric /
    bool source dtypes fall through to Polars' native cast (which Spark accepts).
    For ``try_cast`` (lenient, returns ``None`` on malformed input) use
    :func:`_string_to_bool_expr`.
    """
    return expr.map_batches(_series_to_bool_strict, return_dtype=pl.Boolean)


def _numeric_compare_operands(left: pl.Expr, right: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """
    Build the four operand expressions sparkleframe uses for Spark-like cross-type
    comparisons: ``left_str`` / ``right_str`` cast through string for safe inspection,
    and ``left_num`` / ``right_num`` further cast to ``Float64`` (nulls where the
    string isn't parsable). Callers gate on ``is_not_null`` to decide whether to use
    numeric or lexicographic compare.
    """
    left_str = _expr_as_string_for_compare(left)
    right_str = _expr_as_string_for_compare(right)
    left_num = left_str.cast(pl.Float64, strict=False)
    right_num = right_str.cast(pl.Float64, strict=False)
    return left_num, right_num, left_str, right_str


def _promote_date_to_datetime_exprs(
    left: pl.Expr, right: pl.Expr, ld: Optional[pl.DataType], rd: Optional[pl.DataType]
) -> tuple[pl.Expr, pl.Expr]:
    """Cast Date to Datetime(midnight) when the other side is Datetime.

    Preserves the timezone of the Datetime operand so both sides are comparable.
    """
    if ld == pl.Date and isinstance(rd, pl.Datetime):
        left = left.cast(pl.Datetime(rd.time_unit or "us"), strict=False)
        if rd.time_zone:
            left = left.dt.replace_time_zone(rd.time_zone)
    elif isinstance(ld, pl.Datetime) and rd == pl.Date:
        right = right.cast(pl.Datetime(ld.time_unit or "us"), strict=False)
        if ld.time_zone:
            right = right.dt.replace_time_zone(ld.time_zone)
    return left, right


def _ordering_comparison_expr(left: pl.Expr, right: pl.Expr, op: str) -> pl.Expr:
    """
    Spark-like ``< <= > >=`` over Polars expressions. Numeric operands use numeric
    order; date / timestamp / parsable strings use temporal order (fixes
    ``datetime >= date_sub(current_date(), n)`` under string schemas); unknown
    dtypes prefer numeric parse then temporal then lexicographic string.

    Ordering on List / Struct / Array / Map dtypes is not supported (Polars has no
    lexicographic compare for nested dtypes); raises :class:`NotImplementedError`
    with a clear message at build time when the dtype is known, or at evaluation
    time via :func:`_native_compare_when_complex` otherwise.
    """
    ld = _resolve_expr_output_dtype(left)
    rd = _resolve_expr_output_dtype(right)
    _assert_complex_compare_supported(op, ld, rd)
    _assert_spark_compare_compatible(op, left, right, ld, rd)
    if ld is not None and rd is not None and _needs_date_datetime_promotion(ld, rd):
        left, right = _promote_date_to_datetime_exprs(left, right, ld, rd)
        return _cmp_exprs(left, right, op)
    left_num, right_num, left_str, right_str = _numeric_compare_operands(left, right)
    numeric_ok = left_num.is_not_null() & right_num.is_not_null()
    if _is_numeric_polars_dtype(ld) or _is_numeric_polars_dtype(rd):
        return _spark_compare_guard_expr(left, right, op, _cmp_exprs(left_num, right_num, op), ld, rd)

    left_dt = _coerce_expr_order_datetime(left)
    right_dt = _coerce_expr_order_datetime(right)
    temporal_ok = left_dt.is_not_null() & right_dt.is_not_null()
    if ld is not None or rd is not None:
        result = (
            pl.when(temporal_ok)
            .then(_cmp_exprs(left_dt, right_dt, op))
            .when(numeric_ok)
            .then(_cmp_exprs(left_num, right_num, op))
            .otherwise(_cmp_exprs(left_str, right_str, op))
        )
        return _spark_compare_guard_expr(left, right, op, result, ld, rd)
    coerced_expr = (
        pl.when(numeric_ok)
        .then(_cmp_exprs(left_num, right_num, op))
        .when(temporal_ok)
        .then(_cmp_exprs(left_dt, right_dt, op))
        .otherwise(_cmp_exprs(left_str, right_str, op))
    )
    result = _native_compare_when_complex(left, right, op, coerced_expr)
    return _spark_compare_guard_expr(left, right, op, result, ld, rd)


def _equality_comparison_expr(left: pl.Expr, right: pl.Expr, equal: bool) -> pl.Expr:
    """
    Spark-like ``==`` / ``!=`` over Polars expressions. Cross-type equality coerces
    through string / numeric (so ``col(int) == lit('1')`` matches Spark), while
    list / struct / array operands fall back to Polars' native equality.

    When one operand is Date and the other Datetime, promotes Date to Datetime
    (midnight) before comparing — matching Spark's implicit temporal promotion.

    Raises :class:`NotImplementedError` for MapType operands (Polars stores maps as
    ``List(Struct([key, value]))``, so sparkleframe can't replicate Spark's map
    equality semantics yet).
    """
    ld = _resolve_expr_output_dtype(left)
    rd = _resolve_expr_output_dtype(right)
    op = "eq" if equal else "ne"
    _assert_complex_compare_supported(op, ld, rd)
    _assert_spark_compare_compatible(op, left, right, ld, rd)
    if ld is not None and rd is not None and _needs_date_datetime_promotion(ld, rd):
        left, right = _promote_date_to_datetime_exprs(left, right, ld, rd)
        return (left == right) if equal else (left != right)
    if _is_complex_polars_dtype(ld) or _is_complex_polars_dtype(rd):
        return (left == right) if equal else (left != right)
    left_num, right_num, left_str, right_str = _numeric_compare_operands(left, right)
    numeric_valid = left_num.is_not_null() & right_num.is_not_null()
    if equal:
        coerced = pl.when(numeric_valid).then(left_num == right_num).otherwise(left_str == right_str)
    else:
        coerced = pl.when(numeric_valid).then(left_num != right_num).otherwise(left_str != right_str)
    if ld is not None and rd is not None:
        return _spark_compare_guard_expr(left, right, op, coerced, ld, rd)
    result = _native_compare_when_complex(left, right, op, coerced)
    return _spark_compare_guard_expr(left, right, op, result, ld, rd)


def _apply_getitem_key(expr: pl.Expr, key: Union[str, int]) -> pl.Expr:
    """
    One Spark getItem step on ``expr`` (struct field, list index, or map fallbacks).
    Assumes the active :func:`_polars_schema_for` is set when resolving dtypes.
    """
    if isinstance(key, str):
        dtype = _resolve_expr_output_dtype(expr)
        if isinstance(dtype, pl.Struct):
            field_names = {f.name for f in dtype.fields}
            if key not in field_names:
                return pl.lit(None).cast(pl.String)
            return expr.struct.field(key)
        if isinstance(dtype, pl.List) and isinstance(getattr(dtype, "inner", None), pl.Struct):
            inner: pl.Struct = dtype.inner
            field_names = {f.name for f in inner.fields}
            if key in field_names:
                return expr.list.eval(pl.element().struct.field(key))
            if "key" in field_names and "value" in field_names:
                return (
                    expr.list.eval(
                        pl.when(pl.element().struct.field("key") == pl.lit(key)).then(
                            pl.element().struct.field("value")
                        )
                    )
                    .list.drop_nulls()
                    .list.first()
                )

        def _extract_by_key(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, dict):
                return value.get(key)
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict) and entry.get("key") == key:
                        return entry.get("value")
                return None
            getter = getattr(value, "get", None)
            if callable(getter):
                try:
                    return getter(key)
                except Exception:
                    return None
            return None

        return expr.map_elements(_extract_by_key, return_dtype=pl.Object)
    if isinstance(key, int):
        dtype = _resolve_expr_output_dtype(expr)
        if isinstance(dtype, pl.List):
            return expr.list.get(key)

        def _index_at(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                if key < 0 or key >= len(v):
                    return None
                return v[key]
            return None

        return expr.map_elements(_index_at, return_dtype=pl.Object)
    raise TypeError(f"getItem key must be str or int, got {type(key).__name__}")


def _md5_sparklike(value: Any) -> str | None:
    """MD5 digest as 32-char hex (Spark: UTF-8 for strings, raw bytes for binary)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        b = bytes(value)
    else:
        b = str(value).encode("utf-8")
    return hashlib.md5(b, usedforsecurity=False).hexdigest()


def _re_split_sparklike(value: Any, pattern: str, limit: int) -> list[str] | None:
    """Replicate PySpark ``split`` limit semantics; uses Python :mod:`re` (not the JVM)."""
    if value is None:
        return None
    s = value if isinstance(value, str) else str(value)
    if limit == 0 or limit < 0:
        return re.split(pattern, s)
    if limit == 1:
        return [s]
    return re.split(pattern, s, maxsplit=limit - 1)


def _now_batch(s: pl.Series) -> pl.Series:
    """Batch function producing a constant ``now()`` timestamp for all rows."""
    if s.len() == 0:
        return pl.Series("now", [], dtype=pl.Datetime("us"))
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    return pl.Series("now", [ts] * s.len(), dtype=pl.Datetime("us"))
