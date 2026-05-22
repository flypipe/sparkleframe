from __future__ import annotations

import contextvars
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


def _is_complex_polars_dtype(dt: Optional[pl.DataType]) -> bool:
    if dt is None:
        return False
    return isinstance(dt, (pl.List, pl.Struct, pl.Array))


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
    left_num, right_num, left_str, right_str = _numeric_compare_operands(left, right)
    numeric_ok = left_num.is_not_null() & right_num.is_not_null()
    if _is_numeric_polars_dtype(ld) or _is_numeric_polars_dtype(rd):
        return _cmp_exprs(left_num, right_num, op)

    left_dt = _coerce_expr_order_datetime(left)
    right_dt = _coerce_expr_order_datetime(right)
    temporal_ok = left_dt.is_not_null() & right_dt.is_not_null()
    if ld is not None or rd is not None:
        return (
            pl.when(temporal_ok)
            .then(_cmp_exprs(left_dt, right_dt, op))
            .when(numeric_ok)
            .then(_cmp_exprs(left_num, right_num, op))
            .otherwise(_cmp_exprs(left_str, right_str, op))
        )
    coerced_expr = (
        pl.when(numeric_ok)
        .then(_cmp_exprs(left_num, right_num, op))
        .when(temporal_ok)
        .then(_cmp_exprs(left_dt, right_dt, op))
        .otherwise(_cmp_exprs(left_str, right_str, op))
    )
    return _native_compare_when_complex(left, right, op, coerced_expr)


def _equality_comparison_expr(left: pl.Expr, right: pl.Expr, equal: bool) -> pl.Expr:
    """
    Spark-like ``==`` / ``!=`` over Polars expressions. Cross-type equality coerces
    through string / numeric (so ``col(int) == lit('1')`` matches Spark), while
    list / struct / array operands fall back to Polars' native equality.

    Raises :class:`NotImplementedError` for MapType operands (Polars stores maps as
    ``List(Struct([key, value]))``, so sparkleframe can't replicate Spark's map
    equality semantics yet).
    """
    ld = _resolve_expr_output_dtype(left)
    rd = _resolve_expr_output_dtype(right)
    op = "eq" if equal else "ne"
    _assert_complex_compare_supported(op, ld, rd)
    if _is_complex_polars_dtype(ld) or _is_complex_polars_dtype(rd):
        return (left == right) if equal else (left != right)
    left_num, right_num, left_str, right_str = _numeric_compare_operands(left, right)
    numeric_valid = left_num.is_not_null() & right_num.is_not_null()
    if equal:
        coerced = pl.when(numeric_valid).then(left_num == right_num).otherwise(left_str == right_str)
    else:
        coerced = pl.when(numeric_valid).then(left_num != right_num).otherwise(left_str != right_str)
    if ld is not None and rd is not None:
        return coerced
    return _native_compare_when_complex(left, right, op, coerced)


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
