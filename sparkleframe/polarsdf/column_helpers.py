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


def _series_to_string_compare(s: pl.Series) -> pl.Series:
    """Utf8 cast for comparisons; Object columns (e.g. ``getItem``) become ``str`` per cell."""
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
