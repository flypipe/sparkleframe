from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Optional, Union

import polars as pl

from sparkleframe.polarsdf.column import Column, _to_expr
from sparkleframe.polarsdf.types import (
    ArrayType,
    BooleanType,
    ByteType,
    DataType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructType,
    TimestampType,
    spark_name_to_datatype,
    spark_type_name_to_polars,
)
from sparkleframe.polarsdf.window import WindowSpec

_SPARK_DATE_FORMAT_MAP = [
    ("yyyy", "%Y"),
    ("MM", "%m"),
    ("dd", "%d"),
]


def _convert_spark_date_format(fmt: str) -> str:
    """Translate a Spark-style date format string to strftime-style."""
    for spark_fmt, strftime_fmt in _SPARK_DATE_FORMAT_MAP:
        fmt = fmt.replace(spark_fmt, strftime_fmt)
    return fmt


def _assert_no_unparsed_dates(s: pl.Series) -> pl.Series:
    """Raise when a non-null input produced a null date (Spark CAST_INVALID_INPUT)."""
    df = s.struct.unnest()
    inp = df.get_column("_input")
    res = df.get_column("_result")
    mask = inp.is_not_null() & res.is_null()
    if mask.any():
        bad = inp.filter(mask)[0]
        raise pl.exceptions.InvalidOperationError(
            f"[CAST_INVALID_INPUT] The value '{bad}' of the type \"STRING\" cannot be cast to "
            f'"DATE" because it is malformed. Use `try_to_date` to tolerate malformed '
            f"input and return NULL instead."
        )
    return res


def _to_date_column(col_name: Union[str, Column], fmt: str, *, strict: bool = True) -> Column:
    """
    Parse strings to :class:`Date` using a Spark format string (``yyyy-MM-dd`` style).

    ``to_date`` passes ``strict=True``; ``try_to_date`` passes ``strict=False``. When
    strict, non-null inputs that do not match the pattern raise (Spark 4 ANSI default).
    """
    strftime_fmt = _convert_spark_date_format(fmt)
    input_expr = pl.col(col_name) if isinstance(col_name, str) else _to_expr(col_name)
    string_expr = input_expr.cast(pl.String, strict=False)
    parsed = string_expr.str.strptime(pl.Date, strftime_fmt, strict=strict)
    if strict:
        parsed = pl.struct(
            string_expr.alias("_input"),
            parsed.alias("_result"),
        ).map_batches(_assert_no_unparsed_dates, return_dtype=pl.Date)
    return Column(parsed)


_SPARK_TS_FORMAT_MAP = [
    ("yyyy", "%Y"),
    ("MM", "%m"),
    ("dd", "%d"),
    ("HH", "%H"),
    ("mm", "%M"),
    ("ss", "%S"),
    (".SSSSSS", ".%6f"),
    (".SSSSS", ".%6f"),
    (".SSSS", ".%6f"),
    (".SSS", ".%6f"),
    (".SS", ".%6f"),
    (".S", ".%6f"),
]


def _convert_spark_ts_format(fmt: str) -> str:
    """Translate a Spark-style timestamp format string to strftime-style."""
    if fmt == "yyyy-MM-dd H:m:s":
        return "%Y-%m-%d %H:%M:%S"
    for spark_fmt, strftime_fmt in _SPARK_TS_FORMAT_MAP:
        fmt = fmt.replace(spark_fmt, strftime_fmt)
    return fmt


def _pad_microseconds_expr(expr: pl.Expr) -> pl.Expr:
    """Normalize fractional seconds to 6 digits (microseconds)."""

    def pad_microseconds(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        if "." in val:
            prefix, suffix = val.split(".", 1)
            suffix = (suffix + "000000")[:6]
            return f"{prefix}.{suffix}"
        return val

    return expr.map_elements(pad_microseconds, return_dtype=pl.String)


def _to_datetime_column(col_name: Union[str, Column], fmt: str, *, strict: bool = True) -> Column:
    """
    Parse strings to :class:`Datetime` using a Spark format string (``yyyy-MM-dd HH:mm:ss`` style).

    Spark 4 made ``to_timestamp`` strict by default (raises on malformed input) and
    introduced ``try_to_timestamp`` as the lenient variant (returns null). The ``strict``
    kwarg mirrors this: ``to_timestamp`` passes ``strict=True``, ``try_to_timestamp``
    passes ``strict=False``.

    Trailing ``Z`` / UTC-offset suffixes are stripped before parsing so that common
    ISO-8601 strings work with Spark's ``yyyy-MM-dd HH:mm:ss`` pattern. Input columns
    are cast to string first (``strict=False``) like implicit Spark casts.
    """
    strftime_fmt = _convert_spark_ts_format(fmt)
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    expr = expr.cast(pl.String, strict=False)
    if "%6f" in strftime_fmt:
        expr = _pad_microseconds_expr(expr)
    expr_without_tz = (
        expr.str.replace(r"Z$", "", literal=False)
        .str.replace(r"[+-]\d{2}:\d{2}$", "", literal=False)
        .str.replace(r"[+-]\d{4}$", "", literal=False)
    )
    parsed = expr_without_tz.str.strptime(pl.Datetime, strftime_fmt, strict=strict)
    return Column(parsed)


def _assert_no_unparsed_timestamps(s: pl.Series) -> pl.Series:
    """Raise when a non-null input produced a null timestamp (Spark CAST_INVALID_INPUT)."""
    df = s.struct.unnest()
    inp = df.get_column("_input")
    res = df.get_column("_result")
    mask = inp.is_not_null() & res.is_null()
    if mask.any():
        bad = inp.filter(mask)[0]
        raise pl.exceptions.InvalidOperationError(
            f"[CAST_INVALID_INPUT] The value '{bad}' of the type \"STRING\" cannot be cast to "
            f'"TIMESTAMP" because it is malformed. Use `try_to_timestamp` to tolerate malformed '
            f"input and return NULL instead."
        )
    return res


def _to_timestamp_no_format_column(col_name: Union[str, Column], *, strict: bool = True) -> Column:
    """
    One-argument :func:`to_timestamp` / :func:`try_to_timestamp` behaviour.

    Spark 4 ``to_timestamp(col)`` (no format) uses ANSI-strict ``stringToTimestampAnsi``
    and **raises** ``CAST_INVALID_INPUT`` on malformed strings, just like
    ``to_timestamp(col, fmt)``.  ``try_to_timestamp(col)`` is the lenient variant.

    Polars string->datetime ``cast`` handles many ISO-8601 forms but not some Spark-common
    layouts (e.g. ``yyyy-MM-dd HH:mm:ss`` with a space). We ``pl.coalesce`` a direct
    ``try_cast`` with :func:`_to_datetime_column` (lenient) using Spark's default pattern
    so both ISO and space-separated strings work.

    When ``strict=True`` (``to_timestamp``), a post-validation step raises for any
    non-null input that could not be parsed. When ``strict=False``
    (``try_to_timestamp``), unparseable values become null.
    """
    c = Column(pl.col(col_name)) if isinstance(col_name, str) else col_name
    casted = c.try_cast(TimestampType()).expr
    formatted = _to_datetime_column(col_name, "yyyy-MM-dd HH:mm:ss", strict=False).expr
    result = pl.coalesce(casted, formatted)
    if strict:
        input_expr = pl.col(col_name) if isinstance(col_name, str) else _to_expr(col_name)
        result = pl.struct(
            input_expr.cast(pl.String, strict=False).alias("_input"),
            result.alias("_result"),
        ).map_batches(_assert_no_unparsed_timestamps, return_dtype=pl.Datetime("us"))
    return Column(result)


def _schema_from_string(schema: str) -> Union[DataType, pl.DataType]:
    normalized = schema.strip()
    lowered = normalized.lower()

    if lowered.startswith("array<") and lowered.endswith(">"):
        inner = normalized[6:-1].strip()
        inner_schema = _schema_from_string(inner)
        if isinstance(inner_schema, DataType):
            return ArrayType(inner_schema)
        return pl.List(inner_schema)

    if lowered.startswith("map<") and lowered.endswith(">"):
        inner = normalized[4:-1].strip()
        key_schema, value_schema = [part.strip() for part in inner.split(",", 1)]
        key_type = _schema_from_string(key_schema)
        value_type = _schema_from_string(value_schema)
        if isinstance(key_type, DataType) and isinstance(value_type, DataType):
            return MapType(key_type, value_type)
        raise ValueError(f"Unsupported map schema '{schema}'")

    if "," in normalized and "<" not in normalized and ">" not in normalized:
        fields = []
        for piece in normalized.split(","):
            name_and_type = piece.strip().split()
            if len(name_and_type) < 2:
                raise ValueError(f"Invalid struct field declaration: '{piece}'")
            field_name = name_and_type[0]
            field_type = " ".join(name_and_type[1:])
            parsed = _schema_from_string(field_type)
            if not isinstance(parsed, DataType):
                raise ValueError(f"Unsupported nested struct field type '{field_type}'")
            from sparkleframe.polarsdf.types import StructField

            fields.append(StructField(field_name, parsed))
        return StructType(fields)

    try:
        return spark_name_to_datatype(normalized)
    except ValueError:
        pass

    return spark_type_name_to_polars(normalized)


def _coerce_json_value(value: Any, schema: Union[DataType, pl.DataType]) -> Any:
    if value is None:
        return None

    if isinstance(schema, StringType):
        return str(value)
    if isinstance(schema, (IntegerType, LongType, ShortType, ByteType)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(schema, (FloatType, DoubleType, DecimalType)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(schema, BooleanType):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        return None
    if isinstance(schema, ArrayType):
        if not isinstance(value, list):
            return None
        return [_coerce_json_value(item, schema.elementType) for item in value]
    if isinstance(schema, MapType):
        if not isinstance(value, dict):
            return None
        return [
            {
                "key": _coerce_json_value(k, schema.keyType),
                "value": _coerce_json_value(v, schema.valueType),
            }
            for k, v in value.items()
        ]
    if isinstance(schema, StructType):
        if not isinstance(value, dict):
            return None
        return {field.name: _coerce_json_value(value.get(field.name), field.dataType) for field in schema.fields}

    return value


def _series_as_date_sparklike(s: pl.Series) -> pl.Series:
    """
    Map column values to ``pl.Date`` with coercion closer to Spark ``cast(x as date)`` than plain
    :meth:`polars.Series.cast` alone.
    """
    if s.len() == 0:
        return pl.Series(s.name, [], dtype=pl.Date)
    if s.dtype == pl.Categorical:
        s = s.cast(pl.Utf8, strict=False)
    if s.dtype in (pl.Utf8, pl.String):
        d0 = s.cast(pl.Date, strict=False)
        vals: list[Any] = s.to_list()
        d0_list: list[Any] = d0.to_list()
        out: list[Any] = []
        for v, d in zip(vals, d0_list):
            if d is not None:
                out.append(d)
                continue
            if v is None:
                out.append(None)
                continue
            if not isinstance(v, str):
                out.append(None)
                continue
            try:
                t = pl.Series("_s", [v], dtype=pl.Utf8).str.to_datetime(time_zone="UTC", strict=False)
            except Exception:
                out.append(None)
                continue
            if t.len() == 0 or t.is_null().all():
                out.append(None)
            else:
                d1 = t.dt.replace_time_zone(None).cast(pl.Date, strict=False)
                out.append(d1.item())
        return pl.Series(s.name, out, dtype=pl.Date)
    return s.cast(pl.Date, strict=False)


def _as_date_sparklike_expr(e: pl.Expr) -> pl.Expr:
    return e.map_batches(_series_as_date_sparklike, return_dtype=pl.Date)


def _array_element_at_value(arr: Any, idx_1based: int, *, strict: bool) -> Any:
    """1-based Spark ``element_at`` / ``try_element_at`` array indexing."""
    if arr is None:
        return None
    if idx_1based == 0:
        if strict:
            raise IndexError(
                "Invalid index 0 for element_at. Spark uses 1-based array indexing; " "index 0 is out of bounds."
            )
        return None
    n = len(arr)
    polars_idx = idx_1based - 1 if idx_1based > 0 else idx_1based
    if polars_idx < -n or polars_idx >= n:
        if strict:
            raise IndexError(
                f"Index {idx_1based} out of bounds for array of length {n} "
                "(Spark 4 ANSI element_at raises SparkArrayIndexOutOfBoundsException)."
            )
        return None
    return arr[polars_idx]


def _lookup_map_value_by_key(map_value: Any, key: Any) -> Any:
    if map_value is None or key is None:
        return None
    if isinstance(map_value, dict):
        return map_value.get(key)
    if isinstance(map_value, list):
        for entry in map_value:
            if isinstance(entry, dict) and entry.get("key") == key:
                return entry.get("value")
    return None


def _map_key_lookup_expr(col_expr: pl.Expr, key_expr: pl.Expr) -> pl.Expr:
    """Lookup a key in Spark map layout ``List(Struct(key, value))``."""
    try:
        uses_named_column = len(key_expr.meta.root_names()) > 0
    except Exception:
        uses_named_column = True

    if uses_named_column:
        return pl.struct([col_expr.alias("_map"), key_expr.alias("_key")]).map_elements(
            lambda row: _lookup_map_value_by_key(row["_map"], row["_key"]),
            return_dtype=pl.Object,
        )

    return (
        col_expr.list.eval(
            pl.when(pl.element().struct.field("key") == key_expr).then(pl.element().struct.field("value"))
        )
        .list.drop_nulls()
        .list.first()
    )


def element_at_column(
    col_name: Union[str, Column],
    extraction: Union[str, int, Column],
    *,
    strict: bool,
) -> Column:
    """
    Shared implementation for :func:`~sparkleframe.polarsdf.functions.element_at` and
    ``try_element_at``.

    Spark 4 (ANSI): ``element_at`` raises on invalid array index; ``try_element_at`` returns
    null. For maps with a string ``extraction``, Spark documents that ``element_at`` treats
    the string as a **literal key**, while ``try_element_at`` treats it as a **column name**
    (see SPARK-48766); this helper mirrors that split.
    """
    col_expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)

    if isinstance(extraction, Column):
        key_expr = extraction.to_native()
        return Column(_map_key_lookup_expr(col_expr, key_expr))

    if isinstance(extraction, pl.Expr):
        return Column(_map_key_lookup_expr(col_expr, extraction))

    if isinstance(extraction, int):
        idx = extraction

        def _one(arr: Any) -> Any:
            return _array_element_at_value(arr, idx, strict=strict)

        return Column(col_expr.map_elements(_one, return_dtype=pl.Object))

    if isinstance(extraction, str):
        if strict:
            key_expr = pl.lit(extraction)
        else:
            key_expr = pl.col(extraction)
        return Column(_map_key_lookup_expr(col_expr, key_expr))

    raise TypeError(f"element_at extraction must be int, str, or Column, got {type(extraction).__name__}")


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


def _substring_sparklike(value: Any, pos: int, length: int) -> str | None:
    """Replicate Spark substring semantics (1-based indexing; negative ``pos`` from end)."""
    if value is None:
        return None
    if length <= 0:
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

    end = start + length
    if end > n:
        end = n
    return s[start:end]


class _RankWrapper(Column):
    """
    A wrapper for deferred window function binding, enabling rank().over(...).
    """

    def __init__(self, fn):
        self._fn = fn

    def over(self, window_spec: WindowSpec) -> Column:
        return self._fn(window_spec)
