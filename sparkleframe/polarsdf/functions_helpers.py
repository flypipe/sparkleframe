"""
Private helpers for :mod:`sparkleframe.polarsdf.functions` (Spark-like SQL functions).

Kept out of the public module to keep ``functions.py`` focused on the API surface.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from typing import Any, Optional, Union

import polars as pl

from sparkleframe.polarsdf.column import Column, _output_dtype_of_expr, _polars_schema_ctx, _to_expr
from sparkleframe.polarsdf.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DataType,
    DateType,
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
    spark_type_name_to_polars,
)


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

    # Simple struct shorthand e.g. "field_a STRING, field_b INT"
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
            from sparkleframe.polarsdf.types import StructField  # local import avoids cycle

            fields.append(StructField(field_name, parsed))
        return StructType(fields)

    # Primitive Spark aliases
    primitive_map = {
        "string": StringType(),
        "int": IntegerType(),
        "integer": IntegerType(),
        "bigint": LongType(),
        "long": LongType(),
        "short": ShortType(),
        "smallint": ShortType(),
        "tinyint": ByteType(),
        "byte": ByteType(),
        "float": FloatType(),
        "double": DoubleType(),
        "boolean": BooleanType(),
        "date": DateType(),
        "timestamp": TimestampType(),
        "binary": BinaryType(),
    }
    if lowered in primitive_map:
        return primitive_map[lowered]

    # Decimal(n,p) style
    decimal_match = re.match(r"decimal\((\d+)\s*,\s*(\d+)\)", lowered)
    if decimal_match:
        precision = int(decimal_match.group(1))
        scale = int(decimal_match.group(2))
        return DecimalType(precision, scale)

    # Last attempt: use spark name mapping directly to polars type
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

    # Polars dtype from string schema fallback
    return value


def _map_entries_list_to_json_obj(entries: Any) -> str | None:
    if entries is None:
        return None
    if not isinstance(entries, list):
        return None
    out: dict[str, Any] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        k = e.get("key")
        if k is None:
            continue
        out[str(k)] = e.get("value")
    return json.dumps(out, separators=(",", ":"))


def _is_spark_map_entry_list_dtype(list_dtype: pl.List) -> bool:
    inner = list_dtype.inner
    if not isinstance(inner, pl.Struct):
        return False
    names = {f.name for f in inner.fields}
    return names == {"key", "value"}


def _to_json_batch(s: pl.Series) -> pl.Series:
    name = s.name
    if s.len() == 0:
        return pl.Series(name, [], dtype=pl.String)
    dt = s.dtype
    if isinstance(dt, pl.Struct):
        return s.struct.json_encode()
    if isinstance(dt, pl.List):
        if _is_spark_map_entry_list_dtype(dt):
            rows = s.to_list()
            encoded = [_map_entries_list_to_json_obj(x) for x in rows]
            return pl.Series(name, encoded, dtype=pl.String)
        rows = s.to_list()

        def _dump(v: Any) -> str | None:
            if v is None:
                return None
            return json.dumps(v, separators=(",", ":"), default=str)

        return pl.Series(name, [_dump(x) for x in rows], dtype=pl.String)
    raise TypeError(
        "to_json expects a struct column, an array column, or a sparkleframe map column "
        f"(list<struct<key,value>>); got {dt}"
    )


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

# Spark :func:`date_format` output via ``strftime`` (no ``.%6f`` parse-style tokens).
_SPARK_DATETIME_STRFTIME_MAP = [
    ("yyyy", "%Y"),
    ("MM", "%m"),
    ("dd", "%d"),
    ("HH", "%H"),
    ("mm", "%M"),
    ("ss", "%S"),
]


def _convert_spark_datetime_pattern_to_strftime(fmt: str) -> str:
    """Translate a Spark datetime pattern to ``strftime`` for :func:`date_format` output."""
    out = fmt
    for spark_pat, strf in _SPARK_DATETIME_STRFTIME_MAP:
        out = out.replace(spark_pat, strf)
    return out


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


def _to_datetime_column(col_name: Union[str, Column], fmt: str) -> Column:
    """
    Parse strings to :class:`Datetime` using a Spark format string (``yyyy-MM-dd HH:mm:ss`` style).

    Spark ``to_timestamp`` / ``try_to_timestamp`` (SQL) yield null for values that do not
    match the *given* format; they do not fall back to unrelated ISO-8601 layouts. This
    implementation follows that: only the format-based parse is used, plus stripping common
    trailing ``Z`` / offset suffixes so the remainder matches ``strftime_fmt``.

    Input columns are cast to string first (``strict=False``) like implicit Spark casts
    to string before ``to_timestamp``; unparseable tokens become null, not exceptions.
    """
    strftime_fmt = _convert_spark_ts_format(fmt)
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    # Spark can stringify non-string inputs before parsing; use non-strict cast to string.
    expr = expr.cast(pl.String, strict=False)
    if "%6f" in strftime_fmt:
        expr = _pad_microseconds_expr(expr)
    expr_without_tz = (
        expr.str.replace(r"Z$", "", literal=False)
        .str.replace(r"[+-]\d{2}:\d{2}$", "", literal=False)
        .str.replace(r"[+-]\d{4}$", "", literal=False)
    )
    parsed = expr_without_tz.str.strptime(pl.Datetime, strftime_fmt, strict=False)
    return Column(parsed)


def _to_timestamp_no_format_column(col_name: Union[str, Column]) -> Column:
    """
    One-argument :func:`to_timestamp` / :func:`try_to_timestamp` behaviour.

    PySpark 4 documents omitted-format ``to_timestamp`` as following ``cast("timestamp")``.
    Polars string→datetime ``cast`` handles many ISO-8601 forms but not some Spark-common
    layouts (e.g. ``yyyy-MM-dd HH:mm:ss`` with a space). We ``pl.coalesce`` the cast
    result with :func:`_to_datetime_column` using Spark's usual default pattern
    ``yyyy-MM-dd HH:mm:ss`` so both ISO and space-separated strings align with PySpark
    in practice.
    """
    # Lazy import: ``col`` lives on the public functions module; importing it at helper
    # load time would create a circular import with ``functions``.
    from sparkleframe.polarsdf import functions as _sf_functions

    c = _sf_functions.col(col_name) if isinstance(col_name, str) else col_name
    casted = c.cast(TimestampType()).expr
    formatted = _to_datetime_column(col_name, "yyyy-MM-dd HH:mm:ss").expr
    return Column(pl.coalesce(casted, formatted))


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
        # Spark treats position 0 as starting from the first character.
        start = 0

    if start < 0:
        start = 0
    if start >= n:
        return ""

    end = start + length
    if end > n:
        end = n
    return s[start:end]


def _now_batch(s: pl.Series) -> pl.Series:
    if s.len() == 0:
        return pl.Series("now", [], dtype=pl.Datetime("us"))
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    return pl.Series("now", [ts] * s.len(), dtype=pl.Datetime("us"))


def _series_as_date_sparklike(s: pl.Series) -> pl.Series:
    """
    Map column values to ``pl.Date`` with coercion closer to Spark ``cast(x as date)`` than plain
    :meth:`polars.Series.cast` alone.

    Polars' string→date cast does not parse all ISO-8601 forms (e.g. ``...T...Z``) that Spark
    accepts. For string columns, fall back to parsing as UTC :class:`datetime` then to calendar
    date when the direct cast is null. Non-string columns use ``cast(DATE, strict=False)`` only.
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


def _rand_batch(s: pl.Series, seed: Optional[int]) -> pl.Series:
    n = s.len()
    if n == 0:
        return pl.Series(s.name, [], dtype=pl.Float64)
    rng = random.Random(seed) if seed is not None else random.Random()
    return pl.Series(s.name, [rng.random() for _ in range(n)], dtype=pl.Float64)


def _size_sparklike_value(v: Any) -> int | None:
    """Row-wise length for Spark-like ``size`` when Polars dtype is not ``List`` (e.g. ``Object``)."""
    if v is None:
        return None
    if isinstance(v, pl.Series):
        return v.len()
    if isinstance(v, list):
        return len(v)
    if isinstance(v, dict):
        return len(v)
    return None


def _as_col_expr(col_name: Union[str, Column]) -> pl.Expr:
    return _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)


def _struct_expand_varargs(cols: tuple[Any, ...]) -> tuple[Any, ...]:
    """Match PySpark ``struct`` when called as ``struct([c1, c2])`` or ``struct({...})``."""
    if len(cols) == 1 and isinstance(cols[0], (list, set)):
        return tuple(cols[0])
    return cols


def _struct_child_field_name(arg: Union[str, Column], expr: pl.Expr, index: int) -> str:
    """
    Spark ``CreateStruct`` naming: plain column refs keep their name (last segment if qualified);
    literals and non-trivial expressions become ``col1``, ``col2``, ...
    """
    if isinstance(arg, str):
        return arg.split(".")[-1]
    if b"RepeatBy" in expr.meta.serialize():
        return f"col{index + 1}"
    undone = expr.meta.undo_aliases()
    # Explicit Alias (nested struct(...).alias("nested_x"), col().alias("z"), …): Spark uses output_name.
    # Do not use serialize() inequality — Polars versions disagree for bare struct(); compare output names instead.
    if expr.meta.output_name() != undone.meta.output_name():
        return expr.meta.output_name().split(".")[-1]
    # Alias-of-column (e.g. col("a").alias("z")) is not is_column() in Polars; Spark uses the alias name.
    if undone.meta.is_column():
        return expr.meta.output_name().split(".")[-1]
    if expr.meta.is_literal():
        return f"col{index + 1}"
    return f"col{index + 1}"


def _struct_named_child(arg: Union[str, Column], index: int) -> pl.Expr:
    expr = _to_expr(arg) if isinstance(arg, Column) else pl.col(arg)
    name = _struct_child_field_name(arg, expr, index)
    # Polars rejects ``pl.struct`` with ``Object`` fields (``nested objects are not allowed``).
    # Map-entry structs use ``key`` / ``value`` names; coerce to string like Spark map keys/values.
    if name in ("key", "value"):
        expr = expr.cast(pl.String, strict=False)
    return expr.alias(name)


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
