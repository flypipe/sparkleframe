from __future__ import annotations

import hashlib
import json
import random
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
    """Resolve ``map_value[key]`` for whatever shape ``pl.Expr.map_elements`` hands us.

    The polars engine represents a Spark ``map<k,v>`` column in two ways depending on
    construction — see ``MapType.to_native`` (raw ``List(Struct(key, value))``) and the
    schema-driven auto-materialization in ``DataFrame.__init__`` (``Struct(<union_keys>)``).
    Plus, ``pl.struct(...).map_elements`` can hand the inner cell through as a ``dict``,
    a ``pl.Series``, a pyarrow ``StructScalar``, or another dict-like wrapper depending
    on polars version and dtype. Probe each shape rather than assume one layout.
    """
    if map_value is None or key is None:
        return None
    if isinstance(map_value, dict):
        return map_value.get(key)
    if isinstance(map_value, list):
        for entry in map_value:
            if isinstance(entry, dict) and entry.get("key") == key:
                return entry.get("value")
        return None
    if isinstance(map_value, pl.Series):
        for row in map_value.to_list():
            if isinstance(row, dict) and row.get("key") == key:
                return row.get("value")
        return None
    # Fall through: try common conversions for polars/pyarrow scalar wrappers.
    for conv in (
        lambda v: v.as_py() if hasattr(v, "as_py") else None,
        lambda v: v.to_dict() if hasattr(v, "to_dict") else None,
        lambda v: dict(v.items()) if hasattr(v, "items") else None,
        lambda v: dict(v) if hasattr(v, "keys") else None,
    ):
        try:
            converted = conv(map_value)
        except Exception:
            converted = None
        if isinstance(converted, dict):
            return converted.get(key)
        if isinstance(converted, list):
            for entry in converted:
                if isinstance(entry, dict) and entry.get("key") == key:
                    return entry.get("value")
            return None
    return None


def _map_key_lookup_expr(col_expr: pl.Expr, key_expr: pl.Expr) -> pl.Expr:
    """Lookup a key in a Spark-style map column, layout-agnostic.

    Polars stores a map in one of two layouts (see ``MapType.to_native`` and the
    auto-materialization in ``DataFrame.__init__``):

    - ``List(Struct(key, value))`` — the canonical Spark layout, kept verbatim when no
      ``MapType`` schema is supplied at construction.
    - ``Struct(<union_keys>)`` — produced when a ``MapType`` schema *is* supplied; the
      engine auto-materializes the map as a struct keyed by the ordered union of keys.

    ``list.eval`` works only on the first layout; ``struct.field`` only on the second.
    Dispatch by dtype isn't reliable here because expressions are built outside
    ``_polars_schema_for`` (``select`` only enters that context at evaluation), so we
    use a row-wise ``map_elements`` that works on the actual value regardless of
    layout. ``_lookup_map_value_by_key`` accepts dict, list-of-{k,v}, ``pl.Series``,
    and pyarrow/struct-scalar wrappers.
    """
    try:
        uses_named_column = len(key_expr.meta.root_names()) > 0
    except Exception:
        uses_named_column = True

    if uses_named_column:
        # Probe both layouts: ``row["_map"]`` for nested ``List(Struct(k,v))`` maps and
        # the flattened union (no ``_map`` key) for materialized ``Struct(<union_keys>)``
        # maps. ``return_dtype=pl.Object`` is required because polars probes with a
        # sentinel ``_key=""`` row for dtype inference; that probe lookup returns null,
        # polars would otherwise infer ``Null`` and discard the real value.
        def _resolve_row(row: Any) -> Any:
            if not isinstance(row, dict):
                return _lookup_map_value_by_key(row, None)
            key = row.get("_key")
            if key is None:
                return None
            if "_map" in row:
                return _lookup_map_value_by_key(row["_map"], key)
            return _lookup_map_value_by_key({k: v for k, v in row.items() if k != "_key"}, key)

        return pl.struct([col_expr.alias("_map"), key_expr.alias("_key")]).map_elements(
            _resolve_row,
            return_dtype=pl.Object,
        )

    # Literal key: extract once at expression-build time, then row-wise apply.
    try:
        lit_value = pl.DataFrame({"_x": [0]}).select(key_expr.alias("_v"))["_v"][0]
    except Exception:
        lit_value = None
    return col_expr.map_elements(
        lambda map_value: _lookup_map_value_by_key(map_value, lit_value),
        return_dtype=None,
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
        extraction = extraction.to_native()

    if isinstance(extraction, pl.Expr):
        try:
            lit_value = pl.DataFrame({"_x": [0]}).select(extraction.alias("_v"))["_v"][0]
            if isinstance(lit_value, int):
                extraction = lit_value
        except Exception:
            pass

    if isinstance(extraction, pl.Expr):
        return Column(_map_key_lookup_expr(col_expr, extraction))

    if isinstance(extraction, int):
        if extraction == 0:
            if strict:
                raise ValueError("element_at: index 0 is invalid (Spark uses 1-based indexing)")
            return Column(pl.lit(None))
        polars_idx = extraction - 1 if extraction > 0 else extraction
        if strict:

            def _strict_get(arr: Any) -> Any:
                return _array_element_at_value(arr, extraction, strict=True)

            return Column(col_expr.map_elements(_strict_get, return_dtype=pl.String))
        return Column(col_expr.list.get(polars_idx, null_on_oob=True))

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


def _transform_produces_struct(expr: pl.Expr) -> bool:
    """Return True when the expression contains ``pl.struct`` (Polars ``AsStruct``).

    Polars cannot evaluate ``pl.struct`` inside ``list.eval``, so ``transform``
    must fall back to a Python-side row-wise approach.  Serialization may fail
    when the expression tree contains Python UDFs, so we fall back to the
    string representation.
    """
    try:
        return b"AsStruct" in expr.meta.serialize()
    except Exception:
        try:
            return "as_struct(" in str(expr)
        except Exception:
            return False


def _transform_struct_batch(s: pl.Series, func: Any) -> pl.Series:
    """Apply *func* element-wise to each list cell, returning a list-of-dicts series.

    Used as the ``map_batches`` callback when :func:`transform`'s lambda produces a
    struct — Polars' ``list.eval`` cannot handle ``pl.struct`` natively.

    Each element is placed in a 1-row DataFrame and referenced via ``pl.col("_x")``
    so that nested types (lists, structs) are preserved correctly — ``pl.lit()``
    flattens lists into separate rows.

    Polars also rejects ``as_struct`` wrapping Python UDFs (``nested objects are not
    allowed``), so struct fields are evaluated individually and combined into a dict.

    The returned series has a concrete ``List(Struct(...))`` dtype inferred from the
    first non-null result so that ``map_batches`` produces a properly-typed column.
    """
    from sparkleframe.polarsdf.column import Column as _Col

    results: list[list[dict] | None] = []
    for cell in s:
        if cell is None:
            results.append(None)
            continue
        row_results: list[dict] = []
        for elem in cell:
            elem_col = _Col(pl.col("_x"))
            out = func(elem_col)
            tiny_df = pl.DataFrame({"_x": [elem]})
            if isinstance(out, _Col) and out._struct_parts is not None:
                row_dict: dict[str, Any] = {}
                for part_expr in out._struct_parts:
                    name = part_expr.meta.output_name()
                    val = tiny_df.select(part_expr.alias("_r"))["_r"][0]
                    row_dict[name] = val
                row_results.append(row_dict)
            else:
                out_expr = out.to_native() if isinstance(out, _Col) else _to_expr(out)
                val = tiny_df.select(out_expr.alias("_r"))["_r"][0]
                row_results.append(val)
        results.append(row_results)

    struct_dtype = _infer_list_struct_dtype(results)
    if struct_dtype is not None:
        return pl.Series(results, dtype=struct_dtype)
    return pl.Series(results)


def _infer_list_struct_dtype(results: list) -> Optional[pl.DataType]:
    """Infer ``List(Struct(...))`` dtype from the first non-null dict in *results*."""
    for cell in results:
        if cell is None:
            continue
        for row in cell:
            if isinstance(row, dict) and row:
                fields: list[pl.Field] = []
                for k, v in row.items():
                    if isinstance(v, bool):
                        fields.append(pl.Field(k, pl.Boolean))
                    elif isinstance(v, int):
                        fields.append(pl.Field(k, pl.Int64))
                    elif isinstance(v, float):
                        fields.append(pl.Field(k, pl.Float64))
                    else:
                        fields.append(pl.Field(k, pl.Utf8))
                return pl.List(pl.Struct(fields))
    return None


def _infer_struct_return_dtype_from_expr(expr: pl.Expr) -> pl.DataType:
    """Extract struct field names from an expression's string representation.

    When ``_struct_parts`` is unavailable (e.g. struct wrapped in ``when/otherwise``),
    this parses ``.alias("name")`` patterns from the expression tree to build a
    ``List(Struct(...))`` dtype.  Falls back to a single-field struct rather than
    ``pl.Object``, which causes Polars panics in ``map_batches``.
    """
    try:
        expr_str = str(expr)
        aliases = re.findall(r'\.alias\(["\']([^"\']+)["\']\)', expr_str)
        if aliases:
            seen: list[str] = []
            for a in aliases:
                if a not in seen:
                    seen.append(a)
            fields = [pl.Field(name, pl.Utf8) for name in seen]
            return pl.List(pl.Struct(fields))
    except Exception:
        pass
    return pl.List(pl.Struct([pl.Field("value", pl.Utf8)]))


def _struct_return_dtype_from_parts(struct_parts: list[pl.Expr]) -> pl.DataType:
    """Build a ``List(Struct(...))`` dtype from struct part expressions.

    Field types default to ``Utf8`` since they cannot be determined without data.
    The actual batch function infers concrete types from the data; Polars casts
    the result if the declared ``return_dtype`` doesn't match exactly.
    """
    fields: list[pl.Field] = []
    for part_expr in struct_parts:
        try:
            name = part_expr.meta.output_name()
        except Exception:
            name = f"col{len(fields) + 1}"
        fields.append(pl.Field(name, pl.Utf8))
    return pl.List(pl.Struct(fields))
