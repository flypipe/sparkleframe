from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional, Union
from uuid import uuid4

import polars as pl

from sparkleframe.polarsdf import WindowSpec
from sparkleframe.polarsdf.column import (
    Column,
    _expr_as_string_for_compare,
    _output_dtype_of_expr,
    _polars_schema_ctx,
    _to_expr,
)
from sparkleframe.polarsdf.functions_utils import _RankWrapper
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


def col(name: str) -> Column:
    """
    Mimics pyspark.sql.functions.col by returning a Column object.
    Supports dotted paths for nested struct access, e.g. "col.a.b".

    Args:
        name (str): Name of the column.

    Returns:
        Column: A Column object for building expressions.
    """
    if "." in name:
        parts = name.split(".")
        # Defer struct navigation through :class:`Column` so :func:`_apply_getitem_key`
        # runs under the active frame schema (null-safe / missing-field parity with Spark).
        c: Column = Column(parts[0])
        for seg in parts[1:]:
            c = c.getItem(seg)
        return c
    return Column(pl.col(name))


def get_json_object(col: Union[str, Column], path: str) -> Column:
    """
    Mimics pyspark.sql.functions.get_json_object by extracting a JSON field.

    Args:
        col (str | Column): The column containing the JSON string.
        path (str): The JSON path in the format '$.field.subfield'.

    Returns:
        Column: A column representing the extracted JSON value.
    """
    if not isinstance(path, str) or not path.startswith("$."):
        raise ValueError("Path must be a string starting with '$.'")

    col_expr = col.to_native() if isinstance(col, Column) else pl.col(col)

    return Column(col_expr.str.json_path_match(path))


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


def from_json(col_name: Union[str, Column], schema: Union[DataType, str]) -> Column:
    """
    Mimics pyspark.sql.functions.from_json for common schemas.

    Supports sparkleframe DataType schemas (ArrayType/MapType/StructType and primitives)
    and simple Spark SQL schema strings (e.g. "array<string>", "map<string,string>",
    "field_a STRING, field_b INT").
    """
    parsed_schema = _schema_from_string(schema) if isinstance(schema, str) else schema
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)

    def _parse(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            raw = value
        else:
            try:
                raw = json.loads(value)
            except Exception:
                return None
        return _coerce_json_value(raw, parsed_schema)

    if isinstance(parsed_schema, DataType):
        return_dtype = parsed_schema.to_native()
    else:
        return_dtype = parsed_schema
    return Column(expr.map_elements(_parse, return_dtype=return_dtype))


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


def to_json(col_name: Union[str, Column], options: Any = None) -> Column:
    """
    Mimics pyspark.sql.functions.to_json for struct, array, and map-like columns.

    Map columns produced by :func:`create_map` / :func:`map_from_entries` (encoded as
    ``list<struct<key,value>>``) are serialized as JSON objects, like Spark ``MapType``.

    Supports only ``options={"ignoreNullFields": False}`` from Spark's options map.
    """
    if options is not None and options != {"ignoreNullFields": False}:
        raise ValueError('sparkleframe to_json only supports options={"ignoreNullFields": False}')
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.map_batches(_to_json_batch, return_dtype=pl.String))


def lit(value) -> Column:
    """
    Mimics pyspark.sql.functions.lit.

    Creates a Column of literal value.

    Args:
        value: A literal value (int, float, str, bool, None, etc.)

    Returns:
        Column: A Column object wrapping a literal Polars expression.
    """
    # Plain ``pl.lit`` broadcasts with ``with_columns`` / ``withColumn``. ``select`` is handled
    # in :meth:`DataFrame.select` when Polars collapses literal-only selects to one row.
    if value is None:
        c = Column(pl.lit(value).cast(pl.String))
    else:
        c = Column(pl.lit(value))
    c._broadcast_row_count_in_select = True
    return c


def coalesce(*cols: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.coalesce.

    Returns the first non-null value among the given columns.

    Args:
        *cols: A variable number of columns (str or Column)

    Returns:
        Column: A Column representing the coalesced expression.
    """
    if not cols:
        raise ValueError("coalesce requires at least one column")

    expressions = [_to_expr(col) if isinstance(col, Column) else pl.col(col) for col in cols]

    return Column(pl.coalesce(*expressions))


def least(*cols: Any) -> Column:
    """
    Mimics pyspark.sql.functions.least.

    Returns the smallest value per row across the given columns or literals. Null
    inputs are skipped; the result is null when every argument is null for that row.

    Note:
        Spark also skips IEEE-754 ``NaN`` in floating-point ``least``; Polars
        ``min_horizontal`` may return ``NaN`` when any argument is ``NaN``. Prefer
        nulls for missing floats, or normalize ``NaN`` before calling :func:`least`,
        if you need Spark-identical behaviour on ``NaN``.

    Args:
        *cols: Column names (``str``), :class:`~sparkleframe.polarsdf.column.Column`
            expressions, literals, or ``pl.Expr``.

    Returns:
        Column: Row-wise minimum in the promoted common type.

    Raises:
        ValueError: If fewer than two arguments are provided.
    """
    if len(cols) < 2:
        raise ValueError("least requires at least two arguments")
    expressions: list[pl.Expr] = [pl.col(c) if isinstance(c, str) else _to_expr(c) for c in cols]
    return Column(pl.min_horizontal(*expressions))


def greatest(*cols: Any) -> Column:
    """
    Mimics pyspark.sql.functions.greatest.

    Returns the largest value per row across the given columns or literals. Null
    inputs are skipped; the result is null when every argument is null for that row.

    Note:
        Spark skips IEEE-754 ``NaN`` in floating-point ``greatest``; Polars
        ``max_horizontal`` may return ``NaN`` when any argument is ``NaN``. Prefer
        nulls for missing floats, or normalize ``NaN`` before calling :func:`greatest`,
        if you need Spark-identical behaviour on ``NaN``.

    Args:
        *cols: Column names (``str``), :class:`~sparkleframe.polarsdf.column.Column`
            expressions, literals, or ``pl.Expr``.

    Returns:
        Column: Row-wise maximum in the promoted common type.

    Raises:
        ValueError: If fewer than two arguments are provided.
    """
    if len(cols) < 2:
        raise ValueError("greatest requires at least two arguments")
    expressions: list[pl.Expr] = [pl.col(c) if isinstance(c, str) else _to_expr(c) for c in cols]
    return Column(pl.max_horizontal(*expressions))


def count(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.count.

    Counts the number of non-null elements for the specified column.

    Args:
        col_name (str or Column): The column to count non-null values in.

    Returns:
        Column: A Column representing the count aggregation expression.
    """
    if isinstance(col_name, str) and col_name == "*":
        return Column(pl.len())
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.count())


def sum(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.sum.

    Computes the sum of non-null values in the specified column.

    Args:
        col_name (str or Column): The column to sum.

    Returns:
        Column: A Column representing the sum aggregation expression.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.sum())


def mean(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.mean (alias for avg).

    Computes the mean of non-null values in the specified column.

    Args:
        col_name (str or Column): The column to average.

    Returns:
        Column: A Column representing the mean aggregation expression.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.mean())


def min(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.min.

    Computes the minimum of non-null values in the specified column.

    Args:
        col_name (str or Column): The column to find the minimum value of.

    Returns:
        Column: A Column representing the min aggregation expression.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.min())


def max(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.max.

    Computes the maximum of non-null values in the specified column.

    Args:
        col_name (str or Column): The column to find the maximum value of.

    Returns:
        Column: A Column representing the max aggregation expression.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.max())


def first(col_name: Union[str, Column], ignorenulls: bool = False) -> Column:
    """
    Mimics pyspark.sql.functions.first.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.drop_nulls().first() if ignorenulls else expr.first())


def map_from_entries(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.map_from_entries.

    Expects an array of structs with ``key`` and ``value`` fields.
    """
    # Spark map is represented in sparkleframe as list<struct<key,value>>,
    # which keeps key/value access compatible with element_at/getItem helpers.
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    entry = pl.Struct(
        [
            pl.Field("key", pl.String),
            pl.Field("value", pl.String),
        ]
    )
    expr = expr.cast(pl.List(entry), strict=False)
    return Column(expr)


def create_map(*cols: Any) -> Column:
    """
    Mimics pyspark.sql.functions.create_map.

    Builds a map column from alternating key and value arguments. Keys and values may
    be column names (``str``), :class:`~sparkleframe.polarsdf.column.Column` expressions,
    or literals. The result uses sparkleframe's map encoding (``list<struct<key,value>>``),
    consistent with :func:`map_from_entries` and map helpers.

    Args:
        *cols: Even-length sequence ``k1, v1, k2, v2, ...``.

    Returns:
        Column: One map per row.

    Raises:
        ValueError: If the number of arguments is odd.
    """
    if len(cols) % 2 != 0:
        raise ValueError("create_map requires an even number of arguments (key, value pairs)")
    if not cols:
        empty_dtype = pl.List(
            pl.Struct([pl.Field("key", pl.String), pl.Field("value", pl.String)]),
        )
        return Column(pl.lit([], dtype=empty_dtype))
    parts: list[pl.Expr] = []
    for i in range(0, len(cols), 2):
        key_expr = pl.col(cols[i]) if isinstance(cols[i], str) else _to_expr(cols[i])
        val_expr = pl.col(cols[i + 1]) if isinstance(cols[i + 1], str) else _to_expr(cols[i + 1])
        parts.append(pl.struct([key_expr.alias("key"), val_expr.alias("value")]))
    return Column(pl.concat_list(parts))


def array(*cols: Any) -> Column:
    """
    Mimics pyspark.sql.functions.array.

    Builds a list column from column names (``str``), :class:`~sparkleframe.polarsdf.column.Column`
    values, or literals. Spark requires a common element type; Polars may infer a supertype or
    object list when mixing incompatible types.

    Args:
        *cols: Zero or more arguments (empty ``array()`` yields an empty list column).

    Returns:
        Column: One list value per row.
    """
    if not cols:
        return Column(pl.lit([], dtype=pl.List(pl.Null)))
    parts: list[pl.Expr] = [pl.col(c) if isinstance(c, str) else _to_expr(c) for c in cols]
    return Column(pl.concat_list(parts))


def map_keys(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.map_keys.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)

    def _keys(value: Any):
        if value is None:
            return None
        if isinstance(value, dict):
            return list(value.keys())
        if isinstance(value, list):
            return [entry.get("key") for entry in value if isinstance(entry, dict) and "key" in entry]
        return None

    return Column(expr.map_elements(_keys, return_dtype=pl.List(pl.String)))


def collect_list(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.collect_list.

    Collects values into a list per group when used with ``groupBy`` / ``agg``.
    Null values are omitted from the list, matching PySpark.

    Args:
        col_name (str or Column): The column whose values are collected.

    Returns:
        Column: A Column representing the list aggregation expression.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.filter(expr.is_not_null()).implode())


def collect_set(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.collect_set.

    Collects distinct non-null values per group when used with ``groupBy`` / ``agg``.
    The order of elements in the result array is not guaranteed, matching PySpark.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.filter(expr.is_not_null()).implode().list.unique())


def transform(col_name: Union[str, Column], func: Callable[[Column], Any]) -> Column:
    """
    Mimics pyspark.sql.functions.transform for array columns.

    Applies a lambda expression to each element of an array and returns a new array.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    element_col = Column(pl.element())
    transformed = func(element_col)
    transformed_expr = transformed.to_native() if isinstance(transformed, Column) else _to_expr(transformed)
    return Column(expr.list.eval(transformed_expr))


def round(col_name: Union[str, Column], scale: int = 0) -> Column:
    """
    Mimics pyspark.sql.functions.round.

    Rounds the values of a column to the specified number of decimal places.

    Args:
        col_name (str or Column): The column to round.
        scale (int): Number of decimal places to round to. Default is 0 (nearest integer).

    Returns:
        Column: A Column representing the rounded values.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.round(scale))


class WhenBuilder:
    def __init__(self, condition: Column, value):
        self.branches = [(condition.to_native(), _to_expr(value))]

    def when(self, condition: Any, value) -> "WhenBuilder":
        condition = Column(condition) if not isinstance(condition, Column) else condition
        self.branches.append((condition.to_native(), _to_expr(value)))
        return self

    def otherwise(self, value) -> Column:
        expr = pl.when(self.branches[0][0]).then(self.branches[0][1])
        for cond, val in self.branches[1:]:
            expr = expr.when(cond).then(val)
        return Column(expr.otherwise(_to_expr(value)))


def when(condition: Any, value) -> WhenBuilder:
    """
    Starts a multi-branch conditional expression.

    Returns a WhenBuilder which can be chained with .when(...).otherwise(...).
    """
    condition = Column(condition) if not isinstance(condition, Column) else condition
    return WhenBuilder(condition, value)


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
    c = col(col_name) if isinstance(col_name, str) else col_name
    casted = c.cast(TimestampType()).expr
    formatted = _to_datetime_column(col_name, "yyyy-MM-dd HH:mm:ss").expr
    return Column(pl.coalesce(casted, formatted))


def to_timestamp(
    col_name: Union[str, Column],
    fmt: Optional[str] = None,
) -> Column:
    """
    Mimics pyspark.sql.functions.to_timestamp.

    If ``fmt`` is omitted, uses :func:`_to_timestamp_no_format_column` to mirror PySpark
    "cast" semantics while covering layouts Polars cannot parse via cast alone. If
    ``fmt`` is provided, only that Spark datetime pattern is used (as in SQL
    ``to_timestamp(s, fmt)``), via :func:`_to_datetime_column`.

    Args:
        col_name (str or Column): Column with string values to convert to timestamps.
        fmt (str, optional): Spark datetime pattern, or ``None`` to use one-arg rules.

    Returns:
        Column: A Column with values converted to Polars datetime type.
    """
    if fmt is None:
        return _to_timestamp_no_format_column(col_name)
    return _to_datetime_column(col_name, fmt)


def date_format(col_name: Union[str, Column], fmt: str) -> Column:
    """
    Mimics pyspark.sql.functions.date_format.

    Formats date or timestamp values as strings using Spark datetime pattern letters
    (``yyyy``, ``MM``, ``dd``, ``HH``, ``mm``, ``ss``). Inputs are cast to microsecond
    timestamps like Spark; :class:`date` values use midnight for time-of-day fields.

    Args:
        col_name (str or Column): Temporal column (or values castable to timestamp).
        fmt (str): Spark datetime pattern.

    Returns:
        Column: Utf8 formatted strings; null when the input is null.
    """
    strftime_fmt = _convert_spark_datetime_pattern_to_strftime(fmt)
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    dt_expr = expr.cast(pl.Datetime("us"), strict=False)
    return Column(dt_expr.dt.strftime(strftime_fmt))


def regexp_replace(col_name: Union[str, Column], pattern: str, replacement: str) -> Column:
    """
    Mimics pyspark.sql.functions.regexp_replace.

    Replaces all substrings of the specified string column that match the regular expression
    with the given replacement.

    Args:
        col_name (str or Column): Column containing strings to operate on.
        pattern (str): Regular expression pattern to match.
        replacement (str): Replacement string.

    Returns:
        Column: A Column with the regex-replaced string results.
    """
    col_name = pl.col(col_name) if isinstance(col_name, str) else col_name
    expr = _to_expr(col_name)
    return Column(expr.str.replace_all(pattern, replacement))


def length(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.length.

    Computes the length (number of characters) of the string in the column.

    Args:
        col_name (str or Column): The string column.

    Returns:
        Column: A Column representing the length of each string.
    """
    col_name = pl.col(col_name) if isinstance(col_name, str) else col_name
    expr = _to_expr(col_name)
    return Column(expr.str.len_chars().cast(pl.Int32))


def asc(column: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.asc.

    Specifies ascending sort order for the column.

    Args:
        column (str or Column): The column to sort in ascending order.

    Returns:
        Column: A Column object representing ascending order sort expression.
    """
    descending = False
    nulls_last = False
    col_expr = _to_expr(col(column)) if isinstance(column, str) else column.to_native()
    column_ = Column(col_expr.sort(descending=descending, nulls_last=nulls_last))
    column_._sort_col = col_expr
    column_._sort_descending = descending
    column_._sort_nulls_last = nulls_last
    return column_


def asc_nulls_first(column: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.asc_nulls_first.

    Specifies ascending sort order with nulls first for the column.

    Args:
        column (str or Column): The column to sort in ascending order.

    Returns:
        Column: A Column object representing ascending order sort expression with nulls first.
    """

    return asc(column)


def asc_nulls_last(column: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.asc_nulls_last.

    Specifies ascending sort order with nulls last for the column.

    Args:
        column (str or Column): The column to sort in ascending order.

    Returns:
        Column: A Column object representing ascending order sort expression with nulls last.
    """

    descending = False
    nulls_last = True
    col_expr = _to_expr(col(column)) if isinstance(column, str) else column.to_native()
    column_ = Column(col_expr.sort(descending=descending, nulls_last=nulls_last))
    column_._sort_col = col_expr
    column_._sort_descending = descending
    column_._sort_nulls_last = nulls_last
    return column_


def desc(column: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.desc.

    Specifies descending sort order for the column.

    Args:
        column (str or Column): The column to sort in descending order.

    Returns:
        Column: A Column object representing descending order sort expression.
    """
    descending = True
    nulls_last = True
    col_expr = _to_expr(col(column)) if isinstance(column, str) else column.to_native()
    column_ = Column(col_expr.sort(descending=descending, nulls_last=nulls_last))
    column_._sort_col = col_expr
    column_._sort_descending = descending
    column_._sort_nulls_last = nulls_last
    return column_


def desc_nulls_first(column: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.desc_nulls_first.

    Specifies descending sort order with nulls first for the column.

    Args:
        column (str or Column): The column to sort in descending order.

    Returns:
        Column: A Column object representing descending order sort expression with nulls first.
    """

    descending = True
    nulls_last = False
    col_expr = _to_expr(col(column)) if isinstance(column, str) else column.to_native()
    column_ = Column(col_expr.sort(descending=descending, nulls_last=nulls_last))
    column_._sort_col = col_expr
    column_._sort_descending = descending
    column_._sort_nulls_last = nulls_last
    return column_


def desc_nulls_last(column: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.desc_nulls_last.

    Specifies descending sort order with nulls last for the column.

    Args:
        column (str or Column): The column to sort in descending order.

    Returns:
        Column: A Column object representing descending order sort expression with nulls last.
    """

    return desc(column)


def rank() -> Column:
    """
    Mimics pyspark.sql.functions.rank using Polars rank("dense").rank method.
    Returns a Column that can be used with .withColumn().
    """

    def _rank_fn(window_spec: WindowSpec):
        rank_expr = (
            pl.struct([(col._sort_col).rank(descending=col._sort_descending) for col in window_spec.order_cols])
            .rank(method="min")
            .over(partition_by=window_spec.partition_cols)
        )

        return Column(rank_expr)

    return _RankWrapper(_rank_fn)


def dense_rank() -> Column:
    """
    Mimics pyspark.sql.functions.dense_rank.
    Returns a Column that can be used with .withColumn().
    """

    def _dense_rank_fn(window_spec: WindowSpec):
        rank_expr = (
            pl.struct([(col._sort_col).rank(descending=col._sort_descending) for col in window_spec.order_cols])
            .rank(method="dense")
            .over(partition_by=window_spec.partition_cols)
        )

        return Column(rank_expr)

    return _RankWrapper(_dense_rank_fn)


def row_number() -> Column:
    """
    Mimics pyspark.sql.functions.row_number.
    Returns a Column that can be used with .withColumn().
    """

    def _row_number_fn(window_spec: WindowSpec):
        rank_expr = (
            pl.struct([(col._sort_col).rank(descending=col._sort_descending) for col in window_spec.order_cols])
            .rank(method="ordinal")
            .over(partition_by=window_spec.partition_cols)
        )

        return Column(rank_expr)

    return _RankWrapper(_row_number_fn)


def abs(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.abs.

    Computes the absolute value of a numeric column.

    Args:
        col_name (str or Column): The column for which to compute absolute values.

    Returns:
        Column: A Column representing the absolute value expression.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.abs())


def floor(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.floor.

    Rounds numeric values down to the nearest integer. Non-numeric strings follow
    Spark-like casting via ``cast(..., strict=False)`` before ``floor``.

    Args:
        col_name (str or Column): Numeric or string-castable column.

    Returns:
        Column: Floored values (``Float64`` after rounding when the input was fractional).
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.cast(pl.Float64, strict=False).floor())


def pow(base: Any, exponent: Any) -> Column:
    """
    Mimics pyspark.sql.functions.pow (``base`` ** ``exponent``).

    Both sides may be column names (``str``), :class:`~sparkleframe.polarsdf.column.Column`,
    ``pl.Expr``, or scalar literals (e.g. ``F.pow(1 + col('rate'), -36)``).

    Args:
        base: Base expression or column name.
        exponent: Exponent expression, column name, or scalar.

    Returns:
        Column: ``base ** exponent``; null if either operand is null.
    """
    base_expr = pl.col(base) if isinstance(base, str) else _to_expr(base)
    exp_expr = pl.col(exponent) if isinstance(exponent, str) else _to_expr(exponent)
    return Column(base_expr.pow(exp_expr))


def isnan(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.isnan.

    Returns false for null inputs (Spark behaviour). Non-float values are evaluated
    after casting to ``Float64`` with ``strict=False``; values that cannot be cast
    yield false here, whereas Spark with ANSI enabled may fail the stage on invalid casts.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    as_float = expr.cast(pl.Float64, strict=False)
    return Column(
        pl.when(expr.is_null()).then(pl.lit(False)).otherwise(as_float.is_nan().fill_null(False)).cast(pl.Boolean)
    )


def try_divide(left: Union[str, Column], right: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.try_divide (Spark 3.4+).

    Returns null when the divisor is zero or null, or when the dividend is null.
    Otherwise returns ``left / right`` as ``Float64``.
    """
    l_expr = _to_expr(left) if isinstance(left, Column) else pl.col(left)
    r_expr = _to_expr(right) if isinstance(right, Column) else pl.col(right)
    l64 = l_expr.cast(pl.Float64, strict=False)
    r64 = r_expr.cast(pl.Float64, strict=False)
    return Column(pl.when(l_expr.is_null() | r_expr.is_null() | (r64 == 0)).then(pl.lit(None)).otherwise(l64 / r64))


def lower(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.lower.

    Converts all characters of a string column to lower case.

    Args:
        col_name (str or Column): The string column to transform.

    Returns:
        Column: A Column with lower-cased string values.
    """
    col_name = pl.col(col_name) if isinstance(col_name, str) else col_name
    expr = _to_expr(col_name).cast(pl.String, strict=False)
    return Column(expr.str.to_lowercase())


def initcap(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.initcap.

    Converts the first letter of each word to uppercase and the rest to lowercase.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.str.to_titlecase())


def _md5_sparklike(value: Any) -> str | None:
    """MD5 digest as 32-char hex (Spark: UTF-8 for strings, raw bytes for binary)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        b = bytes(value)
    else:
        b = str(value).encode("utf-8")
    return hashlib.md5(b, usedforsecurity=False).hexdigest()


def md5(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.md5.

    Returns the MD5 hash of a string (UTF-8) or binary column as a 32-character hex string.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.map_elements(_md5_sparklike, return_dtype=pl.String))


def trim(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.trim (single-argument form).

    Removes leading and trailing **ASCII space** (U+0020) only, matching Spark.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.str.strip_chars(" "))


def nullif(e1: Union[str, Column], e2: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.nullif.

    Returns null when both operands are non-null and equal; otherwise returns ``e1``.
    Matches Spark ``CASE WHEN e1 = e2 THEN NULL ELSE e1 END`` (equality unknown when
    either side is null yields ``e1``).
    """
    a = _to_expr(e1) if isinstance(e1, Column) else pl.col(e1)
    b = _to_expr(e2) if isinstance(e2, Column) else pl.col(e2)
    return Column(pl.when(a.is_not_null() & b.is_not_null() & (a == b)).then(None).otherwise(a))


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


def split(col_name: Union[str, Column], pattern: str, limit: int = -1) -> Column:
    """
    Mimics pyspark.sql.functions.split.

    Splits a string on a *regex* ``pattern`` (Python :mod:`re` dialect; subtle differences
    from Spark's Java engine are possible). A non-positive ``limit`` applies the pattern
    as many times as possible; a positive limit caps splits like Spark (``limit - 1``).
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    pat, lim = pattern, limit

    def _one(s: Any) -> list[str] | None:
        return _re_split_sparklike(s, pat, lim)

    return Column(expr.map_elements(_one, return_dtype=pl.List(pl.String)))


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


def substring(col_name: Union[str, Column], pos: int, length: int) -> Column:
    """
    Mimics pyspark.sql.functions.substring.

    ``pos`` is 1-based (negative values count from string end).
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    p, ln = pos, length

    def _one(v: Any) -> str | None:
        return _substring_sparklike(v, p, ln)

    return Column(expr.map_elements(_one, return_dtype=pl.String))


def _now_batch(s: pl.Series) -> pl.Series:
    if s.len() == 0:
        return pl.Series("now", [], dtype=pl.Datetime("us"))
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    return pl.Series("now", [ts] * s.len(), dtype=pl.Datetime("us"))


def now() -> Column:
    """
    Mimics pyspark.sql.functions.now: current timestamp (same value for all rows) at evaluation.

    Uses UTC wall time without tzinfo, comparable to many Spark :class:`TimestampType` outputs.
    """
    return Column(
        pl.int_range(0, pl.len(), dtype=pl.Int64, eager=False).map_batches(
            _now_batch,
            return_dtype=pl.Datetime("us"),
        )
    )


def current_timestamp() -> Column:
    """
    Mimics pyspark.sql.functions.current_timestamp.

    Same semantics as :func:`now` in sparkleframe: one UTC wall-clock timestamp shared by
    all rows in the batch at evaluation (microsecond precision, no tzinfo).
    """
    return now()


def current_date() -> Column:
    """
    Mimics pyspark.sql.functions.current_date.
    """
    today = date.today()
    return Column(pl.lit(today))


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


def date_sub(col_name: Union[str, Column], days: int) -> Column:
    """
    Mimics pyspark.sql.functions.date_sub.

    String arguments use the same Spark-like string-to-date rules as :func:`datediff` (ISO-8601
    ``T`` / ``Z`` in strings, not only ``yyyy-MM-dd``).
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(_as_date_sparklike_expr(expr) - pl.duration(days=int(days)))


def datediff(end: Union[str, Column], start: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.datediff.

    String end/start values are coerced to dates using rules closer to Spark ``cast(… as date)`` than
    a plain Polars string→date cast, so e.g. ISO-8601 ``…T…Z`` strings are handled like Spark.
    """
    end_expr = _to_expr(end) if isinstance(end, Column) else pl.col(end)
    start_expr = _to_expr(start) if isinstance(start, Column) else pl.col(start)
    e = _as_date_sparklike_expr(end_expr)
    s = _as_date_sparklike_expr(start_expr)
    return Column((e - s).dt.total_days().cast(pl.Int32))


def months_between(end: Union[str, Column], start: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.months_between.

    Uses a simplified Spark-like approximation for fractional months. String end/start values use
    the same date coercion as :func:`datediff`.
    """
    end_expr = _to_expr(end) if isinstance(end, Column) else pl.col(end)
    start_expr = _to_expr(start) if isinstance(start, Column) else pl.col(start)
    end_date = _as_date_sparklike_expr(end_expr)
    start_date = _as_date_sparklike_expr(start_expr)
    whole_months = (end_date.dt.year() - start_date.dt.year()) * 12 + (end_date.dt.month() - start_date.dt.month())
    day_fraction = (end_date.dt.day() - start_date.dt.day()) / pl.lit(31.0)
    return Column((whole_months + day_fraction).cast(pl.Float64))


def monotonically_increasing_id() -> Column:
    """
    Mimics pyspark.sql.functions.monotonically_increasing_id for a single in-memory partition.

    Yields 0, 1, 2, … in **current row order** (row index). Does not bit-pack a Spark
    partition id; multi-executor layout is not modeled.
    """
    return Column(pl.int_range(0, pl.len(), dtype=pl.Int64, eager=False))


def _rand_batch(s: pl.Series, seed: Optional[int]) -> pl.Series:
    n = s.len()
    if n == 0:
        return pl.Series(s.name, [], dtype=pl.Float64)
    rng = random.Random(seed) if seed is not None else random.Random()
    return pl.Series(s.name, [rng.random() for _ in range(n)], dtype=pl.Float64)


def rand(seed: Optional[int] = None) -> Column:
    """
    Mimics pyspark.sql.functions.rand: uniform random in ``[0.0, 1.0)`` per row.

    Without ``seed``, each evaluation uses an independent :class:`random.Random` instance
    (system-entropy seed). With ``seed``, draws are deterministic for a given row count within
    one process; values do not match Spark's JVM PRNG for the same seed.
    """

    def _batch(series: pl.Series) -> pl.Series:
        return _rand_batch(series, seed)

    return Column(
        pl.int_range(0, pl.len(), dtype=pl.Int64, eager=False).map_batches(
            _batch,
            return_dtype=pl.Float64,
        )
    )


def broadcast(df: Any) -> Any:
    """
    Mimics pyspark.sql.functions.broadcast.

    Sparkleframe runs in-process and has no join planner hints, so this is a no-op.
    """
    return df


def sort_array(col_name: Union[str, Column], asc: bool = True) -> Column:
    """
    Mimics pyspark.sql.functions.sort_array.

    Sorts each array in the column in ascending or descending order. Null arrays stay
    null; empty arrays stay empty.

    Args:
        col_name (str or Column): Array column.
        asc (bool): If true (default), sort ascending; if false, descending. PySpark 4's
            Python ``sort_array`` expects a boolean here (not a ``Column``).

    Returns:
        Column: Array column of sorted lists.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.list.sort(descending=not asc))


def array_contains(col_name: Union[str, Column], value: Union[str, Column, Any]) -> Column:
    """
    Mimics pyspark.sql.functions.array_contains.
    """
    array_expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    value_expr = _to_expr(value) if isinstance(value, Column) else pl.lit(value)
    return Column(array_expr.list.contains(value_expr))


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


def size(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.size for array/map-like values.

    Uses native ``list.len`` when the column resolves to a Polars ``List`` dtype (including
    under an active frame schema in :meth:`DataFrame.select` / :meth:`DataFrame.filter`).
    For ``Object`` columns that store Python ``list``/``dict`` cells — common for nested
    struct fields built from semi-structured data — falls back to a row-wise length so
    Spark-style ``size`` on dotted struct paths matches Spark behaviour instead of raising
    Polars' \"expected List data type for list operation, got: Object\".
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    sch = _polars_schema_ctx.get()
    use_list_fastpath = False
    if sch is not None:
        dt = _output_dtype_of_expr(expr, sch)
        if isinstance(dt, pl.List):
            use_list_fastpath = True
    if use_list_fastpath:
        return Column(pl.when(expr.is_null()).then(pl.lit(None)).otherwise(expr.list.len()).cast(pl.Int32))
    return Column(
        pl.when(expr.is_null())
        .then(pl.lit(None))
        .otherwise(expr.map_elements(_size_sparklike_value, return_dtype=pl.Int32))
    )


def filter(col_name: Union[str, Column], func: Callable[[Column], Any]) -> Column:
    """
    Mimics pyspark.sql.functions.filter for array columns.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    element_col = Column(pl.element())
    predicate = func(element_col)
    predicate_expr = predicate.to_native() if isinstance(predicate, Column) else _to_expr(predicate)
    return Column(expr.list.eval(pl.when(predicate_expr).then(pl.element()).otherwise(pl.lit(None))).list.drop_nulls())


def explode(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.explode.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    column = Column(expr.explode())
    setattr(column, "_is_explode", True)
    if isinstance(col_name, str):
        setattr(column, "_explode_source_name", col_name)
    return column


def _as_col_expr(col_name: Union[str, Column]) -> pl.Expr:
    return _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)


def concat(*cols: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.concat for string columns (null if any input is null).
    """
    if not cols:
        raise ValueError("concat requires at least one column")
    exprs = [_expr_as_string_for_compare(_as_col_expr(c)) for c in cols]
    return Column(pl.concat_str(exprs, separator="", ignore_nulls=False))


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


def struct(*cols: Any) -> Column:
    """
    Mimics pyspark.sql.functions.struct.

    Builds a struct column from column names and/or Column expressions. If a single
    list or set is passed, it is expanded like PySpark (3.4+).

    Empty ``struct()`` is not supported (PySpark fails at execution).

    Args:
        *cols: Column names (``str``), :class:`~sparkleframe.polarsdf.column.Column` values,
            or a single ``list`` / ``set`` of those.

    Returns:
        Column: A struct column whose field names follow Spark's ``CreateStruct`` rules.
    """
    expanded = _struct_expand_varargs(cols)
    if not expanded:
        raise ValueError("struct requires at least one column")
    parts = [_struct_named_child(c, i) for i, c in enumerate(expanded)]
    out = Column(pl.struct(parts))
    out._broadcast_row_count_in_select = bool(expanded) and all(
        isinstance(c, Column) and getattr(c, "_broadcast_row_count_in_select", False) for c in expanded
    )
    return out


def try_to_timestamp(
    col_name: Union[str, Column],
    fmt: Optional[str] = None,
) -> Column:
    """
    Mimics pyspark.sql.functions.try_to_timestamp (Spark 4+).

    If ``fmt`` is omitted, uses the same expression as one-arg :func:`to_timestamp`
    (see :func:`_to_timestamp_no_format_column`). If ``fmt`` is given, uses the same
    format-based parsing as :func:`to_timestamp`.

    Args:
        col_name (str or Column): Column with string values to convert to timestamps.
        fmt (str, optional): Spark datetime pattern, or ``None`` for one-arg rules.

    Returns:
        Column: A Column with values converted to Polars datetime type (null for failures).
    """
    if fmt is None:
        return _to_timestamp_no_format_column(col_name)
    return _to_datetime_column(col_name, fmt)


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


def try_to_date(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    """
    Mimics pyspark.sql.functions.try_to_date (Spark 4+).

    Converts a string column to a date, returning null for unparseable values.

    Args:
        col_name (str or Column): Column with string values to convert to dates.
        fmt (str, optional): The date format. Defaults to 'yyyy-MM-dd'.

    Returns:
        Column: A Column with values converted to Polars Date type (null for failures).
    """
    fmt = fmt or "yyyy-MM-dd"
    strftime_fmt = _convert_spark_date_format(fmt)
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    parsed_from_string = expr.cast(pl.String, strict=False).str.strptime(pl.Date, strftime_fmt, strict=False)
    cast_direct = expr.cast(pl.Date, strict=False)
    return Column(pl.coalesce(cast_direct, parsed_from_string))


def try_element_at(col_name: Union[str, Column], extraction: Union[str, int, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.try_element_at (Spark 4+).

    For arrays: uses 1-based indexing (positive and negative). Returns null
    for out-of-bounds access instead of raising.

    For maps (materialized as List(Struct(key, value))): looks up the key and
    returns null when absent.

    Args:
        col_name (str or Column): The array or map column.
        extraction (str, int, or Column): The index (1-based int) for arrays,
            or the key (str / Column) for maps.

    Returns:
        Column: A Column with the extracted element, or null on failure.
    """
    col_expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)

    if isinstance(extraction, Column):
        extraction = extraction.to_native()

    if isinstance(extraction, pl.Expr):

        def _lookup_dynamic(value_and_key: dict[str, Any]) -> Any:
            value = value_and_key.get("value")
            key = value_and_key.get("key")
            if value is None or key is None:
                return None
            if isinstance(value, dict):
                return value.get(key)
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict) and entry.get("key") == key:
                        return entry.get("value")
            return None

        return Column(
            pl.struct([col_expr.alias("value"), extraction.alias("key")]).map_elements(
                _lookup_dynamic, return_dtype=pl.String
            )
        )

    if isinstance(extraction, int):
        # Spark uses 1-based indexing; index 0 is invalid -> null
        if extraction == 0:
            return Column(pl.lit(None))
        polars_idx = extraction - 1 if extraction > 0 else extraction
        return Column(col_expr.list.get(polars_idx, null_on_oob=True))

    if isinstance(extraction, str):
        try:
            dt = col_expr.meta.output_dtype()
        except Exception:
            dt = None
        if isinstance(dt, pl.Struct) and extraction in (f.name for f in dt.fields):
            return Column(col_expr.struct.field(extraction))
        if isinstance(dt, pl.List) and isinstance(getattr(dt, "inner", None), pl.Struct):
            inner_s = dt.inner
            if "key" in {f.name for f in inner_s.fields} and "value" in {f.name for f in inner_s.fields}:
                return Column(
                    col_expr.list.eval(
                        pl.when(pl.element().struct.field("key") == pl.lit(extraction)).then(
                            pl.element().struct.field("value")
                        )
                    )
                    .list.drop_nulls()
                    .list.first()
                )

        def _lookup_static(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, dict):
                return value.get(extraction)
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict) and entry.get("key") == extraction:
                        return entry.get("value")
            return None

        return Column(col_expr.map_elements(_lookup_static, return_dtype=pl.Object))

    raise TypeError(f"try_element_at extraction must be int, str, or Column, got {type(extraction).__name__}")


def element_at(col_name: Union[str, Column], extraction: Union[str, int, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.element_at.

    Sparkleframe shares the same null-safe behavior implemented for try_element_at.
    """
    return try_element_at(col_name, extraction)


def uuid() -> Column:
    """
    Mimics :func:`pyspark.sql.functions.uuid` (Spark 4.1+), unseeded form only.

    One random canonical UUID string per row via :func:`uuid.uuid4` (not Spark’s JVM
    output). ``uuid(seed=…)`` is not supported in sparkleframe.
    """

    def _row_uuid4(_: Any) -> str:
        return str(uuid4())

    return Column(
        pl.int_range(0, pl.len(), dtype=pl.Int64, eager=False).map_elements(
            _row_uuid4,
            return_dtype=pl.String,
        )
    )
