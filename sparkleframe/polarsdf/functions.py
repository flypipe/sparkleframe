from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Optional, Union
from uuid import uuid4

import polars as pl

from sparkleframe.polarsdf import WindowSpec
from sparkleframe.polarsdf.column import Column, _to_expr
from sparkleframe.polarsdf.functions_utils import _RankWrapper
from sparkleframe.polarsdf.types import TimestampType


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
        expr = pl.col(parts[0])
        for seg in parts[1:]:
            expr = expr.struct.field(seg)
        return Column(expr)  # pass a Polars Expr directly
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


def lit(value) -> Column:
    """
    Mimics pyspark.sql.functions.lit.

    Creates a Column of literal value.

    Args:
        value: A literal value (int, float, str, bool, None, etc.)

    Returns:
        Column: A Column object wrapping a literal Polars expression.
    """
    if value is None:
        return Column(pl.lit(value).cast(pl.String).repeat_by(pl.len()).explode())
    return Column(pl.lit(value).repeat_by(pl.len()).explode())


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


def count(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.count.

    Counts the number of non-null elements for the specified column.

    Args:
        col_name (str or Column): The column to count non-null values in.

    Returns:
        Column: A Column representing the count aggregation expression.
    """
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
    expr = _to_expr(col_name)
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


def monotonically_increasing_id() -> Column:
    """
    Mimics pyspark.sql.functions.monotonically_increasing_id for a single in-memory partition.

    Yields 0, 1, 2, … in **current row order** (row index). Does not bit-pack a Spark
    partition id; multi-executor layout is not modeled.
    """
    return Column(pl.int_range(0, pl.len(), dtype=pl.Int64, eager=False))


def _as_col_expr(col_name: Union[str, Column]) -> pl.Expr:
    return _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)


def concat(*cols: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.concat for string columns (null if any input is null).
    """
    if not cols:
        raise ValueError("concat requires at least one column")
    exprs = [_as_col_expr(c).cast(pl.String, strict=False) for c in cols]
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
    return Column(pl.struct(parts))


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
        # Dynamic column-based key lookup for maps: list.eval pattern
        return Column(
            col_expr.list.eval(
                pl.when(pl.element().struct.field("key") == extraction).then(pl.element().struct.field("value"))
            )
            .list.drop_nulls()
            .list.first()
        )

    if isinstance(extraction, int):
        # Spark uses 1-based indexing; index 0 is invalid -> null
        if extraction == 0:
            return Column(pl.lit(None))
        polars_idx = extraction - 1 if extraction > 0 else extraction
        return Column(col_expr.list.get(polars_idx, null_on_oob=True))

    if isinstance(extraction, str):
        # Map key lookup: List(Struct(key, value)) layout
        return Column(
            col_expr.list.eval(
                pl.when(pl.element().struct.field("key") == pl.lit(extraction)).then(
                    pl.element().struct.field("value")
                )
            )
            .list.drop_nulls()
            .list.first()
        )

    raise TypeError(f"try_element_at extraction must be int, str, or Column, got {type(extraction).__name__}")


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
