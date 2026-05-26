from __future__ import annotations

import json
from datetime import date
from typing import Any, Callable, Optional, Union
from uuid import uuid4

import polars as pl

from sparkleframe.polarsdf import WindowSpec
from sparkleframe.polarsdf.column import Column, _to_expr
from sparkleframe.polarsdf.functions_helpers import (
    _as_date_sparklike_expr,
    _coerce_json_value,
    _md5_sparklike,
    _now_batch,
    _RankWrapper,
    _re_split_sparklike,
    _schema_from_string,
    _substring_sparklike,
    _to_date_column,
    _to_datetime_column,
    _to_timestamp_no_format_column,
    element_at_column,
)
from sparkleframe.polarsdf.types import DataType


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
        # Defer struct navigation through Column so _apply_getitem_key runs under the
        # active frame schema (null-safe / missing-field parity with Spark).
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


def lit(value) -> Column:
    """
    Mimics pyspark.sql.functions.lit.

    Creates a Column of literal value broadcast to the source DataFrame's row count.

    Spark's ``lit(x)`` produces *one row per source row*; Polars' bare ``pl.lit(x)`` would
    collapse to a single row. We use :func:`polars.repeat` with ``pl.len()`` so the
    literal expands to the source row count (including the empty-frame case where Spark
    yields zero rows -- the previous ``repeat_by(pl.len()).explode()`` trick yielded one
    null row for empty inputs).

    Args:
        value: A literal value (int, float, str, bool, None, etc.)

    Returns:
        Column: A Column object wrapping a literal Polars expression.
    """
    if value is None:
        # Spark ``lit(None)`` has ``StringType`` by default; mirror that.
        return Column(pl.repeat(None, pl.len(), dtype=pl.String))
    return Column(pl.repeat(value, pl.len()))


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


def first(col_name: Union[str, Column], ignorenulls: bool = False) -> Column:
    """Mimics pyspark.sql.functions.first."""
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(expr.drop_nulls().first() if ignorenulls else expr.first())


def map_from_entries(col_name: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.map_from_entries.

    Expects an array of structs with ``key`` and ``value`` fields.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)

    def _entries_to_map(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            out: dict[Any, Any] = {}
            for entry in value:
                if isinstance(entry, dict) and "key" in entry and "value" in entry:
                    out[entry["key"]] = entry["value"]
            return out
        return None

    return Column(expr.map_elements(_entries_to_map, return_dtype=pl.Object))


def map_keys(col_name: Union[str, Column]) -> Column:
    """Mimics pyspark.sql.functions.map_keys."""
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)

    def _keys(value: Any) -> Any:
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
        return _to_timestamp_no_format_column(col_name, strict=True)
    return _to_datetime_column(col_name, fmt, strict=True)


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


def current_date() -> Column:
    """Mimics pyspark.sql.functions.current_date."""
    return Column(pl.lit(date.today()))


def date_sub(col_name: Union[str, Column], days: int) -> Column:
    """
    Mimics pyspark.sql.functions.date_sub.

    String arguments use the same Spark-like string-to-date rules as :func:`datediff`.
    """
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(_as_date_sparklike_expr(expr) - pl.duration(days=int(days)))


def datediff(end: Union[str, Column], start: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.datediff.

    String end/start values are coerced to dates using rules closer to Spark ``cast(… as date)``.
    """
    end_expr = _to_expr(end) if isinstance(end, Column) else pl.col(end)
    start_expr = _to_expr(start) if isinstance(start, Column) else pl.col(start)
    e = _as_date_sparklike_expr(end_expr)
    s = _as_date_sparklike_expr(start_expr)
    return Column((e - s).dt.total_days().cast(pl.Int32))


def months_between(end: Union[str, Column], start: Union[str, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.months_between.

    Uses a simplified Spark-like approximation for fractional months.
    """
    end_expr = _to_expr(end) if isinstance(end, Column) else pl.col(end)
    start_expr = _to_expr(start) if isinstance(start, Column) else pl.col(start)
    end_date = _as_date_sparklike_expr(end_expr)
    start_date = _as_date_sparklike_expr(start_expr)
    whole_months = (end_date.dt.year() - start_date.dt.year()) * 12 + (end_date.dt.month() - start_date.dt.month())
    day_fraction = (end_date.dt.day() - start_date.dt.day()) / pl.lit(31.0)
    return Column((whole_months + day_fraction).cast(pl.Float64))


def broadcast(df: Any) -> Any:
    """
    Mimics pyspark.sql.functions.broadcast.

    Sparkleframe runs in-process and has no join planner hints, so this is a no-op.
    """
    return df


def array_contains(col_name: Union[str, Column], value: Union[str, Column, Any]) -> Column:
    """Mimics pyspark.sql.functions.array_contains."""
    array_expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    value_expr = _to_expr(value) if isinstance(value, Column) else pl.lit(value)
    return Column(array_expr.list.contains(value_expr))


def size(col_name: Union[str, Column]) -> Column:
    """Mimics pyspark.sql.functions.size for array/map-like values."""
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    return Column(pl.when(expr.is_null()).then(pl.lit(None)).otherwise(expr.list.len()).cast(pl.Int32))


def filter(col_name: Union[str, Column], func: Callable[[Column], Any]) -> Column:
    """Mimics pyspark.sql.functions.filter for array columns."""
    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)
    element_col = Column(pl.element())
    predicate = func(element_col)
    predicate_expr = predicate.to_native() if isinstance(predicate, Column) else _to_expr(predicate)
    return Column(expr.list.eval(pl.when(predicate_expr).then(pl.element()).otherwise(pl.lit(None))).list.drop_nulls())


def explode(col_name: Union[str, Column]) -> Column:
    """Mimics pyspark.sql.functions.explode."""
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
    try:
        serialized = expr.meta.serialize()
        # ``lit(value)`` builds either ``pl.repeat(value, pl.len())`` (current) or the
        # legacy ``pl.lit(value).repeat_by(pl.len()).explode()`` -- both broadcast a
        # plain literal to the source row count and Spark names them ``col1``, ``col2``.
        if b"Repeat" in serialized:
            return f"col{index + 1}"
    except Exception:
        # Expressions containing Python UDFs (``map_batches``) can fail to serialize
        # without ``cloudpickle`` installed. Such expressions are never plain
        # broadcast literals, so fall through to the regular naming rules.
        pass
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
        return _to_timestamp_no_format_column(col_name, strict=False)
    return _to_datetime_column(col_name, fmt, strict=False)


def to_date(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    """
    Mimics pyspark.sql.functions.to_date.

    Converts a string column to a date using the given Spark format pattern.
    Malformed values raise under Spark 4 ANSI defaults (see :func:`try_to_date`).

    Args:
        col_name (str or Column): Column with string values to convert to dates.
        fmt (str, optional): The date format. Defaults to ``yyyy-MM-dd``.

    Returns:
        Column: A Column with values converted to Polars Date type.
    """
    fmt = fmt or "yyyy-MM-dd"
    return _to_date_column(col_name, fmt, strict=True)


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
    return _to_date_column(col_name, fmt, strict=False)


def try_element_at(col_name: Union[str, Column], extraction: Union[str, int, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.try_element_at (Spark 4+).

    For arrays: uses 1-based indexing (positive and negative). Returns null
    for out-of-bounds access instead of raising.

    For maps (materialized as List(Struct(key, value))): looks up the key and
    returns null when absent. A string ``extraction`` is interpreted as a **column
    name** (Spark SPARK-48766), not a literal key; use :func:`lit` wrapped in
    :class:`~sparkleframe.polarsdf.column.Column` for a literal map key.

    Args:
        col_name (str or Column): The array or map column.
        extraction (str, int, or Column): The index (1-based int) for arrays,
            or the key column name (str) / key expression (Column) for maps.

    Returns:
        Column: A Column with the extracted element, or null on failure.
    """
    return element_at_column(col_name, extraction, strict=False)


def element_at(col_name: Union[str, Column], extraction: Union[str, int, Column]) -> Column:
    """
    Mimics pyspark.sql.functions.element_at.

    Spark 4 with ANSI enabled: invalid array indices raise
    ``SparkArrayIndexOutOfBoundsException``; use :func:`try_element_at` for null
    on failure. A string ``extraction`` on a map is a **literal key** (unlike
    :func:`try_element_at`, which treats a string as a column name).
    """
    return element_at_column(col_name, extraction, strict=True)


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
