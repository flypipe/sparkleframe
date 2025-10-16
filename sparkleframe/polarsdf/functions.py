from __future__ import annotations
from typing import Union, Any

import polars as pl

from sparkleframe.polarsdf import WindowSpec
from sparkleframe.polarsdf.column import Column, _to_expr
from sparkleframe.polarsdf.functions_utils import _RankWrapper


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


def to_timestamp(col_name: Union[str, Column], fmt: str = "yyyy-MM-dd HH:mm:ss") -> Column:
    """
    Mimics pyspark.sql.functions.to_timestamp.

    Converts a string column to a timestamp using the specified format.

    Args:
        col_name (str or Column): Column with string values to convert to timestamps.
        fmt (str): The timestamp format to parse the strings. Defaults to 'yyyy-MM-dd HH:mm:ss'.

    Returns:
        Column: A Column with values converted to Polars datetime type.
    """
    # Convert Spark-style format to strftime-style for Polars

    format_map = [
        ("yyyy", "%Y"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("HH", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
        (".SSSSSS", ".%6f"),  # microseconds
        (".SSSSS", ".%6f"),
        (".SSSS", ".%6f"),
        (".SSS", ".%6f"),  # also treated as microseconds, will pad
        (".SS", ".%6f"),
        (".S", ".%6f"),
    ]
    # Pad fractional seconds to 6 digits (microseconds)
    for spark_fmt, strftime_fmt in format_map:
        fmt = fmt.replace(spark_fmt, strftime_fmt)

    expr = _to_expr(col_name) if isinstance(col_name, Column) else pl.col(col_name)

    # Normalize fractional seconds (pad with trailing zeros to make 6 digits)
    # appends zeros to fractional part (e.g., .993 -> .993000)
    if "%6f" in fmt:
        # Pad fractional seconds using a map function
        def pad_microseconds(val):
            if val is None:
                return None
            if "." in val:
                prefix, suffix = val.split(".", 1)
                suffix = (suffix + "000000")[:6]  # Ensure exactly 6 digits
                return f"{prefix}.{suffix}"
            return val

        expr = expr.map_elements(pad_microseconds, return_dtype=pl.String)

    return Column(expr.str.strptime(pl.Datetime, fmt))


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
