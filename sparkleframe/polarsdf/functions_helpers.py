from __future__ import annotations

from typing import Optional, Union

import polars as pl

from sparkleframe.polarsdf.column import Column, _to_expr
from sparkleframe.polarsdf.types import TimestampType

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
