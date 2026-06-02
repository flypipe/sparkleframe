from __future__ import annotations

import base64
import json
import math
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
from pyspark.sql.types import StructType as SparkStructType

from sparkleframe.polarsdf import DataFrame
from sparkleframe.polarsdf.types import polars_dtype_to_spark_ddl_name
from sparkleframe.pythondf.dataframe import DataFrame as PythonDataFrame


def _ddl_schema_from_polars_frame(frame: pl.DataFrame) -> str:
    parts: list[str] = []
    for name in frame.columns:
        sql_type = polars_dtype_to_spark_ddl_name(frame.schema[name])
        parts.append(f"{name} {sql_type}")
    return ", ".join(parts)


def spark_rows_from_dict(data: dict[str, list[Any]]) -> list[tuple[Any, ...]]:
    """
    Column-oriented dict -> row tuples for Spark, preserving key order as column order.

    Keeps Python None as None in each row (avoids pandas object-column NaN coercion).

    Usage:
        spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
    """
    if not data:
        return []
    return list(zip(*[data[name] for name in data.keys()]))


def create_spark_df(
    spark,
    df: Union[pl.DataFrame, DataFrame],
    schema: Optional[SparkStructType] = None,
) -> SparkDataFrame:
    """
    Convert a Polars or SparkleFrame-backed frame to a PySpark DataFrame.

    Args:
        spark: Active SparkSession.
        df: Polars DataFrame or SparkleFrame DataFrame.
        schema: Optional PySpark StructType. Use when inference is wrong for a test case;
            omit for normal null-safe tuple conversion with column names.
    """
    native = df.to_native_df() if isinstance(df, DataFrame) else df
    rows = native.to_dicts()
    cols = list(native.columns)

    if not rows:
        if schema is not None:
            return spark.createDataFrame([], schema)
        if cols:
            return spark.createDataFrame([], _ddl_schema_from_polars_frame(native))
        return spark.createDataFrame(pd.DataFrame(native.to_arrow().to_pandas()))

    row_tuples = [tuple(r[c] for c in cols) for r in rows]
    if schema is not None:
        return spark.createDataFrame(row_tuples, schema)
    return spark.createDataFrame(row_tuples, cols)


def _remove_nulls_from_dict_list(data):
    """Recursively remove keys with null/NaN values from a list of dicts."""

    def is_null(x):
        if x is None:
            return True

        if isinstance(x, (np.ndarray, pd.Series, list)):
            return all(is_null(el) for el in x)

        try:
            return bool(pd.isna(x)) or (isinstance(x, float) and math.isnan(x))
        except Exception:
            return False

    def clean_value(v):
        if isinstance(v, dict):
            return {k: clean_value(val) for k, val in v.items() if not is_null(val)}
        if isinstance(v, list):
            return [clean_value(x) for x in v]
        return v

    return [clean_value(d) for d in data]


_FLOAT_SIG_DIGITS = 12


def _round_float_sigfigs(f: float) -> float:
    """Round a float to ``_FLOAT_SIG_DIGITS`` significant figures.

    This absorbs last-ULP differences between JVM (Spark) and Rust (Polars)
    float64 math while still catching real divergences.
    """
    if f == 0.0:
        return 0.0
    magnitude = math.floor(math.log10(abs(f))) + 1
    return round(f, _FLOAT_SIG_DIGITS - magnitude)


def _normalize_compare_value(value: Any) -> Any:
    """
    Convert a Spark/Polars record value into a JSON-friendly form that preserves the
    distinction between integers and floats (so ``2.5 * 2 = 5.0`` does not collapse to ``5``).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if v is None:
                continue
            normalized = _normalize_compare_value(v)
            if normalized is None:
                continue
            out[k] = normalized
        return out
    if isinstance(value, (list, tuple)):
        return [_normalize_compare_value(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, Decimal):
        return _round_float_sigfigs(float(value))
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, datetime):
        ts = value.replace(tzinfo=None) if value.tzinfo is not None else value
        return ts.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.floating,)):
        f = float(value)
        if math.isnan(f):
            return None
        return _round_float_sigfigs(f)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return _round_float_sigfigs(value)
    if isinstance(value, int):
        return value
    return value


def _records_from_spark(df: SparkDataFrame) -> list[Any]:
    """Type-faithful record extraction from a Spark DataFrame (preserves int/float)."""
    return [_normalize_compare_value(row.asDict(recursive=True)) for row in df.collect()]


def _records_from_sparkle(df: Union[DataFrame, pl.DataFrame, PythonDataFrame]) -> list[Any]:
    """
    Type-faithful record extraction from a sparkleframe (polarsdf or pythondf) or Polars DataFrame.
    Uses each backend's native row iterator to avoid pandas dtype coercion (e.g. nullable int
    columns becoming Float64 with NaN).
    """
    if isinstance(df, PythonDataFrame):
        return [_normalize_compare_value(row) for row in df.collect()]
    native = df.to_native_df() if isinstance(df, DataFrame) else df
    return [_normalize_compare_value(row) for row in native.to_dicts()]


def _frame_row_count(df) -> int:
    if isinstance(df, PythonDataFrame):
        return len(df)
    return df.count()


def _get_records(df) -> list[Any]:
    if isinstance(df, SparkDataFrame):
        return _records_from_spark(df)
    return _records_from_sparkle(df)


def _sorted_records_json(records: list[Any]) -> str:
    """Stringify records and sort for set-equality comparison (order-insensitive)."""
    per_row = [json.dumps(r, sort_keys=True, default=str) for r in records]
    per_row.sort()
    return "[" + ",".join(per_row) + "]"


def _get_json_from_dataframe(df):
    """Order-insensitive JSON representation of a DataFrame for parity comparison."""
    return _sorted_records_json(_get_records(df))


def assert_sparkle_spark_frame_are_equal(
    df1: Union[SparkDataFrame, DataFrame, PythonDataFrame],
    df2: Union[SparkDataFrame, DataFrame, PythonDataFrame],
) -> bool:
    assert type(df1) is not type(df2)
    assert _frame_row_count(df1) == _frame_row_count(df2), (
        f"row count mismatch: {_frame_row_count(df1)} vs {_frame_row_count(df2)}"
    )
    json_df1 = _get_json_from_dataframe(df1)
    json_df2 = _get_json_from_dataframe(df2)
    assert json_df1 == json_df2, f"""
{json_df1}
vs
{json_df2}"""

    return True


def assert_frame_ordered_equal(
    df1: Union[SparkDataFrame, DataFrame, PythonDataFrame],
    df2: Union[SparkDataFrame, DataFrame, PythonDataFrame],
) -> bool:
    """Strict positional equality — use when row order is significant (after orderBy)."""
    assert type(df1) is not type(df2)
    assert _frame_row_count(df1) == _frame_row_count(df2), (
        f"row count mismatch: {_frame_row_count(df1)} vs {_frame_row_count(df2)}"
    )
    json_df1 = json.dumps(_get_records(df1), sort_keys=True, default=str)
    json_df2 = json.dumps(_get_records(df2), sort_keys=True, default=str)
    assert json_df1 == json_df2, f"""
{json_df1}
vs
{json_df2}"""
    return True
