from __future__ import annotations

import json
import math
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
from pyspark.sql.types import StructType as SparkStructType

from sparkleframe.polarsdf import DataFrame


def _polars_dtype_to_spark_sql_type(dtype: pl.DataType) -> str:
    """Map Polars dtypes to Spark SQL type names for empty-frame DDL (``createDataFrame([], ddl)``)."""
    if dtype == pl.Null:
        return "void"
    if dtype == pl.Int8:
        return "tinyint"
    if dtype == pl.Int16:
        return "smallint"
    if dtype == pl.Int32:
        return "int"
    if dtype == pl.Int64:
        return "bigint"
    if dtype == pl.UInt8:
        return "smallint"
    if dtype == pl.UInt16:
        return "int"
    if dtype == pl.UInt32:
        return "bigint"
    if dtype == pl.UInt64:
        return "decimal(20,0)"
    if dtype == pl.Float32:
        return "float"
    if dtype == pl.Float64:
        return "double"
    if dtype in (pl.Utf8, pl.String):
        return "string"
    if dtype == pl.Boolean:
        return "boolean"
    if dtype == pl.Binary:
        return "binary"
    if dtype == pl.Date:
        return "date"
    if isinstance(dtype, pl.Datetime):
        return "timestamp"
    if isinstance(dtype, pl.Duration):
        return "bigint"
    if isinstance(dtype, pl.Decimal):
        return f"decimal({dtype.precision},{dtype.scale})"
    if isinstance(dtype, (pl.List, pl.Array, pl.Struct, pl.Object)):
        return "string"
    return "string"


def _ddl_schema_from_polars_frame(frame: pl.DataFrame) -> str:
    parts: list[str] = []
    for name in frame.columns:
        sql_type = _polars_dtype_to_spark_sql_type(frame.schema[name])
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
        # Handle None directly
        if x is None:
            return True

        # Handle numpy/pandas array-like
        if isinstance(x, (np.ndarray, pd.Series, list)):
            # if it's an array, consider it null only if *all* elements are null
            return all(is_null(el) for el in x)

        # Handle NaN and pd.NA safely
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


def _get_json_from_dataframe(df):
    if isinstance(df, SparkDataFrame):
        return json.dumps([json.loads(c) for c in df.toJSON().collect()], sort_keys=True)
    else:
        return json.dumps(_remove_nulls_from_dict_list(df.toPandas().to_dict(orient="records")), sort_keys=True)


def assert_sparkle_spark_frame_are_equal(
    df1: Union[SparkDataFrame, DataFrame], df2: Union[SparkDataFrame, DataFrame]
) -> bool:
    assert type(df1) is not type(df2)
    assert df1.count() == df2.count()
    json_df1 = _get_json_from_dataframe(df1)
    json_df2 = _get_json_from_dataframe(df2)
    assert json_df1 == json_df2, f"""
{json_df1}
vs
{json_df2}"""

    return True
