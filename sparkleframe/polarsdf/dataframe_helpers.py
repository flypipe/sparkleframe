from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from sparkleframe.polarsdf import types as sft
from sparkleframe.polarsdf.types import (
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
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from sparkleframe.polarsdf.types_utils import _MapTypeUtils

# ---------------------------------------------------------------------------
# toPandas conversion helpers
# ---------------------------------------------------------------------------


def _is_null_pandas(x: Any) -> bool:
    try:
        return pd.isna(x) and not isinstance(x, (str, bytes))
    except Exception:
        return False


def _convert_number(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    if isinstance(x, (np.floating, float)):
        if math.isnan(x):
            return None
        return float(x)
    return x


def _is_kv(d: Any) -> bool:
    return isinstance(d, dict) and "key" in d and "value" in d


def _kv_list_to_dict(kv_list: list, converter: Callable[[Any], Any]) -> dict:
    out: dict = {}
    for item in kv_list:
        k = converter(item["key"])
        v = converter(item.get("value"))
        if v is not None:
            out[k] = v
    return out


def convert_pandas_value(val: Any) -> Any:
    """Recursively convert a single pandas cell value to JSON-friendly Python types."""
    if _is_null_pandas(val):
        return None

    if isinstance(val, np.ndarray):
        return [convert_pandas_value(v) for v in val.tolist()]

    if isinstance(val, list):
        if val and all(_is_kv(item) for item in val):
            return _kv_list_to_dict(val, convert_pandas_value)

        if val and all(isinstance(item, list) and (not item or all(_is_kv(x) for x in item)) for item in val):
            return [_kv_list_to_dict(item, convert_pandas_value) for item in val]

        return [convert_pandas_value(v) for v in val]

    if isinstance(val, tuple):
        return [convert_pandas_value(v) for v in val]

    if isinstance(val, dict):
        out: dict = {}
        for k, v in val.items():
            cv = convert_pandas_value(v)
            if cv is not None:
                out[k] = cv
        return out

    return _convert_number(val)


# ---------------------------------------------------------------------------
# Join helpers
# ---------------------------------------------------------------------------

PYSPARK_TO_POLARS_JOIN_MAP = {
    "inner": "inner",
    "cross": "cross",
    "outer": "full",
    "full": "full",
    "fullouter": "full",
    "full_outer": "full",
    "left": "left",
    "leftouter": "left",
    "left_outer": "left",
    "right": "right",
    "rightouter": "right",
    "right_outer": "right",
    "semi": "semi",
    "leftsemi": "semi",
    "left_semi": "semi",
    "anti": "anti",
    "leftanti": "anti",
    "left_anti": "anti",
}


def perform_expression_join(
    left_df: pl.DataFrame,
    right_df: pl.DataFrame,
    predicate: pl.Expr,
    how: str,
    suffix: str,
) -> pl.DataFrame:
    """Cross-join + filter implementation for boolean join predicates."""
    left_with_idx = left_df.with_row_index("__sf_left_idx")
    matched = left_with_idx.join(right_df, how="cross", suffix=suffix).filter(predicate)

    if how in {"inner"}:
        return matched.drop("__sf_left_idx")

    matched_ids = matched.select("__sf_left_idx").unique()
    unmatched = left_with_idx.join(matched_ids, on="__sf_left_idx", how="anti")

    unmatched_right_cols = []
    for right_col in right_df.columns:
        target_col = right_col if right_col not in left_df.columns else f"{right_col}{suffix}"
        unmatched_right_cols.append(pl.lit(None).alias(target_col))
    unmatched = unmatched.with_columns(unmatched_right_cols).drop("__sf_left_idx")
    return pl.concat([matched.drop("__sf_left_idx"), unmatched], how="diagonal_relaxed")


def resolve_equi_join_keys(on: list) -> Union[str, list[str]]:
    """Extract column name strings from a list of equi-join key expressions."""
    key_names: list[str] = []
    for x in on:
        if isinstance(x, str):
            key_names.append(x)
        elif isinstance(x, pl.Expr) and x.meta.is_column():
            key_names.append(x.meta.output_name().split(".")[-1])
        else:
            raise TypeError(f"Unsupported equi-join key expression: {x!r}")
    return key_names[0] if len(key_names) == 1 else key_names


def coalesce_outer_join_keys(
    result: pl.DataFrame,
    suffix: str,
    on_column_wrappers: bool,
) -> pl.DataFrame:
    """
    Coalesce or drop suffixed key columns produced by a full-outer join.

    Polars retains both left and right keys explicitly.  PySpark coalesces
    when join keys are plain strings, and drops when they are Column wrappers.
    """
    for col_name in result.columns:
        if col_name.endswith(suffix):
            if on_column_wrappers:
                result = result.drop(col_name)
            else:
                result = result.with_columns(
                    pl.coalesce(col_name.replace(suffix, ""), col_name).alias(col_name.replace(suffix, ""))
                ).drop(col_name)
    return result


# ---------------------------------------------------------------------------
# Schema / dtypes helpers
# ---------------------------------------------------------------------------

POLARS_TO_PYSPARK_DTYPE_MAP = {
    pl.Int8: "tinyint",
    pl.Int16: "smallint",
    pl.Int32: "int",
    pl.Int64: "bigint",
    pl.UInt8: "tinyint",
    pl.UInt16: "smallint",
    pl.UInt32: "int",
    pl.UInt64: "bigint",
    pl.Float32: "float",
    pl.Float64: "double",
    pl.Boolean: "boolean",
    pl.Utf8: "string",
    pl.Date: "date",
    pl.Datetime: "timestamp",
    pl.Time: "time",
    pl.Duration: "interval",
    pl.Object: "binary",
    pl.List: "array",
    pl.Struct: "struct",
    pl.Decimal: "decimal",
    pl.Binary: "binary",
}


def map_polars_dtype_to_spark_name(dtype: pl.DataType) -> str:
    """Map a Polars dtype to PySpark-style string name (for ``DataFrame.dtypes``)."""
    if isinstance(dtype, pl.Decimal):
        return f"decimal({dtype.precision},{dtype.scale})"

    if isinstance(dtype, pl.Struct):
        fields_str = ",".join(f"{field.name}:{map_polars_dtype_to_spark_name(field.dtype)}" for field in dtype.fields)
        return f"struct<{fields_str}>"

    for polars_type, spark_type in POLARS_TO_PYSPARK_DTYPE_MAP.items():
        if isinstance(dtype, polars_type):
            return spark_type

    return str(dtype)


POLARS_TO_SPARK_SCALARS = {
    pl.Null: StringType(),
    pl.Utf8: StringType(),
    pl.Int32: IntegerType(),
    pl.UInt32: IntegerType(),
    pl.Int64: LongType(),
    pl.UInt64: LongType(),
    pl.Float32: FloatType(),
    pl.Float64: DoubleType(),
    pl.Boolean: BooleanType(),
    pl.Date: DateType(),
    pl.Datetime: TimestampType(),
    pl.Int8: ByteType(),
    pl.UInt8: ByteType(),
    pl.Int16: ShortType(),
    pl.UInt16: ShortType(),
    pl.Binary: BinaryType(),
}


def _declared_type_for(
    col_name: str,
    declared_schema: Optional[Union[DataType, StructType]] = None,
) -> Optional[DataType]:
    """Lookup the user-declared DataType for *col_name* in the construction schema."""
    if isinstance(declared_schema, StructType):
        for f in declared_schema:
            if f.name == col_name:
                return f.dataType
    return None


def to_spark_datatype(
    dtype: pl.DataType,
    declared_schema: Optional[Union[DataType, StructType]] = None,
    col_name: Optional[str] = None,
) -> DataType:
    """Convert a Polars dtype to a sparkleframe ``DataType``."""
    if _MapTypeUtils.is_map_dtype(dtype):
        key_dt_pl = dtype.inner.fields[0].dtype
        val_dt_pl = dtype.inner.fields[1].dtype
        key_dt = to_spark_datatype(key_dt_pl)
        val_dt = to_spark_datatype(val_dt_pl)

        value_contains_null = True
        if col_name is not None:
            decl = _declared_type_for(col_name, declared_schema)
            if isinstance(decl, sft.MapType):
                value_contains_null = decl.valueContainsNull

        return sft.MapType(key_dt, val_dt, valueContainsNull=value_contains_null)

    if isinstance(dtype, pl.Decimal):
        return DecimalType(dtype.precision, dtype.scale)

    if isinstance(dtype, pl.Struct):
        nested_fields = [StructField(f.name, to_spark_datatype(f.dtype)) for f in dtype.fields]
        return StructType(nested_fields)

    if isinstance(dtype, pl.List):
        elem_dtype = to_spark_datatype(dtype.inner)
        contains_null = True
        if col_name is not None:
            decl = _declared_type_for(col_name, declared_schema)
            if isinstance(decl, sft.ArrayType):
                contains_null = decl.containsNull
        return sft.ArrayType(elem_dtype, containsNull=contains_null)

    for pl_type, spark_type in POLARS_TO_SPARK_SCALARS.items():
        if isinstance(dtype, pl_type):
            return spark_type

    raise TypeError(f"Unsupported dtype '{dtype}'")


def polars_dtype_to_spark_structfield(
    name: str,
    dtype: pl.DataType,
    declared_schema: Optional[Union[DataType, StructType]] = None,
) -> StructField:
    """Convert a Polars column name + dtype into a sparkleframe ``StructField``."""
    decl = _declared_type_for(name, declared_schema)
    if isinstance(decl, sft.ArrayType) and isinstance(decl.elementType, sft.MapType):
        return StructField(name, decl)
    if isinstance(decl, sft.MapType):
        return StructField(name, decl)
    return StructField(name, to_spark_datatype(dtype, declared_schema=declared_schema, col_name=name))
