import json
import math
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from sparkleframe.polarsdf import DataFrame
from pyspark.sql.dataframe import DataFrame as SparkDataFrame


def to_records(column_dict: dict) -> list[dict]:
    """
    Converts a column-based dictionary into a list of row-based dictionaries.

    Args:
        column_dict (dict): A dictionary where each key maps to a list of values.

    Returns:
        list[dict]: A list of dictionaries, each representing a row.
    """
    keys = column_dict.keys()
    values = zip(*column_dict.values())
    return [dict(zip(keys, row)) for row in values]


def create_spark_df(spark, df: Union[pl.DataFrame, DataFrame]) -> DataFrame:
    df = df.to_native_df() if isinstance(df, DataFrame) else df
    return spark.createDataFrame(pd.DataFrame(df.to_arrow().to_pandas()))


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
    assert (
        json_df1 == json_df2
    ), f"""
{json_df1}
vs
{json_df2}"""

    return True
