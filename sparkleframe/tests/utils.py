from typing import Union

import pandas as pd
import polars as pl
from sparkleframe.polarsdf import DataFrame


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
