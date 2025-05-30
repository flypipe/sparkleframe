from typing import Any

from pandas import DataFrame as PandasDataFrame


class DataFrame:
    def __init__(self, df: Any):
        self._alias = None
        self.df = df

    def pandas_api(self) -> PandasDataFrame:
        """Mimics Pandas On Spark dataframe"""
        return self.toPandas()

    def to_native_df(self) -> "DataFrame":
        """Return the underlying Polars DataFrame."""
        return self.df

    def to_spark(self) -> "DataFrame":
        return self
