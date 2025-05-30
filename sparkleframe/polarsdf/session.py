from __future__ import annotations

from typing import Union

import pandas as pd
import polars as pl

from sparkleframe.polarsdf.dataframe import DataFrame


class SparkSession:
    def __init__(self):
        self.appName_str = ""
        self.master_str = ""

    def createDataFrame(self, df: Union[pl.DataFrame, pd.DataFrame]) -> DataFrame:
        if isinstance(df, pd.DataFrame):
            return DataFrame(pl.DataFrame(df))
        elif isinstance(df, pl.DataFrame):
            return DataFrame(df)
        else:
            raise TypeError("createDataFrame only supports polars.DataFrame or pandas.DataFrame")

    class Builder:

        def appName(self, name):
            self.appName_str = name
            return self

        def master(self, master_str):
            self.master_str = master_str
            return self

        def getOrCreate(self) -> "SparkSession":
            return SparkSession()

        def config(self, key, value):
            return self

    builder = Builder()

    class SparkContext:

        def setLogLevel(self, level):
            pass

    sparkContext = SparkContext()
