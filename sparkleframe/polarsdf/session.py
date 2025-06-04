from __future__ import annotations

from typing import Union, Iterable, Any, Optional

import pandas as pd
import polars as pl

from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.types import StructType, DataType


class SparkSession:
    def __init__(self):
        self.appName_str = ""
        self.master_str = ""

    def createDataFrame(
        self,
        data: Union[Iterable[Any], pd.DataFrame, pl.DataFrame, list],
        schema: Optional[Union[DataType, StructType, str]] = None,
    ) -> DataFrame:
        if isinstance(data, pd.DataFrame) or isinstance(data, list):
            return DataFrame(pl.DataFrame(data), schema=schema)
        elif isinstance(data, pl.DataFrame):
            return DataFrame(data, schema=schema)
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
