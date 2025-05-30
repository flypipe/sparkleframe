from typing import Union

import polars as pl

from sparkleframe.base.dataframe import DataFrame
from sparkleframe.polarsdf import Column


class GroupedData:
    def __init__(self, df: DataFrame, group_cols: list[Union[str, Column]]):
        self.spark_df = df
        self.df = self.spark_df.df
        self.group_cols = [col.to_native() if isinstance(col, Column) else pl.col(col) for col in group_cols]

    def agg(self, *exprs: Union[str, Column]) -> DataFrame:
        pl_exprs = [e.to_native() if isinstance(e, Column) else e for e in exprs]
        grouped = self.df.group_by(*self.group_cols).agg(pl_exprs)
        return type(self.spark_df)(grouped)

    def count(self) -> DataFrame:
        grouped = self.df.group_by(*self.group_cols).count()
        return type(self.spark_df)(grouped)

    def sum(self) -> DataFrame:
        grouped = self.df.group_by(*self.group_cols).sum()
        return type(self.spark_df)(grouped)

    def mean(self) -> DataFrame:
        grouped = self.df.group_by(*self.group_cols).mean()
        return type(self.spark_df)(grouped)

    def max(self) -> DataFrame:
        grouped = self.df.group_by(*self.group_cols).max()
        return type(self.spark_df)(grouped)

    def min(self) -> DataFrame:
        grouped = self.df.group_by(*self.group_cols).min()
        return type(self.spark_df)(grouped)
