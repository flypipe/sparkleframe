from __future__ import annotations

import pandas as pd
import polars as pl
from typing import List, Union

from sparkleframe.base.dataframe import DataFrame as BaseDataFrame
from sparkleframe.polarsdf.column import Column
from pandas import DataFrame as PandasDataFrame
import pyarrow as pa

class DataFrame(BaseDataFrame):

    def __init__(self, df: Union[pl.DataFrame, pd.DataFrame, pa.Table]):
        if isinstance(df, pl.DataFrame):
            self.df = df
        elif isinstance(df, pd.DataFrame):
            self.df = pl.DataFrame(df)
        elif isinstance(df, pa.Table):
            self.df = pl.from_arrow(df)
        else:
            raise TypeError("DataFrame constructor accepts polars.DataFrame, pandas.DataFrame, or pyarrow.Table")

        super().__init__(self.df)

    def select(self, *cols: Union[str, Column]) -> 'DataFrame':
        """
        Mimics PySpark's select method using Polars.

        Args:
            *cols: Column names or Column wrapper objects.

        Returns:
            A new DataFrame with selected columns.
        """
        pl_expressions = [col.to_native() if isinstance(col, Column) else col for col in cols]
        selected_df = self.df.select(*pl_expressions)
        return DataFrame(selected_df)

    def withColumn(self, name: str, col: Column) -> DataFrame:
        """
        Mimics PySpark's withColumn method using Polars.

        Args:
            name: Name of the new or updated column.
            col: A Column object representing the expression for the new column.

        Returns:
            A new DataFrame with the added or updated column.
        """
        expr = col.to_native().alias(name)
        updated_df = self.df.with_columns(expr)
        return DataFrame(updated_df)

    def withColumnRenamed(self, existing: str, new: str) -> DataFrame:
        """
        Mimics PySpark's withColumnRenamed method using Polars.

        Args:
            existing: The current column name.
            new: The new name to apply.

        Returns:
            A new DataFrame with the renamed column.

        Raises:
            ValueError: If the existing column name is not in the DataFrame.
        """
        if existing not in self.df.columns:
            raise ValueError(f"Column '{existing}' does not exist in the DataFrame.")

        renamed_df = self.df.rename({existing: new})
        return DataFrame(renamed_df)

    def toPandas(self) -> PandasDataFrame:
        """Convert the underlying Polars DataFrame to a Pandas DataFrame."""
        return self.df.to_arrow().to_pandas()

    def to_arrow(self) -> pa.Table:
        """
        Convert the Polars DataFrame to an Apache Arrow Table.

        Returns:
            pyarrow.Table: Arrow representation of the DataFrame.
        """
        return self.df.to_arrow()

    def show(self, n: int = 20, truncate: int = 20, vertical: bool = False):
        """
        Mimics PySpark's DataFrame.show() using Polars' native rendering.

        Args:
            n (int): Number of rows to show.
            truncate (int): Ignored — Polars handles column truncation.
            vertical (bool): If True, displays rows in vertical layout.
        """
        if vertical:
            for i, row in enumerate(self.df.head(n).iter_rows(named=True)):
                print(f"-ROW {i}")
                for key, val in row.items():
                    print(f"{key}: {val}")
        else:
            print(self.df.head(n))