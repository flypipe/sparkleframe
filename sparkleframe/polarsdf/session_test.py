import pandas as pd
import polars as pl
import pytest

from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.session import SparkSession


class TestSparkSession:

    @pytest.fixture
    def spark(self):
        return SparkSession()

    def test_create_dataframe_from_polars(self, spark):
        pl_df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        result = spark.createDataFrame(pl_df)

        assert isinstance(result, DataFrame)

        result_native = result.to_native_df()
        assert result_native.shape == pl_df.shape
        assert result_native.columns == pl_df.columns
        assert result_native.to_dicts() == pl_df.to_dicts()

    def test_create_dataframe_from_pandas(self, spark):
        pd_df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        result = spark.createDataFrame(pd_df)

        assert isinstance(result, DataFrame)

        expected_pl = pl.DataFrame(pd_df)
        result_native = result.to_native_df()

        assert result_native.shape == expected_pl.shape
        assert result_native.columns == expected_pl.columns
        assert result_native.to_dicts() == expected_pl.to_dicts()
