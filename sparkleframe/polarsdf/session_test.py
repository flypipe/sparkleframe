import pandas as pd
import polars as pl
import pytest

from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.session import SparkSession
from sparkleframe.polarsdf.types import StructType, StructField, IntegerType, StringType


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

    @pytest.mark.parametrize(
        "input_data",
        [
            [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, {"x": 3, "y": "c"}],  # Iterable[dict]
            pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}),  # pandas.DataFrame
            pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}),  # polars.DataFrame
        ],
    )
    @pytest.mark.parametrize(
        "schema",
        [
            None,
            StructType(
                [
                    StructField("x", IntegerType()),
                    StructField("y", StringType()),
                ]
            ),
        ],
    )
    def test_create_dataframe_various_inputs_and_schemas(self, input_data, schema):
        spark = SparkSession()

        expected = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

        # Create SparkleFrame DataFrame
        result = spark.createDataFrame(input_data, schema=schema)

        assert isinstance(result, DataFrame)

        result_native = result.to_native_df()
        assert result_native.shape == expected.shape
        assert result_native.columns == expected.columns
        assert result_native.to_dicts() == expected.to_dicts()

        # Compare schema representation
        assert result.schema.json() == DataFrame(expected).schema.json()
