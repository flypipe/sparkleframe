import pandas as pd
import polars as pl
import pytest

from sparkleframe.polarsdf.dataframe import DataFrame

from sparkleframe.polarsdf.types import StructType, StructField, IntegerType, StringType, LongType, MapType


class TestSparkSession:

    def test_create_dataframe_from_polars(self, sparkle):
        pl_df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        result = sparkle.createDataFrame(pl_df)

        assert isinstance(result, DataFrame)

        result_native = result.to_native_df()
        assert result_native.shape == pl_df.shape
        assert result_native.columns == pl_df.columns
        assert result_native.to_dicts() == pl_df.to_dicts()

    def test_create_dataframe_from_pandas(self, sparkle):
        pd_df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        result = sparkle.createDataFrame(pd_df)

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
    def test_create_dataframe_various_inputs_and_schemas(self, sparkle, input_data, schema):

        expected = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        if schema:
            expected = expected.with_columns(pl.col("x").cast(pl.Int32))

        # Create SparkleFrame DataFrame
        result = sparkle.createDataFrame(input_data, schema=schema)

        assert isinstance(result, DataFrame)

        result_native = result.to_native_df()
        assert result_native.shape == expected.shape
        assert result_native.columns == expected.columns
        assert result_native.to_dicts() == expected.to_dicts()

        # Compare schema representation
        assert result.schema.json() == DataFrame(expected).schema.json()

    def test_create_dataframe_nested_struct_map_schema(self, sparkle):

        schema = StructType(
            [
                StructField("x", IntegerType()),
                StructField(
                    "y",
                    StructType(
                        [
                            StructField("x1", LongType()),
                            StructField("y1", MapType(StringType(), IntegerType())),
                        ]
                    ),
                ),
            ]
        )

        input_data = [{"x": 1, "y": {"x1": 1, "y1": {"x2": 2}}}]
        expected = pl.DataFrame(input_data)
        # Cast outer x and inner y.x1 to Int32
        expected = expected.with_columns(
            [
                # top-level x
                pl.col("x").cast(pl.Int32),
                # nested struct y
                pl.struct(
                    [
                        pl.col("y").struct.field("x1").cast(pl.Int64).alias("x1"),
                        pl.struct(
                            [pl.col("y").struct.field("y1").struct.field("x2").cast(pl.Int32).alias("x2")]
                        ).alias("y1"),
                    ]
                ).alias("y"),
            ]
        )

        # Create SparkleFrame DataFrame
        result = sparkle.createDataFrame(input_data, schema=schema)

        assert isinstance(result, DataFrame)

        result_native = result.to_native_df()

        assert result_native.shape == expected.shape
        assert result_native.columns == expected.columns
        assert result_native.to_dicts() == expected.to_dicts()

        # Compare schema representation
        assert result.schema.json() == DataFrame(expected).schema.json()
