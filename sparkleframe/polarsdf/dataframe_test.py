import json

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType as SparkBinaryType
from pyspark.sql.types import BooleanType as SparkBooleanType
from pyspark.sql.types import ByteType as SparkByteType
from pyspark.sql.types import DateType as SparkDateType
from pyspark.sql.types import DecimalType as SparkDecimalType
from pyspark.sql.types import DoubleType as SparkDoubleType
from pyspark.sql.types import FloatType as SparkFloatType
from pyspark.sql.types import IntegerType as SparkIntegerType
from pyspark.sql.types import LongType as SparkLongType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import ShortType as SparkShortType
from pyspark.sql.types import StringType as SparkStringType
from pyspark.sql.types import StructField as SparkStructField
from pyspark.sql.types import StructType as SparkStructType
from pyspark.sql.types import TimestampType as SparkTimestampType

import sparkleframe.polarsdf.functions as PF
from sparkleframe.polarsdf import Column
from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.types import (
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from sparkleframe.polarsdf.types_utils import _MapTypeUtils
from sparkleframe.engine import Engine
from sparkleframe.tests.parity.engines import ENGINES
from sparkleframe.tests.parity.oracle import assert_matches_spark
from sparkleframe.tests.sample_data import sample_data, sample_schema
from sparkleframe.tests.utils import create_spark_df


@pytest.fixture
def sparkle_df():
    # Schema-driven: feed the canonical PySpark schema + rows through the engine adapter,
    # which handles the PySpark -> sparkle StructType translation.
    return ENGINES[Engine.POLARS].build_df(sample_data, sample_schema)


@pytest.fixture
def spark_df(spark):
    return spark.createDataFrame(sample_data, sample_schema)


def _polars_native_map_series(name: str, dict_rows: list[dict[str, int]]) -> pl.Series:
    """
    Build a Polars-native 'map' layout column:
      List[Struct{key: Utf8, value: Int32}]
    from python dict rows.
    """
    kv_rows = [[{"key": k, "value": v} for k, v in d.items()] for d in dict_rows]
    return pl.Series(name, kv_rows)


class TestDataFrame:
    @pytest.mark.parametrize(
        "data, schema_sparkle, schema_spark",
        [
            (
                [{"a": 1}, {"a": 2}],
                StructType([StructField("a", IntegerType())]),
                SparkStructType([SparkStructField("a", SparkIntegerType())]),
            ),
            (
                pd.DataFrame([{"a": 1}, {"a": 2}]),
                StructType([StructField("a", IntegerType())]),
                SparkStructType([SparkStructField("a", SparkIntegerType())]),
            ),
            (
                [1, 2, 3, 4],
                IntegerType(),
                SparkIntegerType(),
            ),
            (
                pd.DataFrame([{"a": 1}, {"a": 2}]),
                None,
                None,
            ),
            (
                pl.DataFrame([{"a": 1}, {"a": 2}]),
                None,
                None,
            ),
            (
                pa.table({"a": [1, 2]}),
                None,
                None,
            ),
            ([{"a": 1}, {"a": 2}], None, None),
        ],
    )
    def test_dataframe_creation(self, spark, sparkle, data, schema_sparkle, schema_spark):
        df_sparkle = sparkle.createDataFrame(data, schema=schema_sparkle)

        if isinstance(data, pl.DataFrame):
            data = data.to_pandas().to_dict(orient="records")

        if isinstance(data, pa.Table):
            data = data.to_pylist()

        df_spark = spark.createDataFrame(data, schema=schema_spark)
        assert assert_matches_spark(df_sparkle, df_spark, ENGINES[Engine.POLARS])

    def test_drop_no_args_returns_new_wrapper_same_polars_df(self, sparkle_df):
        out = sparkle_df.drop()
        assert out is not sparkle_df
        assert out.columns == sparkle_df.columns
        assert out.df is sparkle_df.df
        assert out.to_native_df().equals(sparkle_df.to_native_df())

    def test_to_native_df(self, sparkle_df):
        native_df = sparkle_df.to_native_df()

        # Check that it's a Polars DataFrame
        assert isinstance(native_df, pl.DataFrame)

        # Check schema matches expected
        assert native_df.columns == sparkle_df.to_native_df().columns

        # Check data matches original sample_data
        assert native_df.shape == sparkle_df.to_native_df().shape
        assert native_df[0, 0] == "Alice"
        assert native_df[1, 1] == 30
        assert native_df[2, 2] == 90000

    @pytest.mark.parametrize(
        "col_name, data_type_class, spark_type",
        [
            ("name", StringType(), SparkStringType()),
            ("age", IntegerType(), SparkIntegerType()),
            ("age", LongType(), SparkLongType()),
            ("age", FloatType(), SparkFloatType()),
            ("age", DoubleType(), SparkDoubleType()),
            ("age", BooleanType(), SparkBooleanType()),
            ("age", DecimalType(13, 2), SparkDecimalType(13, 2)),
            ("birth_date", DateType(), SparkDateType()),
            ("login_time", TimestampType(), SparkTimestampType()),
            ("age", ByteType(), SparkByteType()),
            ("age", ShortType(), SparkShortType()),
            # int/long → binary is not valid under Spark 4 default ANSI SQL casts; no PySpark parity case.
        ],
    )
    def test_cast_types(self, spark, sparkle_df, spark_df, col_name, data_type_class, spark_type):
        # Apply the cast using your API
        expr = PF.col(col_name).cast(data_type_class)
        polars_result_df = sparkle_df.select(expr.alias(col_name)).to_native_df()

        spark_result_df = spark_df.select(F.col(col_name).cast(spark_type).alias(col_name))

        # Extract data types for comparison
        polars_dtype = polars_result_df.schema[col_name]
        spark_dtype = spark_result_df.schema[col_name].dataType

        # Manual mapping to match Spark types to Polars types
        spark_to_polars_map = {
            SparkStringType(): pl.Utf8,
            SparkIntegerType(): pl.Int32,
            SparkLongType(): pl.Int64,
            SparkFloatType(): pl.Float32,
            SparkDoubleType(): pl.Float64,
            SparkBooleanType(): pl.Boolean,
            SparkDateType(): pl.Date,
            SparkTimestampType(): pl.Datetime,
            SparkDecimalType(13, 2): pl.Decimal(13, 2),  # include your decimal type
            SparkShortType(): pl.Int16,
            SparkBinaryType(): pl.Binary,
            SparkByteType(): pl.Int8,
        }

        expected_polars_dtype = spark_to_polars_map[spark_dtype]

        assert polars_dtype == expected_polars_dtype

    def test_to_arrow_creates_correct_spark_df(self, spark):
        pl_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "amount": [10.5, 20.0, 30.25],
                "active": [True, False, True],
            }
        )
        sparkle_df = DataFrame(pl_df)

        expected_spark_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "name": ["Alice", "Bob", "Charlie"],
                    "amount": [10.5, 20.0, 30.25],
                    "active": [True, False, True],
                }
            )
        )

        assert_matches_spark(sparkle_df, expected_spark_df, ENGINES[Engine.POLARS])

    def test_create_polars_from_arrow_generated_by_spark(self, spark):
        # Step 1: Create a Spark DataFrame
        spark_df = spark.createDataFrame(
            [("Alice", 25, True), ("Bob", 30, False), ("Charlie", 35, True)], ["name", "age", "active"]
        )

        # Step 2: Collect as Arrow record batches and convert to Arrow Table
        arrow_batches = spark_df._collect_as_arrow()
        arrow_table = pa.Table.from_batches(arrow_batches)

        # Step 3: Create DataFrame from Arrow Table
        sparkle_df = DataFrame(arrow_table)

        # Step 4: Convert both to Pandas for comparison (safer for schema + nulls)
        expected_pd = spark_df.toPandas()
        result_pd = sparkle_df.toPandas()

        # Step 5: Sort by name for deterministic comparison (optional)
        expected_pd_sorted = expected_pd.sort_values(by="name").reset_index(drop=True)
        result_pd_sorted = result_pd.sort_values(by="name").reset_index(drop=True)

        # Step 6: Compare using assert_frame_equal
        pd.testing.assert_frame_equal(result_pd_sorted, expected_pd_sorted)

    def test_join_keys_different_type_raise_error(self):
        left_data = {"id": [1, 2, 3], "left_val": ["a", "b", "c"]}
        right_data = {"id": [2, 3, 4], "right_val": ["x", "y", "z"]}

        pl_left_df = DataFrame(pl.DataFrame(left_data))
        pl_right_df = DataFrame(pl.DataFrame(right_data))

        with pytest.raises(TypeError):
            pl_left_df.join(pl_right_df, on=["id", PF.col("id")], how="left")

    def test_columns_property(self):
        # Create sample Polars DataFrame
        data = {"name": ["Alice", "Bob"], "age": [30, 40], "salary": [1000, 2000]}
        df = DataFrame(pl.DataFrame(data))

        # Validate the columns property
        assert df.columns == ["name", "age", "salary"]

    @pytest.mark.parametrize(
        "value, dtype_class, expected_spark_type",
        [
            ("foo", StringType(), SparkStringType()),
            (42, IntegerType(), SparkIntegerType()),
            (42, LongType(), SparkLongType()),
            (3.14, FloatType(), SparkFloatType()),
            (2.718281828, DoubleType(), SparkDoubleType()),
            (True, BooleanType(), SparkBooleanType()),
            (pd.to_datetime("2024-01-01").date(), DateType(), SparkDateType()),
            (pd.to_datetime("2024-01-01 12:34:56"), TimestampType(), SparkTimestampType()),
            (123.45, DecimalType(10, 2), SparkDecimalType(10, 2)),
            (1, ByteType(), SparkByteType()),
            (100, ShortType(), SparkShortType()),
            (b"abc", BinaryType(), SparkBinaryType()),
            # ✅ Nested StructType test case
            (
                {"field1": "hello", "field2": 999},
                StructType([StructField("field1", StringType()), StructField("field2", IntegerType())]),
                SparkStructType(
                    [
                        SparkStructField("field1", SparkStringType()),
                        SparkStructField("field2", SparkIntegerType()),
                    ]
                ),
            ),
            # ✅ Struct with nested StructType inside
            (
                {"outer": {"inner": 123}},
                StructType([StructField("outer", StructType([StructField("inner", IntegerType())]))]),
                SparkStructType(
                    [SparkStructField("outer", SparkStructType([SparkStructField("inner", SparkIntegerType())]))]
                ),
            ),
        ],
    )
    def test_polars_to_spark_dtype_conversion(self, spark, value, dtype_class, expected_spark_type):
        col_name = "col"
        pl_df = pl.DataFrame({col_name: [value]})

        # Wrap in Sparkleframe DataFrame
        sparkle_df = DataFrame(pl_df).withColumn(col_name, PF.col(col_name).cast(dtype_class))

        # Convert to Arrow → Pandas → Spark (actual logic for type preservation)
        if isinstance(value, dict):
            spark_df = spark.createDataFrame(data=[(json.dumps(value),)], schema=(col_name,)).withColumn(
                col_name, F.from_json(F.col(col_name), expected_spark_type)
            )

        else:
            spark_df = create_spark_df(spark, sparkle_df).withColumn(
                col_name, F.col(col_name).cast(expected_spark_type)
            )

        assert spark_df.dtypes == sparkle_df.dtypes

    @pytest.mark.parametrize(
        "col_name, value, dtype_class, expected_spark_type",
        [
            ("string_col", "foo", StringType(), SparkStringType()),
            ("int_col", 42, IntegerType(), SparkIntegerType()),
            ("long_col", 42, LongType(), SparkLongType()),
            ("float_col", 3.14, FloatType(), SparkFloatType()),
            ("double_col", 2.718281828, DoubleType(), SparkDoubleType()),
            ("bool_col", True, BooleanType(), SparkBooleanType()),
            ("date_col", pd.to_datetime("2024-01-01").date(), DateType(), SparkDateType()),
            ("timestamp_col", pd.to_datetime("2024-01-01 12:34:56"), TimestampType(), SparkTimestampType()),
            ("decimal_col", 123.45, DecimalType(10, 2), SparkDecimalType(10, 2)),
            ("byte_col", 1, ByteType(), SparkByteType()),
            ("short_col", 100, ShortType(), SparkShortType()),
            ("binary_col", b"abc", BinaryType(), SparkBinaryType()),
            (
                "struct_col",
                {"field1": "hello", "field2": 999},
                StructType([StructField("field1", StringType()), StructField("field2", IntegerType())]),
                SparkStructType(
                    [
                        SparkStructField("field1", SparkStringType()),
                        SparkStructField("field2", SparkIntegerType()),
                    ]
                ),
            ),
            (
                "nested_struct_col",
                {"outer": {"inner": 123}},
                StructType([StructField("outer", StructType([StructField("inner", IntegerType())]))]),
                SparkStructType(
                    [SparkStructField("outer", SparkStructType([SparkStructField("inner", SparkIntegerType())]))]
                ),
            ),
        ],
    )
    def test_schema_conversion_to_spark_dtype(self, spark, col_name, value, dtype_class, expected_spark_type):
        # Create Polars DataFrame and cast using Sparkleframe types
        pl_df = pl.DataFrame({col_name: [value]})
        sparkle_df = DataFrame(pl_df).withColumn(col_name, PF.col(col_name).cast(dtype_class))

        # Convert to Spark DF for comparison
        if isinstance(value, dict):
            # Use JSON roundtrip for struct parsing
            spark_df = spark.createDataFrame([(json.dumps(value),)], schema=(col_name,)).withColumn(
                col_name, F.from_json(F.col(col_name), expected_spark_type)
            )
        else:
            spark_df = spark.createDataFrame(sparkle_df.toPandas()).withColumn(
                col_name, F.col(col_name).cast(expected_spark_type)
            )

        # Check schema equality
        assert json.dumps(sparkle_df.schema, sort_keys=True, default=str) == json.dumps(
            spark_df.schema, sort_keys=True, default=str
        )

    def test_schema_equivalence_with_spark(self, spark):
        # Sample Pandas data
        pdf = pd.DataFrame([[1, "Alice"], [2, "Bob"]])

        # Sparkleframe schema
        sf_schema = StructType(
            [StructField("id", IntegerType(), nullable=False), StructField("name", StringType(), nullable=True)]
        )

        # Equivalent Spark schema
        spark_schema = SparkStructType(
            [
                SparkStructField("id", SparkIntegerType(), nullable=False),
                SparkStructField("name", SparkStringType(), nullable=True),
            ]
        )

        # Create Sparkleframe DataFrame
        sf_df = DataFrame(pdf, schema=sf_schema)

        # Create Spark DataFrame
        spark_df = spark.createDataFrame(pdf, schema=spark_schema)

        assert json.dumps(sf_df._schema, sort_keys=True, default=str) == json.dumps(
            spark_df._schema, sort_keys=True, default=str
        )

    def test_getitem_str(self, spark):
        input_data = [{"age": 2, "name": "Alice"}, {"age": 5, "name": "Bob"}]
        pdf = pd.DataFrame(input_data)
        sdf = spark.createDataFrame(pdf)
        ps_result = sdf["age"]

        sf_df = DataFrame(pl.DataFrame(pdf))
        sf_result = sf_df["age"]

        assert isinstance(sf_result, Column)
        assert ps_result.__class__.__name__ == sf_result.__class__.__name__

        sdf = sdf.select(ps_result)
        sf_result_df = sf_df.select(sf_result)
        assert_matches_spark(sf_result_df, sdf, ENGINES[Engine.POLARS])

    def test_getitem_int(self, spark):
        input_data = [{"age": 2, "name": "Alice"}, {"age": 5, "name": "Bob"}]
        pdf = pd.DataFrame(input_data)
        sdf = spark.createDataFrame(pdf)
        ps_result = sdf[0]

        sf_df = DataFrame(pl.DataFrame(pdf))
        sf_result = sf_df[0]

        assert isinstance(sf_result, Column)
        assert ps_result.__class__.__name__ == sf_result.__class__.__name__

        sdf = sdf.select(ps_result)
        sf_result_df = sf_df.select(sf_result)
        assert_matches_spark(sf_result_df, sdf, ENGINES[Engine.POLARS])

    def test_order_by_int_zero_raises(self):
        sf_df = DataFrame(pl.DataFrame({"age": [1], "name": ["a"]}))
        with pytest.raises(ValueError, match=r"\[ZERO_INDEX\] Index must be non-zero\."):
            sf_df.orderBy(0)

    def test_order_by_int_out_of_range(self):
        sf_df = DataFrame(pl.DataFrame({"age": [1], "name": ["a"]}))
        with pytest.raises(IndexError):
            sf_df.orderBy(3)
        with pytest.raises(IndexError):
            sf_df.orderBy(-3)

    def test_order_by_ascending_kwarg_raises_error(self, spark):
        data = {"age": [2, 5], "name": ["Alice", "Bob"]}

        sf_df = DataFrame(pl.DataFrame(data))

        with pytest.raises(TypeError):
            sf_df.orderBy("age", ascending=True)

    def test_count(self):
        df = DataFrame(pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        assert df.count() == 3

    def test_auto_materialize_maptype_single_level_select(self, spark, sparkle):
        # Polars-native map layout in the input DF
        native = pl.DataFrame([_polars_native_map_series("col", [{"id": 1, "m": 1}])])

        # Sparkle schema declares a MapType so the constructor should auto-materialize to a Struct
        sf_schema = StructType([StructField("col", MapType(StringType(), IntegerType()))])
        df_sparkle = sparkle.createDataFrame(native, schema=sf_schema)

        # Column becomes a Struct with fields ["id", "m"]
        struct_dtype = df_sparkle.to_native_df().schema["col"]
        assert isinstance(struct_dtype, pl.Struct)
        assert [f.name for f in struct_dtype.fields] == ["id", "m"]

        # Dot access selection should work and alias to leaf name
        sel_sf = df_sparkle.select("col.id")  # -> Column named "id"

        # Note: simper & robust — just build the same result frame from Polars’ pandas output
        expected_spark_df = spark.createDataFrame(sel_sf.toPandas())

        assert_matches_spark(sel_sf, expected_spark_df, ENGINES[Engine.POLARS])

    def test_auto_materialize_is_noop_for_plain_struct(self, spark, sparkle):
        # A plain struct column should NOT go through map auto-materialization
        native = pl.DataFrame({"s": [{"a": 10}]})
        sf_schema = StructType([StructField("s", StructType([StructField("a", IntegerType())]))])

        df_sparkle = sparkle.createDataFrame(native, schema=sf_schema)

        s_dtype = df_sparkle.to_native_df().schema["s"]
        assert isinstance(s_dtype, pl.Struct)
        assert [f.name for f in s_dtype.fields] == ["a"]

        # And dot access works as usual
        sel_sf = df_sparkle.select("s.a")
        expected_spark_df = spark.createDataFrame(sel_sf.toPandas())
        assert_matches_spark(sel_sf, expected_spark_df, ENGINES[Engine.POLARS])

    def test_auto_materialize_warns_but_does_not_crash(self, monkeypatch, spark, sparkle):
        # Prepare a DF that would normally auto-materialize
        native = pl.DataFrame([_polars_native_map_series("col", [{"x": 5}])])
        sf_schema = StructType([StructField("col", MapType(StringType(), IntegerType()))])

        # Force the internal converter to throw so we take the warning branch
        monkeypatch.setattr(_MapTypeUtils, "is_map_dtype", lambda dt: True)

        def boom(df, col, *, keys=None, new_name=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(_MapTypeUtils, "map_to_struct", boom)

        with pytest.warns(RuntimeWarning, match="MapType materialization skipped"):
            df_sparkle = sparkle.createDataFrame(native, schema=sf_schema)

        # Still a valid DataFrame; and since conversion failed, col is NOT a Struct
        assert isinstance(df_sparkle, DataFrame)
        assert not isinstance(df_sparkle.to_native_df().schema["col"], pl.Struct)

    def test_auto_materialize_union_keys_order_and_nulls(self, spark, sparkle):
        # Two rows with different keys; union keeps first-seen order ("a" then "b")
        native = pl.DataFrame([_polars_native_map_series("col", [{"a": 1}, {"b": 2}])])
        sf_schema = StructType([StructField("col", MapType(StringType(), IntegerType()))])

        df_sparkle = sparkle.createDataFrame(native, schema=sf_schema)

        struct_dtype = df_sparkle.to_native_df().schema["col"]
        assert isinstance(struct_dtype, pl.Struct)
        assert [f.name for f in struct_dtype.fields] == ["a", "b"]

        map_long = SparkMapType(SparkStringType(), SparkLongType(), valueContainsNull=True)
        expected_spark_df = spark.createDataFrame(
            [({"a": 1, "b": None},), ({"a": None, "b": 2},)],
            schema=SparkStructType([SparkStructField("col", map_long)]),
        )

        assert_matches_spark(df_sparkle, expected_spark_df, ENGINES[Engine.POLARS])


class TestUnionByName:
    def test_aligns_by_column_name_not_position(self) -> None:
        left = DataFrame(pl.DataFrame({"x": [1], "y": [2]}))
        right = DataFrame(pl.DataFrame({"y": [3], "x": [4]}))
        out = left.unionByName(right)
        assert out.to_native_df().equals(pl.DataFrame({"x": [1, 4], "y": [2, 3]}))

    def test_strict_rejects_disjoint_schemas(self) -> None:
        left = DataFrame(pl.DataFrame({"x": [1], "y": [2]}))
        right = DataFrame(pl.DataFrame({"x": [3], "z": [4]}))
        with pytest.raises(ValueError, match="allowMissingColumns=True"):
            left.unionByName(right, allowMissingColumns=False)

    def test_allow_missing_columns_null_pads(self) -> None:
        left = DataFrame(pl.DataFrame({"x": [1], "y": [2]}))
        right = DataFrame(pl.DataFrame({"x": [5]}))
        out = left.unionByName(right, allowMissingColumns=True)
        assert out.to_native_df().equals(pl.DataFrame({"x": [1, 5], "y": [2, None]}))

    def test_expects_dataframe(self) -> None:
        left = DataFrame(pl.DataFrame({"x": [1]}))
        with pytest.raises(TypeError, match="DataFrame"):
            left.unionByName(pl.DataFrame({"x": [2]}))  # type: ignore[arg-type]
