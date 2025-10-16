import json

import pytest
import pyspark.sql.types as pst
from sparkleframe.polarsdf import types as sft
import polars as pl

from pyspark.sql.functions import col as spark_col
from sparkleframe.polarsdf.functions import col as sparkle_col


class TestTypes:

    @pytest.mark.parametrize(
        "spark_type, sf_type",
        [
            (pst.StringType(), sft.StringType()),
            (pst.IntegerType(), sft.IntegerType()),
            (pst.LongType(), sft.LongType()),
            (pst.FloatType(), sft.FloatType()),
            (pst.DoubleType(), sft.DoubleType()),
            (pst.BooleanType(), sft.BooleanType()),
            (pst.DateType(), sft.DateType()),
            (pst.TimestampType(), sft.TimestampType()),
            (pst.ByteType(), sft.ByteType()),
            (pst.ShortType(), sft.ShortType()),
            (pst.BinaryType(), sft.BinaryType()),
        ],
    )
    def test_simple_type_equivalence(self, spark_type, sf_type):
        assert spark_type.typeName() == sf_type.typeName()
        assert spark_type.simpleString() == sf_type.simpleString()
        assert spark_type.jsonValue() == sf_type.jsonValue()

    @pytest.mark.parametrize("precision, scale", [(10, 2), (5, 0), (20, 10)])
    def test_decimal_type_equivalence(self, precision, scale):
        spark_type = pst.DecimalType(precision, scale)
        sf_type = sft.DecimalType(precision, scale)

        assert spark_type.typeName() == sf_type.typeName()
        assert spark_type.simpleString() == sf_type.simpleString()
        assert spark_type.jsonValue() == sf_type.jsonValue()
        assert sf_type.precision == spark_type.precision
        assert sf_type.scale == spark_type.scale

    def test_struct_type_equivalence(self):
        sf_struct = sft.StructType(
            [
                sft.StructField("id", sft.IntegerType(), True),
                sft.StructField("name", sft.StringType(), False),
            ]
        )

        ps_struct = pst.StructType(
            [
                pst.StructField("id", pst.IntegerType(), True),
                pst.StructField("name", pst.StringType(), False),
            ]
        )

        assert sf_struct.typeName() == ps_struct.typeName()
        assert isinstance(sf_struct.fields[0].dataType, sft.IntegerType)
        assert sf_struct.fields[0].name == ps_struct.fields[0].name
        assert sf_struct.fields[1].nullable == ps_struct.fields[1].nullable

    def test_struct_field_methods(self):
        sf = sft.StructField("name", sft.StringType(), False, {"meta": 1})
        psf = pst.StructField("name", pst.StringType(), False, {"meta": 1})

        assert sf.name == psf.name
        assert sf.nullable == psf.nullable
        assert sf.dataType.typeName() == psf.dataType.typeName()
        assert sf.simpleString() == psf.simpleString()
        assert sf.__repr__() == psf.__repr__()
        assert sf.jsonValue() == psf.jsonValue()

    def test_struct_type_methods(self):
        sf1 = sft.StructField("id", sft.IntegerType(), True)
        sf2 = sft.StructField("name", sft.StringType(), False)
        sftype = sft.StructType([sf1, sf2])

        psf1 = pst.StructField("id", pst.IntegerType(), True)
        psf2 = pst.StructField("name", pst.StringType(), False)
        pstype = pst.StructType([psf1, psf2])

        # simpleString
        assert sftype.simpleString() == pstype.simpleString()

        # repr
        assert repr(sftype) == repr(pstype)

        # jsonValue
        assert sftype.jsonValue() == pstype.jsonValue()

        # __len__
        assert len(sftype) == 2

        # __getitem__ by index
        assert isinstance(sftype[0], sft.StructField)
        assert sftype[0].name == pstype[0].name

        # __getitem__ by name
        assert sftype["name"].dataType.typeName() == pstype["name"].dataType.typeName()

        # __getitem__ slice
        sliced = sftype[0:1]
        assert isinstance(sliced, sft.StructType)
        assert len(sliced) == 1
        assert sliced[0].name == "id"

        sliced = pstype[0:1]
        assert isinstance(sliced, pst.StructType)
        assert len(sliced) == 1
        assert sliced[0].name == "id"

        # __iter__
        assert [f.name for f in sftype] == [f.name for f in pstype]

        # fieldNames
        assert sftype.fieldNames() == pstype.fieldNames()

    def test_struct_type_getitem_errors(self):
        sftype = sft.StructType([sft.StructField("a", sft.StringType())])

        with pytest.raises(KeyError):
            _ = sftype["missing"]

        with pytest.raises(IndexError):
            _ = sftype[99]

        with pytest.raises(ValueError):
            _ = sftype[{"bad": "key"}]

    @pytest.mark.parametrize(
        "key_sf,value_sf,key_ps,value_ps,value_contains_null",
        [
            (sft.StringType(), sft.IntegerType(), pst.StringType(), pst.IntegerType(), True),
            (sft.StringType(), sft.IntegerType(), pst.StringType(), pst.IntegerType(), False),
            (sft.StringType(), sft.StringType(), pst.StringType(), pst.StringType(), True),
        ],
    )
    def test_maptype_equivalence(self, key_sf, value_sf, key_ps, value_ps, value_contains_null):
        sf_map = sft.MapType(key_sf, value_sf, valueContainsNull=value_contains_null)
        ps_map = pst.MapType(key_ps, value_ps, valueContainsNull=value_contains_null)

        # API parity with pyspark
        assert sf_map.typeName() == ps_map.typeName() == "map"
        assert sf_map.simpleString() == ps_map.simpleString()

        map_obj = {
            "type": "map",
            "keyType": key_ps().jsonValue() if callable(getattr(key_ps, "__call__", None)) else key_ps.jsonValue(),
            "valueType": (
                value_ps().jsonValue() if callable(getattr(value_ps, "__call__", None)) else value_ps.jsonValue()
            ),
            "valueContainsNull": value_contains_null,
        }
        assert sf_map.jsonValue() == ps_map.jsonValue() == map_obj

        # Native polars dtype shape: List(Struct([Field("key", ...), Field("value", ...)]))
        native = sf_map.to_native()
        assert isinstance(native, pl.List)
        assert isinstance(native.inner, pl.Struct)
        fields = native.inner.fields
        assert fields[0].name == "key"
        assert fields[1].name == "value"

        # key/value inner dtypes match
        assert fields[0].dtype == key_sf.to_native()
        assert fields[1].dtype == value_sf.to_native()

    @pytest.mark.parametrize(
        "rows, spark_schema, sparkle_schema, pointer_select",
        [
            (
                [
                    ({"id": 1, "m": 1},),
                    ({"id": 1, "m": 1},),
                ],
                pst.StructType([pst.StructField("col", pst.MapType(pst.StringType(), pst.IntegerType()))]),
                sft.StructType([sft.StructField("col", sft.MapType(sft.StringType(), sft.IntegerType()))]),
                "col.id",
            ),
            (
                [
                    ({"id": {"id2": 1, "m2": 1}},),
                ],
                pst.StructType(
                    [
                        pst.StructField(
                            "col", pst.MapType(pst.StringType(), pst.MapType(pst.StringType(), pst.IntegerType()))
                        )
                    ]
                ),
                sft.StructType(
                    [
                        sft.StructField(
                            "col", sft.MapType(sft.StringType(), sft.MapType(sft.StringType(), sft.IntegerType()))
                        )
                    ]
                ),
                "col.id.id2",
            ),
        ],
    )
    def test_roundtrip_polars_to_spark_map_column(
        self, spark, sparkle, rows, spark_schema, sparkle_schema, pointer_select
    ):
        """
        Build a Polars DF using the MapType native representation (List[Struct{key,value}]),
        convert to pandas -> Spark DF with a MapType schema, and compare against an expected Spark DF.
        """

        df_spark = spark.createDataFrame(rows, schema=spark_schema)
        df_spark.select(pointer_select).show(truncate=False)

        df_pl = sparkle.createDataFrame(rows, schema=sparkle_schema)

        json_df_spark = json.dumps(df_spark.toPandas().to_dict(orient="records"), sort_keys=True)
        json_df_pl = json.dumps(df_pl.toPandas().to_dict(orient="records"), sort_keys=True)
        assert json_df_spark == json_df_pl

        json_df_spark = json.dumps(
            df_spark.select(pointer_select).toPandas().to_dict(orient="records"), sort_keys=True
        )
        json_df_pl = json.dumps(df_pl.select(pointer_select).toPandas().to_dict(orient="records"), sort_keys=True)
        assert json_df_spark == json_df_pl

    @pytest.mark.parametrize(
        "rows, spark_schema, sparkle_schema, pointer_select",
        [
            # 1) empty map
            (
                [
                    ({},),
                ],
                # Spark: single map column named 'value' -> {} ; but we use StructType below for parity with "col"
                pst.StructType([pst.StructField("col", pst.MapType(pst.StringType(), pst.IntegerType()))]),
                sft.StructType([sft.StructField("col", sft.MapType(sft.StringType(), sft.IntegerType()))]),
                None,
            ),
            # 2) native [{key,value}] input instead of dict
            (
                [
                    ({"key": 2, "value": 5},),
                    ({"key": 3, "value": 7},),
                ],
                pst.StructType([pst.StructField("col", pst.MapType(pst.StringType(), pst.IntegerType()))]),
                sft.StructType([sft.StructField("col", sft.MapType(sft.StringType(), sft.IntegerType()))]),
                "col.key",
            ),
        ],
    )
    def test_maptype_misc_inputs(self, spark, sparkle, rows, spark_schema, sparkle_schema, pointer_select):
        df_spark = spark.createDataFrame(rows, schema=spark_schema)
        df_pl = sparkle.createDataFrame(rows, schema=sparkle_schema)

        # Data equality
        assert json.dumps(df_spark.toPandas().to_dict(orient="records"), sort_keys=True) == json.dumps(
            df_pl.toPandas().to_dict(orient="records"), sort_keys=True
        )

        # Optional pointer select (if provided)
        if pointer_select:
            # selected column names should be the LAST segment (aliasing behavior)
            pl_sel = df_pl.select(pointer_select)

            assert json.dumps(df_spark.toPandas().to_dict(orient="records"), sort_keys=True) == json.dumps(
                df_pl.toPandas().to_dict(orient="records"), sort_keys=True
            )
            # alias = last segment
            last = pointer_select.split(".")[-1]
            assert list(pl_sel.toPandas().columns) == [last]

    @pytest.mark.parametrize(
        "rows, spark_schema, sparkle_schema",
        [
            (
                [
                    ({"key": "2", "value": 5},),
                ],
                pst.StructType([pst.StructField("col", pst.MapType(pst.StringType(), pst.IntegerType()))]),
                sft.StructType([sft.StructField("col", sft.MapType(sft.StringType(), sft.IntegerType()))]),
            ),
        ],
    )
    def test_maptype_mix_types_fails_spark_passes_polars(self, spark, sparkle, rows, spark_schema, sparkle_schema):
        with pytest.raises(TypeError):
            spark.createDataFrame(rows, schema=spark_schema)

        df_pl = sparkle.createDataFrame(rows, schema=sparkle_schema)
        assert json.dumps(df_pl.toPandas().to_dict(orient="records"), sort_keys=True) == json.dumps(
            [{"col": {"key": 2, "value": 5}}], sort_keys=True
        )

    def test_maptype_column_manipulation(self, spark, sparkle):
        data = [({"key": {"key2": "a"}},)]
        schema_spark = pst.StructType(
            [pst.StructField("col", pst.MapType(pst.StringType(), pst.MapType(pst.StringType(), pst.StringType())))]
        )
        schema_sparkle = sft.StructType(
            [sft.StructField("col", sft.MapType(sft.StringType(), sft.MapType(sft.StringType(), sft.StringType())))]
        )

        df_spark = spark.createDataFrame(data, schema=schema_spark)
        df_sparkle = sparkle.createDataFrame(data, schema=schema_sparkle)

        json_spark = json.dumps(
            df_spark.withColumn("test", spark_col("col.key.key2")).select("test").toPandas().to_dict(orient="records"),
            sort_keys=True,
        )
        json_sparkle = json.dumps(
            df_sparkle.withColumn("test", sparkle_col("col.key.key2"))
            .select("test")
            .toPandas()
            .to_dict(orient="records"),
            sort_keys=True,
        )

        assert json_spark == json_sparkle

    def test_maptype_column_getitem(self, spark, sparkle):
        data = [({"key": {"key2": "a"}},)]
        schema_spark = pst.StructType(
            [pst.StructField("col", pst.MapType(pst.StringType(), pst.MapType(pst.StringType(), pst.StringType())))]
        )
        df_spark = spark.createDataFrame(data, schema=schema_spark)

        schema_sparkle = sft.StructType(
            [sft.StructField("col", sft.MapType(sft.StringType(), sft.MapType(sft.StringType(), sft.StringType())))]
        )
        df_sparkle = sparkle.createDataFrame(data, schema=schema_sparkle)

        json_spark = json.dumps(
            df_spark.withColumn("test", spark_col("col").getItem("key").getItem("key2"))
            .select("test")
            .toPandas()
            .to_dict(orient="records"),
            sort_keys=True,
        )
        json_sparkle = json.dumps(
            df_sparkle.withColumn("test", sparkle_col("col").getItem("key").getItem("key2"))
            .select("test")
            .toPandas()
            .to_dict(orient="records"),
            sort_keys=True,
        )

        assert json_spark == json_sparkle
