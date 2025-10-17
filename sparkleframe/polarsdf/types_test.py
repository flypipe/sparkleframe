import json

import pytest
import pyspark.sql.types as pst
from sparkleframe.polarsdf import types as sft
import polars as pl

import pyspark.sql.functions as F
import sparkleframe.polarsdf.functions as SF
from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal, _get_json_from_dataframe


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
            df_spark.withColumn("test", F.col("col.key.key2")).select("test").toPandas().to_dict(orient="records"),
            sort_keys=True,
        )
        json_sparkle = json.dumps(
            df_sparkle.withColumn("test", SF.col("col.key.key2")).select("test").toPandas().to_dict(orient="records"),
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
            df_spark.withColumn("test", F.col("col").getItem("key").getItem("key2"))
            .select("test")
            .toPandas()
            .to_dict(orient="records"),
            sort_keys=True,
        )
        json_sparkle = json.dumps(
            df_sparkle.withColumn("test", SF.col("col").getItem("key").getItem("key2"))
            .select("test")
            .toPandas()
            .to_dict(orient="records"),
            sort_keys=True,
        )

        assert json_spark == json_sparkle

    @pytest.mark.parametrize(
        "elem_sf, elem_ps, contains_null",
        [
            (sft.IntegerType(), pst.IntegerType(), True),
            (sft.StringType(), pst.StringType(), False),
            (sft.DoubleType(), pst.DoubleType(), True),
        ],
    )
    def test_arraytype_equivalence_api(self, elem_sf, elem_ps, contains_null):
        sf_arr = sft.ArrayType(elem_sf, containsNull=contains_null)
        ps_arr = pst.ArrayType(elem_ps, containsNull=contains_null)

        # API parity with pyspark
        assert sf_arr.typeName() == ps_arr.typeName() == "array"
        assert sf_arr.simpleString() == ps_arr.simpleString()

        arr_json = {
            "type": "array",
            "elementType": elem_ps.jsonValue(),
            "containsNull": contains_null,
        }
        assert sf_arr.jsonValue() == ps_arr.jsonValue() == arr_json

        # __repr__ shape (not byte-for-byte identical classes, but structure should match)
        assert "ArrayType(" in repr(sf_arr)
        assert sf_arr.containsNull == contains_null

    @pytest.mark.parametrize(
        "elem_sf, expected_inner",
        [
            (sft.IntegerType(), pl.Int32),
            (sft.LongType(), pl.Int64),
            (sft.StringType(), pl.Utf8),
        ],
    )
    def test_arraytype_to_native_dtype(self, elem_sf, expected_inner):
        native = sft.ArrayType(elem_sf).to_native()
        assert isinstance(native, pl.List)
        assert native.inner == expected_inner

    @pytest.mark.parametrize(
        "rows, spark_schema, sparkle_schema",
        [
            # 1) array<int> with containsNull=False
            (
                [
                    ([1, 2, 3],),
                    ([4, 5, 6],),
                ],
                pst.StructType([pst.StructField("col", pst.ArrayType(pst.IntegerType(), containsNull=False))]),
                sft.StructType([sft.StructField("col", sft.ArrayType(sft.IntegerType(), containsNull=False))]),
            ),
            # 2) array<int> with null elements allowed
            (
                [
                    ([1, 2, 3],),
                    ([4, None, 6],),
                ],
                pst.StructType([pst.StructField("col", pst.ArrayType(pst.IntegerType(), containsNull=True))]),
                sft.StructType([sft.StructField("col", sft.ArrayType(sft.IntegerType(), containsNull=True))]),
            ),
            # 3) array<array<int>>
            (
                [
                    ([[1, 2], [3]],),
                    ([[4], []],),
                ],
                pst.StructType(
                    [
                        pst.StructField(
                            "col",
                            pst.ArrayType(pst.ArrayType(pst.IntegerType(), containsNull=True), containsNull=True),
                        )
                    ]
                ),
                sft.StructType(
                    [
                        sft.StructField(
                            "col",
                            sft.ArrayType(sft.ArrayType(sft.IntegerType(), containsNull=True), containsNull=True),
                        )
                    ]
                ),
            ),
            # 4) array<struct<id:int,name:string>>
            (
                [
                    ([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],),
                    ([{"id": 3, "name": None}],),
                ],
                pst.StructType(
                    [
                        pst.StructField(
                            "col",
                            pst.ArrayType(
                                pst.StructType(
                                    [
                                        pst.StructField("id", pst.IntegerType()),
                                        pst.StructField("name", pst.StringType()),
                                    ]
                                ),
                                containsNull=True,
                            ),
                        )
                    ]
                ),
                sft.StructType(
                    [
                        sft.StructField(
                            "col",
                            sft.ArrayType(
                                sft.StructType(
                                    [
                                        sft.StructField("id", sft.IntegerType()),
                                        sft.StructField("name", sft.StringType()),
                                    ]
                                ),
                                containsNull=True,
                            ),
                        )
                    ]
                ),
            ),
            # 5) array<map<string,int>>
            (
                [
                    ([{"a": 1, "b": 2}],),
                ],
                pst.StructType(
                    [pst.StructField("col", pst.ArrayType(pst.MapType(pst.StringType(), pst.IntegerType()), True))]
                ),
                sft.StructType(
                    [sft.StructField("col", sft.ArrayType(sft.MapType(sft.StringType(), sft.IntegerType()), True))]
                ),
            ),
            (
                [({"a": 1, "b": 2},)],
                pst.StructType([pst.StructField("col", pst.MapType(pst.StringType(), pst.IntegerType(), True))]),
                sft.StructType([sft.StructField("col", sft.MapType(sft.StringType(), sft.IntegerType(), True))]),
            ),
        ],
    )
    def test_roundtrip_polars_to_spark_array_column(self, spark, sparkle, rows, spark_schema, sparkle_schema):
        """
        Build both Spark and Sparkle DataFrames with ArrayType schemas and ensure
        data round-trips to identical pandas JSON for easy deep-equality.
        """

        # Spark DF from Python rows
        df_spark = spark.createDataFrame(rows, schema=spark_schema)

        # Sparkle DF (Polars backend) with equivalent schema
        df_pl = sparkle.createDataFrame(rows, schema=sparkle_schema)

        assert assert_sparkle_spark_frame_are_equal(df_spark, df_pl)

    def test_getItem_mixed_array_map_struct(self, spark, sparkle):
        # ---------- Data ----------
        # 3 columns:
        #   m:       map<string,int>                    -> {"a":1,"b":2}
        #   arr_m:   array<map<string,int>>             -> [{"a":1,"b":2}, {"a":3}]
        #   st:      struct<inner_m:map<string,int>,    -> {"inner_m":{"k":5}, "arr":[9,8]}
        #                   arr:array<int>>
        rows = [
            (
                {"a": 1, "b": 2},
                [{"a": 1, "b": 2}, {"a": 3}],
                {"inner_m": {"k": 5}, "arr": [9, 8]},
            )
        ]

        # ---------- Schemas ----------
        spark_schema = pst.StructType(
            [
                pst.StructField("m", pst.MapType(pst.StringType(), pst.IntegerType()), True),
                pst.StructField("arr_m", pst.ArrayType(pst.MapType(pst.StringType(), pst.IntegerType()), True), True),
                pst.StructField(
                    "st",
                    pst.StructType(
                        [
                            pst.StructField("inner_m", pst.MapType(pst.StringType(), pst.IntegerType()), True),
                            pst.StructField("arr", pst.ArrayType(pst.IntegerType(), True), True),
                        ]
                    ),
                    True,
                ),
            ]
        )

        sparkle_schema = sft.StructType(
            [
                sft.StructField("m", sft.MapType(sft.StringType(), sft.IntegerType()), True),
                sft.StructField("arr_m", sft.ArrayType(sft.MapType(sft.StringType(), sft.IntegerType()), True), True),
                sft.StructField(
                    "st",
                    sft.StructType(
                        [
                            sft.StructField("inner_m", sft.MapType(sft.StringType(), sft.IntegerType()), True),
                            sft.StructField("arr", sft.ArrayType(sft.IntegerType(), True), True),
                        ]
                    ),
                    True,
                ),
            ]
        )

        # ---------- DataFrames ----------
        df_spark = spark.createDataFrame(rows, schema=spark_schema)
        df_sparkle = sparkle.createDataFrame(rows, schema=sparkle_schema)

        # ---------- Selections using getItem ----------
        # What we check:
        #   m["a"]                        -> 1
        #   arr_m[0]["b"]                 -> 2
        #   arr_m[1]["a"]                 -> 3
        #   arr_m[1]["b"]                 -> NULL (missing key)
        #   st["inner_m"]["k"]            -> 5
        #   st["arr"][1]                  -> 8
        sel_spark = df_spark.select(
            F.col("m").getItem("a").alias("m_a"),
            F.col("arr_m").getItem(0).getItem("b").alias("arrm0_b"),
            F.col("arr_m").getItem(1).getItem("a").alias("arrm1_a"),
            F.col("arr_m").getItem(1).getItem("b").alias("arrm1_b_missing"),
            F.col("st").getItem("inner_m").getItem("k").alias("st_k"),
            F.col("st").getItem("arr").getItem(1).alias("st_arr1"),
        )

        sel_sparkle = df_sparkle.select(
            SF.col("m").getItem("a").alias("m_a"),
            SF.col("arr_m").getItem(0).getItem("b").alias("arrm0_b"),
            SF.col("arr_m").getItem(1).getItem("a").alias("arrm1_a"),
            SF.col("arr_m").getItem(1).getItem("b").alias("arrm1_b_missing"),
            SF.col("st").getItem("inner_m").getItem("k").alias("st_k"),
            SF.col("st").getItem("arr").getItem(1).alias("st_arr1"),
        )
        sel_spark.show(truncate=False)
        sel_sparkle.show(truncate=False)

        def _to_sorted_json(df):
            """Utility: collect a single-row DF to stable JSON for easy equality asserts."""
            pdf = df.toPandas()
            return json.dumps(pdf.to_dict(orient="records"), sort_keys=True)

        # ---------- Compare ----------
        got_spark = _get_json_from_dataframe(sel_spark)
        got_sparkle = _get_json_from_dataframe(sel_sparkle)

        # (Optional) quick visibility during dev:
        print(got_spark)
        print(got_sparkle)

        assert got_spark == got_sparkle
