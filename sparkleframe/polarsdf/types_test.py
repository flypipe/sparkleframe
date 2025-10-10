import pytest
import pyspark.sql.types as pst
from sparkleframe.polarsdf import types as sft
import polars as pl
from pyspark.sql import Row


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
            "valueType": value_ps().jsonValue() if callable(getattr(value_ps, "__call__", None)) else value_ps.jsonValue(),
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
        "rows",
        [
            # integer values
            [
                {"id": 1, "m": [{"key": "a", "value": 1}, {"key": "b", "value": 2}]},
                {"id": 2, "m": [{"key": "x", "value": 3}]},
                {"id": 3, "m": []},  # empty map
            ],
            # string values + a None (to exercise valueContainsNull=True)
            [
                {"id": 1, "m": [{"key": "a", "value": "foo"}, {"key": "b", "value": None}]},
                {"id": 2, "m": [{"key": "x", "value": "bar"}]},
            ],
        ],
    )
    def test_roundtrip_polars_to_spark_map_column(self, spark, rows):
        """
        Build a Polars DF using the MapType native representation (List[Struct{key,value}]),
        convert to pandas -> Spark DF with a MapType schema, and compare against an expected Spark DF.
        """
        # Infer value type from first non-empty/non-None value
        def infer_value_dtype(data):
            for r in data:
                for kv in r["m"]:
                    if kv["value"] is not None:
                        return kv["value"].__class__
            return int  # default to int if we never find one

        py_type = infer_value_dtype(rows)
        if py_type is int:
            sf_map = sft.MapType(sft.StringType(), sft.IntegerType(), valueContainsNull=True)
            ps_map = pst.MapType(pst.StringType(), pst.IntegerType(), valueContainsNull=True)
        else:
            sf_map = sft.MapType(sft.StringType(), sft.StringType(), valueContainsNull=True)
            ps_map = pst.MapType(pst.StringType(), pst.StringType(), valueContainsNull=True)

        # Build Polars DataFrame with native MapType: List(Struct([key, value]))
        m_dtype = sf_map.to_native()
        df_pl = pl.DataFrame(
            {
                "id": [r["id"] for r in rows],
                "m": [r["m"] for r in rows],  # each is a list of {"key":..., "value": ...}
            },
            schema=[("id", pl.Int32), ("m", m_dtype)],
        )

        # Convert Polars -> pandas
        df_pd = df_pl.to_pandas()

        # Convert list-of-struct [{"key":k, "value":v}, ...] -> real Python dicts {k: v, ...}
        def list_struct_to_dict(lst):
            if lst is None:
                return None
            return {d.get("key"): d.get("value") for d in lst}

        df_pd["m"] = df_pd["m"].apply(list_struct_to_dict)

        # Spark DF from the Polars-built data
        spark_df_from_polars = spark.createDataFrame(
            df_pd,
            schema=pst.StructType(
                [
                    pst.StructField("id", pst.IntegerType(), nullable=False),
                    pst.StructField("m", ps_map, nullable=True),
                ]
            ),
        )

        # Expected Spark DF built directly from Python dicts
        expected_python_rows = [Row(id=r["id"], m=list_struct_to_dict(r["m"])) for r in rows]
        expected_spark_df = spark.createDataFrame(
            expected_python_rows,
            schema=pst.StructType(
                [
                    pst.StructField("id", pst.IntegerType(), nullable=False),
                    pst.StructField("m", ps_map, nullable=True),
                ]
            ),
        )

        # Compare: schema and data (order-insensitive)
        assert spark_df_from_polars.schema == expected_spark_df.schema

        # Sort by id for deterministic comparison
        lhs = spark_df_from_polars.orderBy("id").collect()
        rhs = expected_spark_df.orderBy("id").collect()
        assert lhs == rhs
    
    # Tests for ArrayType
    @pytest.mark.parametrize(
        "element_type_sf, element_type_ps, contains_null",
        [
            (sft.StringType(), pst.StringType(), True),
            (sft.StringType(), pst.StringType(), False),
            (sft.IntegerType(), pst.IntegerType(), True),
            (sft.IntegerType(), pst.IntegerType(), False),
            (sft.DoubleType(), pst.DoubleType(), True),
            (sft.DoubleType(), pst.DoubleType(), False),
            (sft.BooleanType(), pst.BooleanType(), True),
            (sft.BooleanType(), pst.BooleanType(), False),
        ],
    )
    def test_arraytype_equivalence(self, element_type_sf, element_type_ps, contains_null):
        """Test ArrayType basic equivalence with PySpark ArrayType"""
        sf_array = sft.ArrayType(element_type_sf, containsNull=contains_null)
        ps_array = pst.ArrayType(element_type_ps, containsNull=contains_null)
        
        # Basic type properties
        assert sf_array.typeName() == ps_array.typeName() == "array"
        assert sf_array.simpleString() == ps_array.simpleString()
        assert sf_array.containsNull == ps_array.containsNull == contains_null
        
        # Element type equivalence
        assert sf_array.elementType.typeName() == ps_array.elementType.typeName()
        
        # JSON representation
        expected_json = {
            "type": "array",
            "elementType": element_type_ps().jsonValue() if callable(getattr(element_type_ps, "__call__", None)) else element_type_ps.jsonValue(),
            "containsNull": contains_null,
        }
        assert sf_array.jsonValue() == ps_array.jsonValue() == expected_json
        
        # Native polars representation
        native = sf_array.to_native()
        assert isinstance(native, pl.List)
        assert native.inner == element_type_sf.to_native()
    
    @pytest.mark.parametrize(
        "elem_sf, elem_ps, contains_null, rows",
        [
            # int arrays with nulls allowed
            (
                sft.IntegerType(),
                pst.IntegerType(),
                True,
                [
                    {"id": 1, "a": [1, 2, None]},
                    {"id": 2, "a": []},
                    {"id": 3, "a": [5]},
                ],
            ),
            # int arrays without nulls
            (
                sft.IntegerType(),
                pst.IntegerType(),
                False,
                [
                    {"id": 1, "a": [1, 2, 3]},
                    {"id": 2, "a": []},
                    {"id": 3, "a": [5]},
                ],
            ),
            # string arrays with nulls
            (
                sft.StringType(),
                pst.StringType(),
                True,
                [
                    {"id": 1, "a": ["x", None, "y"]},
                    {"id": 2, "a": []},
                    {"id": 3, "a": ["z"]},
                ],
            ),
            # double arrays with nulls
            (
                sft.DoubleType(),
                pst.DoubleType(),
                True,
                [
                    {"id": 1, "a": [1.5, None, 2.0]},
                    {"id": 2, "a": [3.25]},
                    {"id": 3, "a": []},
                ],
            ),
            # nested arrays: array<array<int>> with nulls allowed at top level
            (
                sft.ArrayType(sft.IntegerType()),
                pst.ArrayType(pst.IntegerType()),
                True,
                [
                    {"id": 1, "a": [[1, None], [2]]},
                    {"id": 2, "a": []},
                    {"id": 3, "a": [[3, 4], []]},
                ],
            ),
        ],
    )
    def test_arraytype_roundtrip_polars_to_spark(self, spark, elem_sf, elem_ps, contains_null, rows):
        """
        Test ArrayType roundtrip: Build a Polars DataFrame using ArrayType native representation,
        convert to pandas -> Spark DataFrame, and compare against expected Spark DataFrame.
        """
        pd = pytest.importorskip("pandas", reason="This test requires pandas for Spark conversion")

        # Build ArrayType in both implementations
        sf_array = sft.ArrayType(elem_sf, containsNull=contains_null)
        ps_array = pst.ArrayType(elem_ps, containsNull=contains_null)

        # Quick API parity checks
        assert sf_array.typeName() == ps_array.typeName() == "array"
        assert sf_array.simpleString() == ps_array.simpleString()

        # Build Polars DataFrame using the native dtype from ArrayType
        a_dtype = sf_array.to_native()
        df_pl = pl.DataFrame(
            {
                "id": [r["id"] for r in rows],
                "a": [r["a"] for r in rows],
            },
            schema=[("id", pl.Int32), ("a", a_dtype)],
        )

        # Convert Polars -> pandas
        df_pd = df_pl.to_pandas()

        # Normalize lists after Polars->pandas to fix float upcasting and NaNs
        def resolve_leaf_py_type(dt):
            # traverse nested ArrayType to the leaf data type
            while isinstance(dt, sft.ArrayType):
                dt = dt.elementType
            if isinstance(dt, sft.IntegerType):
                return int
            if isinstance(dt, sft.DoubleType):
                return float
            if isinstance(dt, sft.StringType):
                return str
            return object

        leaf_py_type = resolve_leaf_py_type(elem_sf)

        def normalize(value, leaf_type):
            if value is None:
                return None
            if isinstance(value, list):
                return [normalize(v, leaf_type) for v in value]
            # Handle numpy arrays (from Polars->pandas conversion)
            if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                try:
                    # Convert numpy array to list and normalize each element
                    return [normalize(v, leaf_type) for v in value]
                except (TypeError, ValueError):
                    pass
            # scalar
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            if leaf_type is int:
                return int(value)
            if leaf_type is float:
                return float(value)
            if leaf_type is str:
                return value if isinstance(value, str) else str(value)
            return value

        df_pd["a"] = df_pd["a"].apply(lambda lst: normalize(lst, leaf_py_type))

        # Spark DF from the Polars-built data
        spark_df_from_polars = spark.createDataFrame(
            df_pd,
            schema=pst.StructType(
                [
                    pst.StructField("id", pst.IntegerType(), nullable=False),
                    pst.StructField("a", ps_array, nullable=True),
                ]
            ),
        )

        # Expected Spark DF built directly from Python lists
        expected_python_rows = [Row(id=r["id"], a=r["a"]) for r in rows]
        expected_spark_df = spark.createDataFrame(
            expected_python_rows,
            schema=pst.StructType(
                [
                    pst.StructField("id", pst.IntegerType(), nullable=False),
                    pst.StructField("a", ps_array, nullable=True),
                ]
            ),
        )

        # Compare schema and data (order-insensitive)
        assert spark_df_from_polars.schema == expected_spark_df.schema
        lhs = spark_df_from_polars.orderBy("id").collect()
        rhs = expected_spark_df.orderBy("id").collect()
        assert lhs == rhs
    