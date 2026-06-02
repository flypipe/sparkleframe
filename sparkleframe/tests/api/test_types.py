"""Batch 24 parity tests: types system (simpleString + StructType + DDL).

Covers ONLY public-facing type behaviour not already exercised by
Batch 5 (cast/try_cast value conversion) or Batch 10 (schema inference
on ``createDataFrame`` of bare dict rows). Specifically:

- ``simpleString()`` parity for every atomic ``DataType`` (Byte/Short/Int/Long/
  Float/Double/Decimal/String/Boolean/Date/Timestamp/Binary/Null).
- ``simpleString()`` parity for the three complex types (``ArrayType``,
  ``MapType``, ``StructType``).
- ``StructType`` iteration / equality / ``fieldNames`` / ``__getitem__``.
- ``createDataFrame(data, schema=StructType(...))`` round-trips the schema.
- DDL string parsing (``createDataFrame([], "a INT, b STRING")``) — xfailed on
  pythondf because pythondf has no Spark DDL parser.

The ``T`` fixture (from ``tests/api/conftest.py``) yields the backend's
``types`` module so polarsdf gets polarsdf types and pythondf gets pythondf
types — backends reject foreign type instances.
"""
from __future__ import annotations

import pyspark.sql.types as pst
import pytest


# -----------------------------------------------------------------------------
# Atomic simpleString parity
# -----------------------------------------------------------------------------


_ATOMIC_TYPE_NAMES = [
    "ByteType",
    "ShortType",
    "IntegerType",
    "LongType",
    "FloatType",
    "DoubleType",
    "StringType",
    "BooleanType",
    "DateType",
    "TimestampType",
    "BinaryType",
]


class TestSimpleStringAtomic:
    @pytest.mark.parametrize("type_name", _ATOMIC_TYPE_NAMES)
    def test_simple_string_matches_pyspark(self, T, type_name):
        sf_type = getattr(T, type_name)()
        ps_type = getattr(pst, type_name)()
        assert sf_type.simpleString() == ps_type.simpleString()

    def test_null_type_simple_string(self, T):
        # PySpark exposes NullType (Spark name "void").
        assert T.NullType().simpleString() == pst.NullType().simpleString() == "void"

    @pytest.mark.parametrize("precision, scale", [(10, 2), (5, 0), (20, 10), (38, 18)])
    def test_decimal_simple_string(self, T, precision, scale):
        sf_type = T.DecimalType(precision, scale)
        ps_type = pst.DecimalType(precision, scale)
        assert sf_type.simpleString() == ps_type.simpleString() == f"decimal({precision},{scale})"


# -----------------------------------------------------------------------------
# Complex simpleString parity
# -----------------------------------------------------------------------------


class TestSimpleStringComplex:
    def test_array_of_int(self, T):
        assert (
            T.ArrayType(T.IntegerType()).simpleString()
            == pst.ArrayType(pst.IntegerType()).simpleString()
            == "array<int>"
        )

    def test_array_of_string(self, T):
        assert (
            T.ArrayType(T.StringType()).simpleString()
            == pst.ArrayType(pst.StringType()).simpleString()
            == "array<string>"
        )

    def test_array_nested(self, T):
        sf = T.ArrayType(T.ArrayType(T.LongType()))
        ps = pst.ArrayType(pst.ArrayType(pst.LongType()))
        assert sf.simpleString() == ps.simpleString() == "array<array<bigint>>"

    def test_map_string_int(self, T):
        assert (
            T.MapType(T.StringType(), T.IntegerType()).simpleString()
            == pst.MapType(pst.StringType(), pst.IntegerType()).simpleString()
            == "map<string,int>"
        )

    def test_map_nested_value(self, T):
        sf = T.MapType(T.StringType(), T.MapType(T.StringType(), T.IntegerType()))
        ps = pst.MapType(pst.StringType(), pst.MapType(pst.StringType(), pst.IntegerType()))
        assert sf.simpleString() == ps.simpleString() == "map<string,map<string,int>>"

    def test_struct_simple(self, T):
        sf = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
        ps = pst.StructType([pst.StructField("a", pst.IntegerType()), pst.StructField("b", pst.StringType())])
        assert sf.simpleString() == ps.simpleString() == "struct<a:int,b:string>"

    def test_struct_with_map_and_array(self, T):
        sf = T.StructType(
            [
                T.StructField("m", T.MapType(T.StringType(), T.IntegerType())),
                T.StructField("a", T.ArrayType(T.LongType())),
            ]
        )
        ps = pst.StructType(
            [
                pst.StructField("m", pst.MapType(pst.StringType(), pst.IntegerType())),
                pst.StructField("a", pst.ArrayType(pst.LongType())),
            ]
        )
        assert sf.simpleString() == ps.simpleString()

    def test_struct_nested(self, T):
        sf = T.StructType(
            [
                T.StructField(
                    "outer",
                    T.StructType([T.StructField("inner", T.IntegerType())]),
                )
            ]
        )
        ps = pst.StructType(
            [
                pst.StructField(
                    "outer",
                    pst.StructType([pst.StructField("inner", pst.IntegerType())]),
                )
            ]
        )
        assert sf.simpleString() == ps.simpleString() == "struct<outer:struct<inner:int>>"

    def test_struct_empty(self, T):
        sf = T.StructType([])
        ps = pst.StructType([])
        assert sf.simpleString() == ps.simpleString() == "struct<>"


# -----------------------------------------------------------------------------
# StructType behaviour: iteration / equality / fieldNames / __getitem__
# -----------------------------------------------------------------------------


class TestStructTypeBehaviour:
    def test_field_names(self, T):
        sf = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
        ps = pst.StructType([pst.StructField("a", pst.IntegerType()), pst.StructField("b", pst.StringType())])
        assert sf.fieldNames() == ps.fieldNames() == ["a", "b"]

    def test_iteration_yields_fields_in_order(self, T):
        sf = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
        names = [f.name for f in sf]
        assert names == ["a", "b"]

    def test_len(self, T):
        sf = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
        assert len(sf) == 2

    def test_len_empty(self, T):
        assert len(T.StructType([])) == 0

    def test_equality_same(self, T):
        a = T.StructType([T.StructField("x", T.IntegerType())])
        b = T.StructType([T.StructField("x", T.IntegerType())])
        assert a == b

    def test_equality_diff_field_type(self, T):
        a = T.StructType([T.StructField("x", T.IntegerType())])
        b = T.StructType([T.StructField("x", T.StringType())])
        assert a != b

    def test_equality_diff_field_name(self, T):
        a = T.StructType([T.StructField("x", T.IntegerType())])
        b = T.StructType([T.StructField("y", T.IntegerType())])
        assert a != b

    def test_getitem_by_name(self, T):
        sf = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
        assert sf["a"].name == "a"
        assert sf["b"].dataType.simpleString() == "string"

    def test_getitem_by_index(self, T):
        sf = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
        assert sf[0].name == "a"
        assert sf[1].name == "b"

    def test_getitem_missing_name_raises_keyerror(self, T):
        sf = T.StructType([T.StructField("a", T.IntegerType())])
        with pytest.raises(KeyError):
            _ = sf["missing"]

    def test_getitem_out_of_range_raises_indexerror(self, T):
        sf = T.StructType([T.StructField("a", T.IntegerType())])
        with pytest.raises(IndexError):
            _ = sf[99]

    def test_getitem_slice_returns_structtype(self, T):
        sf = T.StructType(
            [
                T.StructField("a", T.IntegerType()),
                T.StructField("b", T.StringType()),
                T.StructField("c", T.LongType()),
            ]
        )
        sliced = sf[0:2]
        assert isinstance(sliced, T.StructType)
        assert sliced.fieldNames() == ["a", "b"]


# -----------------------------------------------------------------------------
# ArrayType / MapType behaviour
# -----------------------------------------------------------------------------


class TestArrayMapBehaviour:
    def test_array_equality(self, T):
        a = T.ArrayType(T.IntegerType())
        b = T.ArrayType(T.IntegerType())
        c = T.ArrayType(T.LongType())
        assert a == b
        assert a != c

    def test_array_element_type_attribute(self, T):
        a = T.ArrayType(T.IntegerType())
        assert a.elementType.simpleString() == "int"

    def test_map_equality(self, T):
        a = T.MapType(T.StringType(), T.IntegerType())
        b = T.MapType(T.StringType(), T.IntegerType())
        c = T.MapType(T.StringType(), T.LongType())
        d = T.MapType(T.IntegerType(), T.IntegerType())
        assert a == b
        assert a != c
        assert a != d

    def test_map_key_value_attributes(self, T):
        m = T.MapType(T.StringType(), T.IntegerType())
        assert m.keyType.simpleString() == "string"
        assert m.valueType.simpleString() == "int"


# -----------------------------------------------------------------------------
# createDataFrame with a typed StructType schema
# -----------------------------------------------------------------------------


class TestCreateDataFrameWithStructType:
    def test_tuple_rows_with_typed_schema(self, session, T, spark):
        sf_schema = T.StructType(
            [T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())]
        )
        ps_schema = pst.StructType(
            [pst.StructField("a", pst.IntegerType()), pst.StructField("b", pst.StringType())]
        )
        rows = [(1, "x"), (2, "y")]
        sf_df = session.createDataFrame(rows, schema=sf_schema)
        ps_df = spark.createDataFrame(rows, schema=ps_schema)
        assert sf_df.columns == ps_df.columns == ["a", "b"]
        assert sf_df.schema.simpleString() == ps_df.schema.simpleString()

    def test_schema_round_trips_to_input(self, session, T):
        sf_schema = T.StructType(
            [T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())]
        )
        rows = [(1, "x"), (2, "y")]
        sf_df = session.createDataFrame(rows, schema=sf_schema)
        assert sf_df.schema.simpleString() == sf_schema.simpleString()

    def test_dtypes_reflects_typed_schema(self, session, T, spark):
        sf_schema = T.StructType(
            [
                T.StructField("a", T.IntegerType()),
                T.StructField("b", T.StringType()),
                T.StructField("c", T.LongType()),
            ]
        )
        ps_schema = pst.StructType(
            [
                pst.StructField("a", pst.IntegerType()),
                pst.StructField("b", pst.StringType()),
                pst.StructField("c", pst.LongType()),
            ]
        )
        rows = [(1, "x", 10), (2, "y", 20)]
        sf_df = session.createDataFrame(rows, schema=sf_schema)
        ps_df = spark.createDataFrame(rows, schema=ps_schema)
        assert sf_df.dtypes == ps_df.dtypes

    def test_typed_schema_long_vs_int(self, session, T, spark):
        # Verify the declared type — not Python int inference — wins.
        sf_schema = T.StructType([T.StructField("a", T.IntegerType())])
        ps_schema = pst.StructType([pst.StructField("a", pst.IntegerType())])
        rows = [(1,), (2,)]
        sf_df = session.createDataFrame(rows, schema=sf_schema)
        ps_df = spark.createDataFrame(rows, schema=ps_schema)
        assert sf_df.dtypes == ps_df.dtypes
        # PySpark default inference for python int would be "bigint"; the
        # declared schema overrides that.
        assert sf_df.dtypes[0][1] == "int"


# -----------------------------------------------------------------------------
# DDL string parsing
# -----------------------------------------------------------------------------


class TestDDLStringSchema:
    @pytest.mark.xfail(
        reason="pythondf has no Spark DDL parser; polarsdf passes the DDL to Polars which mis-handles non-numeric values",
        strict=False,
    )
    def test_simple_ddl_schema(self, session):
        rows = [(1, "x"), (2, "y")]
        df = session.createDataFrame(rows, schema="a INT, b STRING")
        assert df.columns == ["a", "b"]
        assert df.schema.simpleString() == "struct<a:int,b:string>"
