"""The python engine's own type system, checked against real PySpark's type API.

The engine defines its ``DataType`` hierarchy without importing pyspark (it must work under
``activate()`` with no pyspark installed), so these tests pin its public surface — ``repr``,
``simpleString``, ``json``, equality, ``StructType`` access, ``Row`` — to PySpark's, and its DDL
type-name parser to Spark's (valid and malformed names).
"""

import pyspark.sql.types as ST
import pytest
from pyspark.sql import SparkSession

from sparkleframe.python import types as T

# (engine type, the equivalent real PySpark type)
_PAIRS = [
    (T.NullType(), ST.NullType()),
    (T.StringType(), ST.StringType()),
    (T.BinaryType(), ST.BinaryType()),
    (T.BooleanType(), ST.BooleanType()),
    (T.DateType(), ST.DateType()),
    (T.TimestampType(), ST.TimestampType()),
    (T.TimestampNTZType(), ST.TimestampNTZType()),
    (T.ByteType(), ST.ByteType()),
    (T.ShortType(), ST.ShortType()),
    (T.IntegerType(), ST.IntegerType()),
    (T.LongType(), ST.LongType()),
    (T.FloatType(), ST.FloatType()),
    (T.DoubleType(), ST.DoubleType()),
    (T.DecimalType(), ST.DecimalType()),
    (T.DecimalType(13, 2), ST.DecimalType(13, 2)),
    (T.ArrayType(T.StringType()), ST.ArrayType(ST.StringType())),
    (T.ArrayType(T.IntegerType(), False), ST.ArrayType(ST.IntegerType(), False)),
    (T.MapType(T.StringType(), T.LongType()), ST.MapType(ST.StringType(), ST.LongType())),
    (
        T.StructType([T.StructField("a", T.DecimalType(5, 4)), T.StructField("b", T.ArrayType(T.DateType()), False)]),
        ST.StructType(
            [ST.StructField("a", ST.DecimalType(5, 4)), ST.StructField("b", ST.ArrayType(ST.DateType()), False)]
        ),
    ),
]


@pytest.mark.parametrize("engine_type, spark_type", _PAIRS, ids=lambda t: repr(t))
def test_type_surface_matches_pyspark(engine_type, spark_type):
    assert repr(engine_type) == repr(spark_type)
    assert engine_type.simpleString() == spark_type.simpleString()
    assert engine_type.typeName() == spark_type.typeName()
    assert engine_type.json() == spark_type.json()


def test_equality_and_hash_are_structural():
    assert T.DecimalType(13, 2) == T.DecimalType(13, 2)
    assert T.DecimalType(13, 2) != T.DecimalType(13, 4)
    assert T.IntegerType() != T.LongType()
    assert len({T.ArrayType(T.StringType()), T.ArrayType(T.StringType()), T.ArrayType(T.LongType())}) == 2


class TestStructType:
    def test_access_by_name_index_and_slice(self):
        schema = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
        assert schema["b"].dataType == T.StringType()
        assert schema[0].name == "a"
        assert schema[1:].names == ["b"]
        assert schema.fieldNames() == ["a", "b"]
        assert [f.name for f in schema] == ["a", "b"]
        assert len(schema) == 2

    def test_missing_name_and_bad_key_raise_like_pyspark(self):
        schema = T.StructType([T.StructField("a", T.IntegerType())])
        with pytest.raises(KeyError):
            schema["missing"]
        with pytest.raises(IndexError):
            schema[5]
        with pytest.raises(TypeError, match="NOT_INT_OR_SLICE_OR_STR"):
            schema[1.5]

    def test_add_accepts_field_or_name_and_ddl_type(self):
        schema = T.StructType().add("a", "decimal(6,4)").add(T.StructField("b", T.StringType(), False))
        spark_schema = ST.StructType().add("a", "decimal(6,4)").add(ST.StructField("b", ST.StringType(), False))
        assert repr(schema) == repr(spark_schema)


class TestRow:
    def test_named_row_access_and_as_dict(self):
        inner = T.Row.from_fields(["x"], [1])
        row = T.Row(a=1, b=inner, c=[inner])
        assert row.a == 1 and row["a"] == 1 and row[0] == 1
        assert row.asDict() == {"a": 1, "b": inner, "c": [inner]}
        assert row.asDict(recursive=True) == {"a": 1, "b": {"x": 1}, "c": [{"x": 1}]}
        assert repr(row.b) == repr(ST.Row(x=1))

    def test_missing_field_raises(self):
        row = T.Row(a=1)
        with pytest.raises(AttributeError):
            row.zzz
        with pytest.raises(KeyError):
            row["zzz"]

    def test_mixing_args_and_kwargs_raises(self):
        with pytest.raises(ValueError):
            T.Row(1, a=2)


@pytest.fixture(scope="module")
def ddl_spark():
    return SparkSession.builder.getOrCreate()


@pytest.mark.parametrize(
    "ddl",
    [
        "string",
        "STRING",
        "int",
        "integer",
        "bigint",
        "long",
        "smallint",
        "tinyint",
        "double",
        "float",
        "boolean",
        "date",
        "timestamp",
        "timestamp_ntz",
        "binary",
        "decimal",
        "decimal(6,4)",
        "decimal(10)",
        "array<string>",
        "map<string,array<int>>",
        "struct<a:int,b:decimal(13,2)>",
    ],
)
def test_parse_ddl_type_matches_spark(ddl, ddl_spark):
    spark_type = ddl_spark.createDataFrame([], f"c {ddl}").schema["c"].dataType
    assert repr(T.parse_ddl_type(ddl)) == repr(spark_type)


@pytest.mark.parametrize(
    "ddl, error",
    [
        ("decimal(6,", "PARSE_SYNTAX_ERROR"),
        ("array<string", "PARSE_SYNTAX_ERROR"),
        ("int int", "PARSE_SYNTAX_ERROR"),
        ("", "PARSE_SYNTAX_ERROR"),
        ("decimal(40,2)", "DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION"),
        ("decimal(2,4)", r"Decimal scale \(4\) cannot be greater than precision \(2\)"),
        ("notatype", "UNSUPPORTED_DATATYPE"),
    ],
)
def test_parse_ddl_type_malformed_raises_like_spark(ddl, error, ddl_spark):
    with pytest.raises(Exception, match=error):
        ddl_spark.createDataFrame([], f"c {ddl}").collect()
    with pytest.raises(ValueError, match=error):
        T.parse_ddl_type(ddl)
