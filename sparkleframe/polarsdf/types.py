import json
from typing import Any, Union, Dict, Optional, List, Iterator

import polars as pl


class DataType:
    def to_native(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @classmethod
    def typeName(cls) -> str:
        return cls.__name__[:-4].lower()

    def simpleString(self) -> str:
        return self.typeName()

    def jsonValue(self) -> Union[str, Dict[str, Any]]:
        return self.typeName()

    def json(self) -> str:
        return json.dumps(self.jsonValue(), separators=(",", ":"), sort_keys=True)


class StringType(DataType):

    def to_native(self):
        return pl.Utf8


class IntegerType(DataType):

    def to_native(self):
        return pl.Int32

    def simpleString(self) -> str:
        return "int"


class LongType(DataType):

    def to_native(self):
        return pl.Int64

    def simpleString(self) -> str:
        return "bigint"


class FloatType(DataType):

    def to_native(self):
        return pl.Float32


class DoubleType(DataType):

    def to_native(self):
        return pl.Float64


class BooleanType(DataType):

    def to_native(self):
        return pl.Boolean


class DateType(DataType):

    def to_native(self):
        return pl.Date


class TimestampType(DataType):

    def to_native(self):
        return pl.Datetime


class DecimalType(DataType):
    def __init__(self, precision: int, scale: int):
        self.precision = precision
        self.scale = scale

    def simpleString(self) -> str:
        return "decimal(%d,%d)" % (self.precision, self.scale)

    def jsonValue(self) -> str:
        return "decimal(%d,%d)" % (self.precision, self.scale)

    def __repr__(self) -> str:
        return "DecimalType(%d,%d)" % (self.precision, self.scale)

    def to_native(self):
        return pl.Decimal(self.precision, self.scale)


class ByteType(DataType):
    def to_native(self):
        return pl.Int8

    def simpleString(self) -> str:
        return "tinyint"


class ShortType(DataType):

    def to_native(self):
        return pl.Int16

    def simpleString(self) -> str:
        return "smallint"


class BinaryType(DataType):

    def to_native(self):
        return pl.Binary


class StructField(DataType):
    def __init__(
        self,
        name: str,
        dataType: DataType,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        assert isinstance(dataType, DataType), "dataType %s should be an instance of %s" % (
            dataType,
            DataType,
        )
        assert isinstance(name, str), "field name %s should be a string" % (name)
        self.name = name
        self.dataType = dataType
        self.nullable = nullable
        self.metadata = metadata or {}

    def simpleString(self) -> str:
        return "%s:%s" % (self.name, self.dataType.simpleString())

    def __repr__(self) -> str:
        return "StructField('%s', %s, %s)" % (self.name, self.dataType, str(self.nullable))

    def jsonValue(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.dataType.jsonValue(),
            "nullable": self.nullable,
            "metadata": self.metadata,
        }

    def to_native(self):
        return pl.Field(self.name, self.dataType.to_native())


class StructType(DataType):
    def __init__(self, fields: Optional[List[StructField]] = None):
        if not fields:
            self.fields = []
            self.names = []
        else:
            self.fields = fields
            self.names = [f.name for f in fields]
            assert all(isinstance(f, StructField) for f in fields), "fields should be a list of StructField"

    def __iter__(self) -> Iterator[StructField]:
        """Iterate the fields"""
        return iter(self.fields)

    def __len__(self) -> int:
        """Return the number of fields."""
        return len(self.fields)

    def __getitem__(self, key: Union[str, int]) -> StructField:
        """Access fields by name or slice."""
        if isinstance(key, str):
            for field in self:
                if field.name == key:
                    return field
            raise KeyError("No StructField named {0}".format(key))
        elif isinstance(key, int):
            try:
                return self.fields[key]
            except IndexError:
                raise IndexError("StructType index out of range")
        elif isinstance(key, slice):
            return StructType(self.fields[key])
        else:
            raise ValueError(
                """
                PySparkTypeError(
                    error_class="NOT_INT_OR_SLICE_OR_STR",
                    message_parameters={"arg_name": "key", "arg_type": type(key).__name__},
                )
                """
            )

    def simpleString(self) -> str:
        return "struct<%s>" % (",".join(f.simpleString() for f in self))

    def __repr__(self) -> str:
        return "StructType([%s])" % ", ".join(str(field) for field in self)

    def jsonValue(self) -> Dict[str, Any]:
        return {"type": self.typeName(), "fields": [f.jsonValue() for f in self]}

    def fieldNames(self) -> List[str]:
        return list(self.names)

    def to_native(self):
        return pl.Struct([field.to_native() for field in self.fields])


class Row:
    def __init__(self, *args, **kwargs):
        # FIXME: IMPLEMENT IT
        raise NotImplementedError("Row is not implemented in Polars backend.")


class MapType(DataType):
    """
    Mirror of pyspark.sql.types.MapType

    Parameters
    ----------
    keyType : DataType
        The type of the map keys. (In Spark, keys are non-nullable by definition.)
    valueType : DataType
        The type of the map values.
    valueContainsNull : bool, default True
        Whether map values can be null (Spark semantics).
    """

    def __init__(self, keyType: DataType, valueType: DataType, valueContainsNull: bool = True):
        assert isinstance(keyType, DataType), "keyType %s should be an instance of %s" % (keyType, DataType)
        assert isinstance(valueType, DataType), "valueType %s should be an instance of %s" % (valueType, DataType)
        assert isinstance(valueContainsNull, bool), "valueContainsNull should be a bool"
        # Spark guarantees keys are non-nullable. We mirror that invariant here.
        self.keyType = keyType
        self.valueType = valueType
        self.valueContainsNull = valueContainsNull

    def simpleString(self) -> str:
        # Matches Spark's 'map<keyType,valueType>' format (does not encode valueContainsNull here)
        return "map<%s,%s>" % (self.keyType.simpleString(), self.valueType.simpleString())

    def __repr__(self) -> str:
        return "MapType(%s, %s, %s)" % (self.keyType, self.valueType, str(self.valueContainsNull))

    def jsonValue(self) -> Dict[str, Any]:
        # Matches Spark's JSON shape:
        # {"type":"map","keyType":<...>,"valueType":<...>,"valueContainsNull":true/false}
        return {
            "type": self.typeName(),
            "keyType": self.keyType.jsonValue(),
            "valueType": self.valueType.jsonValue(),
            "valueContainsNull": self.valueContainsNull,
        }

    def to_native(self):
        """
        Polars representation:
        - Encode a map as a List of Structs with two fields: "key" and "value".
        - This layout is stable across Polars versions and interoperates well.
        """
        key_field = pl.Field("key", self.keyType.to_native())
        value_field = pl.Field("value", self.valueType.to_native())
        return pl.List(pl.Struct([key_field, value_field]))


class ArrayType(DataType):
    """
    Mirror of pyspark.sql.types.ArrayType

    Parameters
    ----------
    elementType : DataType
        The type of elements in the array.
    containsNull : bool, default True
        Whether the array can contain null values or not.
    """

    def __init__(self, elementType: DataType, containsNull: bool = True):
        assert isinstance(elementType, DataType), "elementType %s should be an instance of %s" % (
            elementType,
            DataType,
        )
        assert isinstance(containsNull, bool), "containsNull should be a bool"
        self.elementType = elementType
        self.containsNull = containsNull

    @classmethod
    def typeName(cls) -> str:
        # Keep the canonical Spark typename ("array")
        return "array"

    def simpleString(self) -> str:
        # Matches Spark's 'array<elementType>' format (does not encode containsNull here)
        return "array<%s>" % self.elementType.simpleString()

    def __repr__(self) -> str:
        # Matches Spark's repr: ArrayType(elementType, containsNull)
        return "ArrayType(%s, %s)" % (self.elementType, str(self.containsNull))

    def jsonValue(self) -> Dict[str, Any]:
        # Matches Spark's JSON shape:
        # {"type":"array","elementType":<...>,"containsNull":true/false}
        return {
            "type": self.typeName(),
            "elementType": self.elementType.jsonValue(),
            "containsNull": self.containsNull,
        }

    def to_native(self):
        """
        Polars representation:
        - Encode an array as a List of the element's native dtype.
        - Note: Polars allows null elements by default; `containsNull=False` is
          a semantic constraint we track here and can enforce at higher layers.
        """
        return pl.List(self.elementType.to_native())
