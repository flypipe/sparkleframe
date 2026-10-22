"""Type system for the pure-Python engine.

The engine owns its ``DataType`` hierarchy instead of re-exporting ``pyspark.sql.types``: under
``activate()`` the real ``pyspark`` package is replaced by a mock (and a JVM-free deployment does
not install pyspark at all), yet ``pyspark.sql.types`` must still resolve to working classes. The
classes mirror PySpark's public surface (constructor signatures, ``typeName``/``simpleString``/
``jsonValue``/``json``, ``repr``, equality) so user code and libraries such as flypipe treat them
like the real thing. Unlike ``polarsdf/types.py`` they carry no engine-native dtype — values are
plain Python objects (see ``docs/design/python-engine-ast.md``).

:func:`parse_ddl_type` turns a Spark SQL type name (``"int"``, ``"decimal(6,4)"``,
``"array<string>"``) into a ``DataType``, for APIs that accept a type as a string
(``Column.cast("string")``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class DataType:
    """Base class of every engine data type."""

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


class NullType(DataType):
    @classmethod
    def typeName(cls) -> str:
        return "void"


class StringType(DataType):
    pass


class BinaryType(DataType):
    pass


class BooleanType(DataType):
    pass


class DateType(DataType):
    pass


class TimestampType(DataType):
    pass


class TimestampNTZType(DataType):
    @classmethod
    def typeName(cls) -> str:
        return "timestamp_ntz"


class DayTimeIntervalType(DataType):
    """``interval day [to second]``; values are ``datetime.timedelta``, as PySpark collects them."""

    DAY, HOUR, MINUTE, SECOND = 0, 1, 2, 3
    _FIELD_NAMES = {DAY: "day", HOUR: "hour", MINUTE: "minute", SECOND: "second"}

    def __init__(self, startField: int = DAY, endField: int = SECOND) -> None:
        self.startField = startField
        self.endField = endField

    def simpleString(self) -> str:
        start, end = self._FIELD_NAMES[self.startField], self._FIELD_NAMES[self.endField]
        return f"interval {start}" if start == end else f"interval {start} to {end}"

    def jsonValue(self) -> str:
        return self.simpleString()

    def __repr__(self) -> str:
        return "DayTimeIntervalType(%d, %d)" % (self.startField, self.endField)


class ByteType(DataType):
    def simpleString(self) -> str:
        return "tinyint"


class ShortType(DataType):
    def simpleString(self) -> str:
        return "smallint"


class IntegerType(DataType):
    def simpleString(self) -> str:
        return "int"


class LongType(DataType):
    def simpleString(self) -> str:
        return "bigint"


class FloatType(DataType):
    pass


class DoubleType(DataType):
    pass


class DecimalType(DataType):
    def __init__(self, precision: int = 10, scale: int = 0) -> None:
        self.precision = precision
        self.scale = scale

    def simpleString(self) -> str:
        return "decimal(%d,%d)" % (self.precision, self.scale)

    def jsonValue(self) -> str:
        return self.simpleString()

    def __repr__(self) -> str:
        return "DecimalType(%d,%d)" % (self.precision, self.scale)


class ArrayType(DataType):
    def __init__(self, elementType: DataType, containsNull: bool = True) -> None:
        assert isinstance(elementType, DataType), f"elementType {elementType} should be an instance of {DataType}"
        assert isinstance(containsNull, bool), "containsNull should be a bool"
        self.elementType = elementType
        self.containsNull = containsNull

    def simpleString(self) -> str:
        return "array<%s>" % self.elementType.simpleString()

    def __repr__(self) -> str:
        return "ArrayType(%s, %s)" % (self.elementType, self.containsNull)

    def jsonValue(self) -> Dict[str, Any]:
        return {
            "type": self.typeName(),
            "elementType": self.elementType.jsonValue(),
            "containsNull": self.containsNull,
        }


class MapType(DataType):
    def __init__(self, keyType: DataType, valueType: DataType, valueContainsNull: bool = True) -> None:
        assert isinstance(keyType, DataType), f"keyType {keyType} should be an instance of {DataType}"
        assert isinstance(valueType, DataType), f"valueType {valueType} should be an instance of {DataType}"
        assert isinstance(valueContainsNull, bool), "valueContainsNull should be a bool"
        self.keyType = keyType
        self.valueType = valueType
        self.valueContainsNull = valueContainsNull

    def simpleString(self) -> str:
        return "map<%s,%s>" % (self.keyType.simpleString(), self.valueType.simpleString())

    def __repr__(self) -> str:
        return "MapType(%s, %s, %s)" % (self.keyType, self.valueType, self.valueContainsNull)

    def jsonValue(self) -> Dict[str, Any]:
        return {
            "type": self.typeName(),
            "keyType": self.keyType.jsonValue(),
            "valueType": self.valueType.jsonValue(),
            "valueContainsNull": self.valueContainsNull,
        }


class StructField(DataType):
    def __init__(
        self, name: str, dataType: DataType, nullable: bool = True, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        assert isinstance(dataType, DataType), f"dataType {dataType} should be an instance of {DataType}"
        assert isinstance(name, str), f"field name {name} should be a string"
        self.name = name
        self.dataType = dataType
        self.nullable = nullable
        self.metadata = metadata or {}

    def simpleString(self) -> str:
        return "%s:%s" % (self.name, self.dataType.simpleString())

    def __repr__(self) -> str:
        return "StructField('%s', %s, %s)" % (self.name, self.dataType, self.nullable)

    def jsonValue(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.dataType.jsonValue(),
            "nullable": self.nullable,
            "metadata": self.metadata,
        }


class StructType(DataType):
    def __init__(self, fields: Optional[List[StructField]] = None) -> None:
        self.fields: List[StructField] = list(fields) if fields else []
        assert all(isinstance(f, StructField) for f in self.fields), "fields should be a list of StructField"
        self.names: List[str] = [f.name for f in self.fields]

    def add(
        self,
        field: Union[str, StructField],
        data_type: Union[str, DataType, None] = None,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StructType":
        if isinstance(field, StructField):
            self.fields.append(field)
        else:
            dtype = parse_ddl_type(data_type) if isinstance(data_type, str) else data_type
            self.fields.append(StructField(field, dtype, nullable, metadata))
        self.names.append(self.fields[-1].name)
        return self

    def __iter__(self) -> Iterator[StructField]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, key: Union[str, int, slice]) -> Union[StructField, "StructType"]:
        if isinstance(key, str):
            for field in self.fields:
                if field.name == key:
                    return field
            raise KeyError(f"No StructField named {key}")
        if isinstance(key, int):
            try:
                return self.fields[key]
            except IndexError:
                raise IndexError("StructType index out of range") from None
        if isinstance(key, slice):
            return StructType(self.fields[key])
        raise TypeError(
            f"[NOT_INT_OR_SLICE_OR_STR] Argument `key` should be an int, slice or str, got {type(key).__name__}."
        )

    def fieldNames(self) -> List[str]:
        return list(self.names)

    def simpleString(self) -> str:
        return "struct<%s>" % ",".join(f.simpleString() for f in self.fields)

    def __repr__(self) -> str:
        return "StructType([%s])" % ", ".join(repr(f) for f in self.fields)

    def jsonValue(self) -> Dict[str, Any]:
        return {"type": self.typeName(), "fields": [f.jsonValue() for f in self.fields]}


class Row(tuple):
    """A struct value: a tuple whose items are also reachable by field name (``row.a`` / ``row["a"]``)."""

    __fields__: Tuple[str, ...]

    def __new__(cls, *args: Any, **kwargs: Any) -> "Row":
        if args and kwargs:
            raise ValueError("Can not use both args and kwargs to create Row")
        if kwargs:
            row = tuple.__new__(cls, kwargs.values())
            row.__fields__ = tuple(kwargs)
            return row
        return tuple.__new__(cls, args)

    @classmethod
    def from_fields(cls, names: List[str], values: List[Any]) -> "Row":
        row = tuple.__new__(cls, values)
        row.__fields__ = tuple(names)
        return row

    def asDict(self, recursive: bool = False) -> Dict[str, Any]:
        if not hasattr(self, "__fields__"):
            raise TypeError("Cannot convert a Row class into dict")
        if not recursive:
            return dict(zip(self.__fields__, self))
        return {name: _as_dict_value(value) for name, value in zip(self.__fields__, self)}

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        try:
            return super().__getitem__(self.__fields__.index(item))
        except (AttributeError, ValueError):
            raise KeyError(item) from None

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item) from None

    def __repr__(self) -> str:
        if hasattr(self, "__fields__"):
            return "Row(%s)" % ", ".join(f"{k}={v!r}" for k, v in zip(self.__fields__, self))
        return "<Row(%s)>" % ", ".join(repr(v) for v in self)


def _as_dict_value(value: Any) -> Any:
    if isinstance(value, Row):
        return value.asDict(recursive=True)
    if isinstance(value, list):
        return [_as_dict_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _as_dict_value(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- #
# DDL type names
# --------------------------------------------------------------------------- #

_ATOMIC_DDL_NAMES = {
    "string": StringType,
    "binary": BinaryType,
    "boolean": BooleanType,
    "date": DateType,
    "timestamp": TimestampType,
    "timestamp_ntz": TimestampNTZType,
    "tinyint": ByteType,
    "byte": ByteType,
    "smallint": ShortType,
    "short": ShortType,
    "int": IntegerType,
    "integer": IntegerType,
    "bigint": LongType,
    "long": LongType,
    "float": FloatType,
    "real": FloatType,
    "double": DoubleType,
    "void": NullType,
}
_DDL_TOKEN_RE = re.compile(r"\s*(?:(?P<word>[A-Za-z_][A-Za-z0-9_]*)|(?P<num>[0-9]+)|(?P<sym>[<>(),:])|(?P<bad>\S))")


def parse_ddl_type(ddl: str) -> DataType:
    """Parse a Spark SQL type name (``int``, ``decimal(6,4)``, ``array<string>``, ``struct<a:int>``).

    Raises ``ValueError`` carrying Spark's ``PARSE_SYNTAX_ERROR`` / ``UNSUPPORTED_DATATYPE`` /
    decimal-range conditions for malformed or unknown names.
    """
    tokens = []
    for match in _DDL_TOKEN_RE.finditer(ddl):
        if match["bad"] is not None:
            raise ValueError(f"[PARSE_SYNTAX_ERROR] Syntax error at or near '{match['bad']}'.")
        tokens.append(match["word"] or match["num"] or match["sym"])
    parser = _DdlParser(tokens)
    dtype = parser.data_type()
    parser.expect_end()
    return dtype


def parse_ddl_schema(ddl: str, pyspark_colons: bool = False) -> DataType:
    """A schema string the way Spark's ``DataType.fromDDL`` reads it (``from_json``, ``createDataFrame``).

    A field list (``"a int, b string"``) is a struct; otherwise the string is a single type
    (``"int"``, ``"struct<a:int>"``, ``"array<string>"``). Only PySpark's ``_parse_datatype_string``
    (``pyspark_colons``) also takes ``name: type`` in the top-level field list.
    """
    try:
        if not pyspark_colons and _has_top_level_colon(ddl):
            raise ValueError("[PARSE_SYNTAX_ERROR] Syntax error at or near ':'.")
        return parse_ddl_type(f"struct<{ddl}>")
    except ValueError:
        return parse_ddl_type(ddl)


def _has_top_level_colon(ddl: str) -> bool:
    depth = 0
    for char in ddl:
        depth += {"<": 1, ">": -1}.get(char, 0)
        if char == ":" and depth == 0:
            return True
    return False


class _DdlParser:
    def __init__(self, tokens: List[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("[PARSE_SYNTAX_ERROR] Syntax error at or near end of input.")
        self.pos += 1
        return token

    def _expect(self, symbol: str) -> None:
        token = self._next()
        if token != symbol:
            raise ValueError(f"[PARSE_SYNTAX_ERROR] Syntax error at or near '{token}'.")

    def expect_end(self) -> None:
        if self._peek() is not None:
            raise ValueError(f"[PARSE_SYNTAX_ERROR] Syntax error at or near '{self._peek()}'.")

    def _int(self) -> int:
        token = self._next()
        if not token.isdigit():
            raise ValueError(f"[PARSE_SYNTAX_ERROR] Syntax error at or near '{token}'.")
        return int(token)

    def data_type(self) -> DataType:
        word = self._next()
        name = word.lower()
        if name in ("decimal", "dec", "numeric"):
            return self._decimal()
        if name == "array":
            self._expect("<")
            element = self.data_type()
            self._expect(">")
            return ArrayType(element)
        if name == "map":
            self._expect("<")
            key = self.data_type()
            self._expect(",")
            value = self.data_type()
            self._expect(">")
            return MapType(key, value)
        if name == "struct":
            return self._struct()
        if name in _ATOMIC_DDL_NAMES:
            return _ATOMIC_DDL_NAMES[name]()
        if not (word[0].isalpha() or word[0] == "_"):
            raise ValueError(f"[PARSE_SYNTAX_ERROR] Syntax error at or near '{word}'.")
        raise ValueError(f'[UNSUPPORTED_DATATYPE] Unsupported data type "{word.upper()}".')

    def _decimal(self) -> DecimalType:
        if self._peek() != "(":
            return DecimalType(10, 0)
        self._next()
        precision = self._int()
        scale = 0
        if self._peek() == ",":
            self._next()
            scale = self._int()
        self._expect(")")
        if precision > 38:
            raise ValueError(
                f"[DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION] Decimal precision {precision} exceeds max precision 38."
            )
        if scale > precision:
            raise ValueError(f"Decimal scale ({scale}) cannot be greater than precision ({precision}).")
        return DecimalType(precision, scale)

    def _struct(self) -> StructType:
        self._expect("<")
        fields: List[StructField] = []
        if self._peek() == ">":
            self._next()
            return StructType(fields)
        while True:
            name = self._next()
            if self._peek() == ":":
                self._next()
            fields.append(StructField(name, self.data_type()))
            token = self._next()
            if token == ">":
                return StructType(fields)
            if token != ",":
                raise ValueError(f"[PARSE_SYNTAX_ERROR] Syntax error at or near '{token}'.")
