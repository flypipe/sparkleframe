"""Minimal PySpark-compatible type system for pythondf.

Each DataType subclass exposes ``simpleString()`` for type names and
``_cast(value, strict)`` for value conversion. ``strict=True`` raises on
malformed input (Spark ANSI behavior); ``strict=False`` returns ``None``
(``try_cast`` semantics).
"""
from __future__ import annotations

import re
import struct
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, getcontext
from typing import Any, List, Optional


# Spark string -> boolean literals (case-insensitive). Matches Spark 4 behaviour:
# Cast.scala BooleanLiteral.canCast accepts {true,false,t,f,y,n,yes,no,1,0}.
_TRUE_LITERALS = {"true", "yes", "1", "y", "t"}
_FALSE_LITERALS = {"false", "no", "0", "n", "f"}

# Spark integer literal grammar: optional sign + decimal digits. Whitespace
# is trimmed by the caller. Decimal points and exponents are NOT accepted
# when casting a STRING to an integer type (Spark ANSI CAST_INVALID_INPUT).
_INT_LITERAL_RE = re.compile(r"^[+-]?\d+$")


class DataType:
    def simpleString(self) -> str:
        return self.__class__.__name__.replace("Type", "").lower()

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        return value

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))


class _IntegralType(DataType):
    """Integral type with bounded range (Spark ANSI overflow → raise/null)."""
    _min: int = 0
    _max: int = 0

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        try:
            if isinstance(value, bool):
                ival = 1 if value else 0
            elif isinstance(value, int):
                ival = value
            elif isinstance(value, float):
                # Spark truncates float toward zero when casting to integral types.
                if value != value or value in (float("inf"), float("-inf")):
                    if strict:
                        raise ValueError(f"Cannot cast non-finite float to integer")
                    return None
                ival = int(value)
            elif isinstance(value, Decimal):
                ival = int(value)
            elif isinstance(value, str):
                s = value.strip()
                if not _INT_LITERAL_RE.match(s):
                    raise ValueError(f"Cannot cast string {value!r} to integer")
                ival = int(s)
            else:
                raise TypeError(f"Cannot cast {type(value).__name__} to integer")
        except (ValueError, TypeError):
            if strict:
                raise
            return None

        if ival < self._min or ival > self._max:
            if strict:
                raise OverflowError(
                    f"value {ival} out of range for {self.simpleString()} "
                    f"[{self._min}, {self._max}]"
                )
            return None
        return ival


class ByteType(_IntegralType):
    _min, _max = -(1 << 7), (1 << 7) - 1
    def simpleString(self) -> str: return "tinyint"


class ShortType(_IntegralType):
    _min, _max = -(1 << 15), (1 << 15) - 1
    def simpleString(self) -> str: return "smallint"


class IntegerType(_IntegralType):
    _min, _max = -(1 << 31), (1 << 31) - 1
    def simpleString(self) -> str: return "int"


class LongType(_IntegralType):
    _min, _max = -(1 << 63), (1 << 63) - 1
    def simpleString(self) -> str: return "bigint"


class _FloatingType(DataType):
    _py_type: type = float

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        try:
            if isinstance(value, bool):
                return self._py_type(1.0 if value else 0.0)
            if isinstance(value, (int, float, Decimal)):
                return self._py_type(value)
            if isinstance(value, str):
                return self._py_type(value.strip())
            raise TypeError(f"Cannot cast {type(value).__name__} to {self.simpleString()}")
        except (ValueError, TypeError):
            if strict:
                raise
            return None


class FloatType(_FloatingType):
    """IEEE 754 single-precision. Round-trips through ``struct`` to match
    PySpark's FloatType representation (e.g. ``1.7`` -> ``1.7000000476837158``)."""

    def simpleString(self) -> str: return "float"

    def _cast(self, value: Any, strict: bool) -> Any:
        v = super()._cast(value, strict)
        if v is None:
            return None
        try:
            return struct.unpack("f", struct.pack("f", v))[0]
        except (struct.error, OverflowError):
            if strict:
                raise
            return None


class DoubleType(_FloatingType):
    def simpleString(self) -> str: return "double"


class DecimalType(DataType):
    def __init__(self, precision: int = 10, scale: int = 0):
        self.precision = precision
        self.scale = scale

    def simpleString(self) -> str:
        return f"decimal({self.precision},{self.scale})"

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        try:
            if isinstance(value, bool):
                d = Decimal(1 if value else 0)
            elif isinstance(value, Decimal):
                d = value
            elif isinstance(value, (int,)):
                d = Decimal(value)
            elif isinstance(value, float):
                if value != value or value in (float("inf"), float("-inf")):
                    raise InvalidOperation("non-finite float")
                d = Decimal(str(value))
            elif isinstance(value, str):
                d = Decimal(value.strip())
            else:
                raise TypeError(f"Cannot cast {type(value).__name__} to decimal")
        except (InvalidOperation, ValueError, TypeError):
            if strict:
                raise
            return None

        # Quantize to the target scale (Spark uses HALF_UP rounding).
        try:
            from decimal import ROUND_HALF_UP
            quant = Decimal(1).scaleb(-self.scale)
            d_scaled = d.quantize(quant, rounding=ROUND_HALF_UP)
        except (InvalidOperation, ValueError):
            if strict:
                raise
            return None

        # Enforce precision (integer digits must fit precision - scale).
        sign, digits, exponent = d_scaled.as_tuple()
        # Total number of significant digits in the unscaled integer.
        # Effective integer-part digits = len(digits) + exponent (since scale gives exponent=-scale).
        # Equivalent: max precision allowed = self.precision; total digits must be <= precision.
        total_digits = len(digits) if d_scaled != 0 else 1
        # Adjust: if the value < 1 and digits are leading zeros, that's OK — but Decimal's
        # tuple representation already drops leading zeros for nonzero values.
        if total_digits > self.precision:
            if strict:
                raise OverflowError(
                    f"value {d_scaled} exceeds precision {self.precision} for {self.simpleString()}"
                )
            return None
        return d_scaled

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DecimalType) and self.precision == other.precision and self.scale == other.scale

    def __hash__(self) -> int:
        return hash((type(self), self.precision, self.scale))


class StringType(DataType):
    def simpleString(self) -> str: return "string"

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


class BinaryType(DataType):
    def simpleString(self) -> str: return "binary"

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, str):
            return value.encode()
        if strict:
            raise TypeError(f"Cannot cast {type(value).__name__} to binary")
        return None


class BooleanType(DataType):
    def simpleString(self) -> str: return "boolean"

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _TRUE_LITERALS:
                return True
            if v in _FALSE_LITERALS:
                return False
            if strict:
                raise ValueError(f"Cannot cast '{value}' to boolean")
            return None
        if strict:
            raise TypeError(f"Cannot cast {type(value).__name__} to boolean")
        return None


_DATE_PARSE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T].*)?$")


class DateType(DataType):
    def simpleString(self) -> str: return "date"

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            m = _DATE_PARSE_RE.match(value.strip())
            if m is None:
                if strict:
                    raise ValueError(f"Cannot cast {value!r} to date")
                return None
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                if strict:
                    raise
                return None
        if strict:
            raise TypeError(f"Cannot cast {type(value).__name__} to date")
        return None


class TimestampType(DataType):
    def simpleString(self) -> str: return "timestamp"

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError:
                if strict:
                    raise
                return None
        if strict:
            raise TypeError(f"Cannot cast {type(value).__name__} to timestamp")
        return None


class NullType(DataType):
    def simpleString(self) -> str: return "void"
    def _cast(self, value: Any, strict: bool) -> Any: return None


class ArrayType(DataType):
    def __init__(self, element_type: DataType, contains_null: bool = True):
        self.elementType = element_type
        self.containsNull = contains_null

    def simpleString(self) -> str:
        return f"array<{self.elementType.simpleString()}>"

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            if strict:
                raise TypeError(f"Cannot cast {type(value).__name__} to array")
            return None
        return [self.elementType._cast(v, strict) for v in value]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArrayType) and self.elementType == other.elementType

    def __hash__(self) -> int:
        return hash((type(self), self.elementType))


class MapType(DataType):
    def __init__(self, key_type: DataType, value_type: DataType, value_contains_null: bool = True):
        self.keyType = key_type
        self.valueType = value_type
        self.valueContainsNull = value_contains_null

    def simpleString(self) -> str:
        return f"map<{self.keyType.simpleString()},{self.valueType.simpleString()}>"

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        if not isinstance(value, dict):
            if strict:
                raise TypeError(f"Cannot cast {type(value).__name__} to map")
            return None
        return {self.keyType._cast(k, strict): self.valueType._cast(v, strict) for k, v in value.items()}

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MapType)
            and self.keyType == other.keyType
            and self.valueType == other.valueType
        )

    def __hash__(self) -> int:
        return hash((type(self), self.keyType, self.valueType))


class StructField:
    def __init__(self, name: str, data_type: DataType, nullable: bool = True):
        self.name = name
        self.dataType = data_type
        self.nullable = nullable

    def simpleString(self) -> str:
        return f"{self.name}:{self.dataType.simpleString()}"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StructField)
            and self.name == other.name
            and self.dataType == other.dataType
            and self.nullable == other.nullable
        )

    def __hash__(self) -> int:
        return hash((self.name, self.dataType, self.nullable))


class StructType(DataType):
    def __init__(self, fields: Optional[List[StructField]] = None):
        self.fields = list(fields or [])

    def add(self, field: StructField) -> "StructType":
        self.fields.append(field)
        return self

    def simpleString(self) -> str:
        return "struct<" + ",".join(f.simpleString() for f in self.fields) + ">"

    def fieldNames(self) -> List[str]:
        return [f.name for f in self.fields]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, key):
        """Access a field by name (str), position (int), or slice.

        Mirrors ``pyspark.sql.types.StructType.__getitem__``: returns the
        matched ``StructField``, raises ``KeyError`` for unknown names,
        ``IndexError`` for out-of-range positions, and ``ValueError`` for
        other key types.
        """
        if isinstance(key, str):
            for field in self.fields:
                if field.name == key:
                    return field
            raise KeyError(f"No StructField named {key}")
        if isinstance(key, bool):
            # ``bool`` is a subclass of ``int``; reject explicitly to match
            # PySpark behaviour (booleans are not valid struct indices).
            raise ValueError(f"key should be a string, int or slice, got {type(key).__name__}")
        if isinstance(key, int):
            try:
                return self.fields[key]
            except IndexError:
                raise IndexError("StructType index out of range")
        if isinstance(key, slice):
            return StructType(self.fields[key])
        raise ValueError(f"key should be a string, int or slice, got {type(key).__name__}")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StructType) and self.fields == other.fields

    def __hash__(self) -> int:
        return hash((type(self), tuple(self.fields)))

    def _cast(self, value: Any, strict: bool) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {f.name: f.dataType._cast(value.get(f.name), strict) for f in self.fields}
        if strict:
            raise TypeError(f"Cannot cast {type(value).__name__} to struct")
        return None


_NAME_TO_TYPE = {
    "byte": ByteType, "tinyint": ByteType,
    "short": ShortType, "smallint": ShortType,
    "int": IntegerType, "integer": IntegerType,
    "long": LongType, "bigint": LongType,
    "float": FloatType,
    "double": DoubleType,
    "string": StringType,
    "binary": BinaryType,
    "bool": BooleanType, "boolean": BooleanType,
    "date": DateType,
    "timestamp": TimestampType,
    "void": NullType, "null": NullType,
}


def spark_type_name_to_type(name: str) -> DataType:
    """Resolve a Spark type-name string (e.g. ``"int"``, ``"decimal(10,2)"``) to a DataType."""
    s = name.strip().lower()
    if s.startswith("decimal"):
        inside = s[len("decimal"):].strip()
        if inside.startswith("(") and inside.endswith(")"):
            parts = [p.strip() for p in inside[1:-1].split(",")]
            precision = int(parts[0])
            scale = int(parts[1]) if len(parts) > 1 else 0
            return DecimalType(precision, scale)
        return DecimalType()
    cls = _NAME_TO_TYPE.get(s)
    if cls is None:
        raise ValueError(f"Unknown Spark type name: {name!r}")
    return cls()
