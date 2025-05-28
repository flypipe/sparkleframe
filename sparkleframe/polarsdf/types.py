import polars as pl

class DataType:
    def to_native(self):
        raise NotImplementedError

class StringType(DataType):
    def __repr__(self):
        return "StringType()"
    def to_native(self):
        return pl.Utf8

class IntegerType(DataType):
    def __repr__(self):
        return "IntegerType()"
    def to_native(self):
        return pl.Int32

class LongType(DataType):
    def __repr__(self):
        return "LongType()"
    def to_native(self):
        return pl.Int64

class FloatType(DataType):
    def __repr__(self):
        return "FloatType()"
    def to_native(self):
        return pl.Float32

class DoubleType(DataType):
    def __repr__(self):
        return "DoubleType()"
    def to_native(self):
        return pl.Float64

class BooleanType(DataType):
    def __repr__(self):
        return "BooleanType()"
    def to_native(self):
        return pl.Boolean

class DateType(DataType):
    def __repr__(self):
        return "DateType()"
    def to_native(self):
        return pl.Date

class TimestampType(DataType):
    def __repr__(self):
        return "TimestampType()"
    def to_native(self):
        return pl.Datetime

class DecimalType(DataType):
    def __init__(self, precision: int, scale: int):
        self.precision = precision
        self.scale = scale

    def __repr__(self):
        return f"DecimalType({self.precision}, {self.scale})"

    def to_native(self):
        return pl.Decimal(self.precision, self.scale)

class ByteType(DataType):
    def __repr__(self):
        return "ByteType()"
    def to_native(self):
        return pl.Int8

class ShortType(DataType):
    def __repr__(self):
        return "ShortType()"
    def to_native(self):
        return pl.Int16

class BinaryType(DataType):
    def __repr__(self):
        return "BinaryType()"
    def to_native(self):
        return pl.Binary


class StructField:
    def __init__(self, name: str, data_type: DataType, nullable: bool = True):
        self.name = name
        self.data_type = data_type
        self.nullable = nullable

    def __repr__(self):
        return f"StructField('{self.name}', {repr(self.data_type)}, {self.nullable})"

    def to_native(self):
        return (self.name, self.data_type.to_native())


class StructType(DataType):
    def __init__(self, fields: list[StructField]):
        self.fields = fields

    def __repr__(self):
        return f"StructType({self.fields})"

    def to_native(self):
        return pl.Struct([field.to_native() for field in self.fields])


class Row:
    def __init__(self, *args, **kwargs):
        # FIXME: IMPLEMENT IT
        raise NotImplementedError("Row is not implemented in Polars backend.")