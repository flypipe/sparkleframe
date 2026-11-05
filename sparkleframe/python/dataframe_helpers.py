"""Shared, private helpers for ``dataframe.py`` (pure-Python engine).

Per the repo convention (CLAUDE.md), shared/private logic for the public ``dataframe`` surface
lives here rather than in ``dataframe.py``:

- row/schema plumbing: normalizing ``createDataFrame`` input, schema inference, records;
- Spark's auto-generated output names for unaliased expressions;
- grouping keys and sort ordering with Spark semantics (NaN equals NaN and sorts last, nulls
  first ascending / last descending unless told otherwise);
- window ranking functions over partitioned, ordered rows;
- the small SQL predicate grammar ``DataFrame.filter("rn = 1")`` accepts.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from decimal import Decimal
from functools import cmp_to_key
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast.evaluator import EvalContext, eval_row
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    CaseWhen,
    Cast,
    Expression,
    FunctionCall,
    Literal,
    SortOrder,
    Star,
    UnaryExpression,
    WindowExpression,
)
from sparkleframe.python.column_helpers import cast_value, spark_order
from sparkleframe.python.types import (
    ArrayType,
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
    NullType,
    Row,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    parse_ddl_type,
)

# --------------------------------------------------------------------------- #
# Rows and schemas
# --------------------------------------------------------------------------- #


def field_names(schema: Any) -> List[str]:
    """Column names of a ``StructType`` schema."""
    return [field.name for field in schema.fields]


def rows_as_dicts(rows: List[tuple], schema: Any) -> List[dict]:
    """Pair each stored row tuple with the schema's column names → list of dicts (structs as dicts)."""
    names = field_names(schema)
    return [{name: _plain(value) for name, value in zip(names, row)} for row in rows]


def _plain(value: Any) -> Any:
    if isinstance(value, Row):
        return value.asDict(recursive=True) if hasattr(value, "__fields__") else [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def normalize_rows(data: Sequence[Any], schema: StructType) -> List[tuple]:
    """``createDataFrame`` input rows (tuples, lists, dicts, ``Row``s) as tuples in schema order."""
    names = field_names(schema)
    rows = []
    for row in data:
        if isinstance(row, Row) and hasattr(row, "__fields__"):
            values = [row[name] if name in row.__fields__ else None for name in names]
        elif isinstance(row, dict):
            values = [row.get(name) for name in names]
        elif isinstance(row, (tuple, list)):
            values = list(row)
        else:
            values = [row]
        if len(values) != len(names):
            raise ValueError(
                f"[LENGTH_SHOULD_BE_THE_SAME] obj and fields should be of the same length, got {len(values)} and "
                f"{len(names)}."
            )
        rows.append(tuple(_to_struct_values(v, f.dataType) for v, f in zip(values, schema.fields)))
    return rows


def _to_struct_values(value: Any, dtype: Any) -> Any:
    """Turn nested dicts/tuples given for struct columns into ``Row`` values; rescale decimals to the field."""
    if value is None:
        return None
    if isinstance(dtype, StructType):
        if isinstance(value, Row) and hasattr(value, "__fields__"):
            items = [value[f.name] if f.name in value.__fields__ else None for f in dtype.fields]
        elif isinstance(value, dict):
            items = [value.get(f.name) for f in dtype.fields]
        else:
            items = list(value)
        return Row.from_fields(dtype.names, [_to_struct_values(v, f.dataType) for v, f in zip(items, dtype.fields)])
    if isinstance(dtype, ArrayType):
        return [_to_struct_values(v, dtype.elementType) for v in value]
    if isinstance(dtype, MapType):
        return {k: _to_struct_values(v, dtype.valueType) for k, v in value.items()}
    if isinstance(dtype, StringType):
        # PySpark's ``convert_string``: Python's ``str`` of whatever a string field is given, bools lowercased.
        return str(value).lower() if isinstance(value, bool) else str(value)
    accepted = _ACCEPTED_VALUES.get(type(dtype))
    if accepted is not None and (
        not isinstance(value, accepted) or (isinstance(value, bool) and bool not in accepted)
    ):
        raise TypeError(
            f"[FIELD_DATA_TYPE_UNACCEPTABLE] {dtype!r} can not accept object {value!r} in type {type(value)}."
        )
    if type(dtype) in _INTEGRAL_RANGES:
        low, high = _INTEGRAL_RANGES[type(dtype)]
        if not low <= value <= high:
            raise ValueError(f"[VALUE_OUT_OF_BOUNDS] Value for `obj` must be between {low} and {high}, got {value}.")
    if isinstance(dtype, DecimalType):
        return cast_value(value, dtype, strict=True)  # Spark stores it at the field's scale (HALF_UP)
    if isinstance(dtype, DateType) and isinstance(value, datetime):
        return value.date()
    return value


# PySpark's ``_acceptable_types``: the Python values ``createDataFrame`` takes for each field type.
_ACCEPTED_VALUES: Dict[type, Tuple[type, ...]] = {
    BooleanType: (bool,),
    ByteType: (int,),
    ShortType: (int,),
    IntegerType: (int,),
    LongType: (int,),
    FloatType: (float,),
    DoubleType: (float,),
    DecimalType: (Decimal,),
    BinaryType: (bytes, bytearray),
    DateType: (date,),
    TimestampType: (datetime,),
}
_INTEGRAL_RANGES = {
    ByteType: (-(2**7), 2**7 - 1),
    ShortType: (-(2**15), 2**15 - 1),
    IntegerType: (-(2**31), 2**31 - 1),
    LongType: (-(2**63), 2**63 - 1),
}


def infer_type(value: Any) -> Any:
    """The type Spark's ``createDataFrame`` infers for a Python value (``int`` is ``bigint`` here)."""
    if value is None:
        return NullType()
    if isinstance(value, bool):
        return BooleanType()
    if isinstance(value, int):
        return LongType()
    if isinstance(value, float):
        return DoubleType()
    if isinstance(value, str):
        return StringType()
    if isinstance(value, Decimal):
        return DecimalType(38, 18)
    if isinstance(value, datetime):
        return TimestampType()
    if isinstance(value, date):
        return DateType()
    if isinstance(value, (bytes, bytearray)):
        return BinaryType()
    if isinstance(value, list):
        element = next((infer_type(v) for v in value if v is not None), NullType())
        return ArrayType(element, True)
    if isinstance(value, dict):
        key = next((infer_type(k) for k in value), NullType())
        val = next((infer_type(v) for v in value.values() if v is not None), NullType())
        return MapType(key, val, True)
    not_implemented_yet(f"schema inference for {type(value).__name__}")


def infer_schema(data: Sequence[Any], names: Optional[Sequence[str]]) -> StructType:
    """A schema from rows plus optional column names; each column's type is merged across all rows."""
    if not data:
        raise ValueError("[CANNOT_INFER_EMPTY_SCHEMA] Can not infer schema from an empty dataset.")
    first = data[0]
    if names is None:
        if isinstance(first, Row) and hasattr(first, "__fields__"):
            names = list(first.__fields__)
        elif isinstance(first, dict):
            names = list(first)
        else:
            names = [f"_{i + 1}" for i in range(len(first))]
    fields = []
    for index, name in enumerate(names):
        dtype: Any = NullType()
        for row in data:
            value = row.get(name) if isinstance(row, dict) else row[index]
            dtype = _merge_types(dtype, infer_type(value), name)
        if isinstance(dtype, NullType):
            raise ValueError(
                f"[CANNOT_DETERMINE_TYPE] Some of types cannot be determined after inferring, column `{name}` is "
                f"all null."
            )
        fields.append(StructField(name, dtype, True))
    return StructType(fields)


def _merge_types(left: Any, right: Any, name: str) -> Any:
    """PySpark's ``_merge_type`` for inference: nulls give way, a string absorbs other atomics, nested types
    merge element-wise, else the types must agree."""
    if isinstance(left, NullType) or left == right:
        return right
    if isinstance(right, NullType):
        return left
    complex_types = (ArrayType, MapType, StructType)
    if (isinstance(left, StringType) or isinstance(right, StringType)) and not (
        isinstance(left, complex_types) or isinstance(right, complex_types)
    ):
        return StringType()
    if isinstance(left, ArrayType) and isinstance(right, ArrayType):
        return ArrayType(_merge_types(left.elementType, right.elementType, name), True)
    if isinstance(left, MapType) and isinstance(right, MapType):
        key = _merge_types(left.keyType, right.keyType, name)
        return MapType(key, _merge_types(left.valueType, right.valueType, name), True)
    raise TypeError(
        f'[CANNOT_MERGE_TYPE] Can not merge type "{left.simpleString()}" and "{right.simpleString()}" for column '
        f"`{name}`."
    )


def tabular_input(data: Any, schema: Any) -> Tuple[List[Any], Any]:
    """``createDataFrame`` input as row values plus the schema to build them with.

    A pandas frame follows PySpark's non-Arrow path: ``to_records`` values, with the column names
    as the schema unless one is given. An Arrow table, or any frame exporting the Arrow C stream
    (polars), takes its schema from the Arrow types. Anything else is an iterable of rows.
    """
    if type(data).__module__.split(".")[0] == "pandas":
        return _pandas_rows(data), list(data.columns) if schema is None else schema
    if hasattr(data, "__arrow_c_stream__"):
        import pyarrow as pa

        table = pa.table(data)
        columns = [[_from_arrow_value(v, c.type) for v in c.to_pylist()] for c in table.columns]
        rows = list(zip(*columns)) if columns else [()] * table.num_rows
        if schema is None or isinstance(schema, (list, tuple)):
            names = list(schema) if schema is not None else table.column_names
            schema = StructType([StructField(n, _from_arrow_type(c.type), True) for n, c in zip(names, table.columns)])
        return rows, schema
    return list(data) if data is not None else [], schema


def _pandas_rows(frame: Any) -> List[list]:
    """PySpark's ``_convert_from_pandas``: zoned timestamps to session-local, then numpy records as lists."""
    import numpy as np
    import pandas as pd

    frame = frame.copy()
    for index in range(frame.shape[1]):
        series = frame.iloc[:, index]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            local = datetime.now().astimezone().tzinfo
            frame.isetitem(index, series.dt.tz_convert(local).dt.tz_localize(None))
    records = frame.set_axis([f"col_{i}" for i in range(frame.shape[1])], axis="columns").to_records(index=False)
    # datetime64[ns] values come out of ``tolist`` as ints; microseconds give ``datetime`` (SPARK-22417).
    fixed = np.dtype(
        [
            (n, "datetime64[us]" if records.dtype[n] == np.dtype("datetime64[ns]") else records.dtype[n])
            for n in records.dtype.names
        ]
    )
    return [r.astype(fixed).tolist() for r in records]


def _from_arrow_type(arrow_type: Any) -> Any:
    """PySpark's ``from_arrow_type``: the Spark type of an Arrow column."""
    import pyarrow.types as pat

    if pat.is_struct(arrow_type):
        return StructType([StructField(f.name, _from_arrow_type(f.type), True) for f in arrow_type])
    if pat.is_map(arrow_type):
        return MapType(_from_arrow_type(arrow_type.key_type), _from_arrow_type(arrow_type.item_type), True)
    if pat.is_list(arrow_type) or pat.is_large_list(arrow_type):
        return ArrayType(_from_arrow_type(arrow_type.value_type), True)
    if pat.is_decimal(arrow_type):
        return DecimalType(arrow_type.precision, arrow_type.scale)
    for predicate, spark_type in _arrow_predicates():
        if predicate(arrow_type):
            return spark_type()
    raise TypeError(
        f"[UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION] {arrow_type} is not supported in conversion to Arrow."
    )


def _from_arrow_value(value: Any, arrow_type: Any) -> Any:
    """An Arrow ``to_pylist`` value as a Spark row value: maps as dicts, zoned timestamps session-local."""
    import pyarrow.types as pat

    if value is None:
        return None
    if pat.is_struct(arrow_type):
        return {f.name: _from_arrow_value(value[f.name], f.type) for f in arrow_type}
    if pat.is_map(arrow_type):
        return {k: _from_arrow_value(v, arrow_type.item_type) for k, v in value}
    if pat.is_list(arrow_type) or pat.is_large_list(arrow_type):
        return [_from_arrow_value(v, arrow_type.value_type) for v in value]
    if pat.is_timestamp(arrow_type) and value.tzinfo is not None:
        return value.astimezone().replace(tzinfo=None)
    return value


def _arrow_predicates() -> List[Tuple[Callable[[Any], bool], type]]:
    import pyarrow.types as pat

    return [
        (pat.is_boolean, BooleanType),
        (pat.is_int8, ByteType),
        (pat.is_int16, ShortType),
        (pat.is_int32, IntegerType),
        (pat.is_int64, LongType),
        (pat.is_float32, FloatType),
        (pat.is_float64, DoubleType),
        (lambda t: pat.is_string(t) or pat.is_large_string(t) or pat.is_string_view(t), StringType),
        (lambda t: pat.is_binary(t) or pat.is_large_binary(t), BinaryType),
        (pat.is_date32, DateType),
        (pat.is_timestamp, TimestampType),
        (pat.is_null, NullType),
    ]


def to_arrow_table(rows: List[tuple], schema: StructType) -> Any:
    """PySpark's ``toArrow``: a ``pyarrow.Table`` with ``to_arrow_type`` column types.

    Timestamps are the same instants Spark produces, labelled ``UTC`` (Spark labels them with the
    session time zone; the engine's naive timestamps are process-local).
    """
    import pyarrow as pa

    fields = [pa.field(f.name, _to_arrow_type(f.dataType), f.nullable) for f in schema.fields]
    arrays = [
        pa.array([_to_arrow_value(row[i], f.dataType) for row in rows], type=field.type)
        for i, (f, field) in enumerate(zip(schema.fields, fields))
    ]
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _to_arrow_type(dtype: Any) -> Any:
    import pyarrow as pa

    if isinstance(dtype, StructType):
        return pa.struct([pa.field(f.name, _to_arrow_type(f.dataType), f.nullable) for f in dtype.fields])
    if isinstance(dtype, ArrayType):
        return pa.list_(pa.field("element", _to_arrow_type(dtype.elementType), dtype.containsNull))
    if isinstance(dtype, MapType):
        value = pa.field("value", _to_arrow_type(dtype.valueType), dtype.valueContainsNull)
        return pa.map_(_to_arrow_type(dtype.keyType), value)
    if isinstance(dtype, DecimalType):
        return pa.decimal128(dtype.precision, dtype.scale)
    if isinstance(dtype, TimestampType):
        return pa.timestamp("us", tz="UTC")
    atomic = {
        BooleanType: pa.bool_,
        ByteType: pa.int8,
        ShortType: pa.int16,
        IntegerType: pa.int32,
        LongType: pa.int64,
        FloatType: pa.float32,
        DoubleType: pa.float64,
        StringType: pa.string,
        BinaryType: pa.binary,
        DateType: pa.date32,
        NullType: pa.null,
    }.get(type(dtype))
    if atomic is None:
        raise TypeError(f"[UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION] {dtype.simpleString()} is not supported.")
    return atomic()


def _to_arrow_value(value: Any, dtype: Any) -> Any:
    if value is None:
        return None
    if isinstance(dtype, StructType):
        return {f.name: _to_arrow_value(v, f.dataType) for f, v in zip(dtype.fields, value)}
    if isinstance(dtype, ArrayType):
        return [_to_arrow_value(v, dtype.elementType) for v in value]
    if isinstance(dtype, MapType):
        return [(_to_arrow_value(k, dtype.keyType), _to_arrow_value(v, dtype.valueType)) for k, v in value.items()]
    if isinstance(dtype, TimestampType):
        return value.astimezone()  # session-local wall clock -> the instant
    return value


# --------------------------------------------------------------------------- #
# Output names
# --------------------------------------------------------------------------- #

_SQL_OPS = {"==": "=", "and": "AND", "or": "OR", "<=>": "<=>"}


def output_name(expr: Expression) -> str:
    """Derive the output column name of an expression the way Spark auto-names it.

    ``Alias`` and ``AttributeReference`` carry an explicit name; other nodes get Spark's generated
    label, e.g. ``(a + b)``, ``max(x)``, ``count(1)``.
    """
    if isinstance(expr, Alias):
        return expr.name
    if isinstance(expr, AttributeReference):
        return expr.name.split(".")[-1] if "`" not in expr.name else expr.name.strip("`")
    if isinstance(expr, Literal):
        if expr.value is None:
            return "NULL"
        if isinstance(expr.value, bool):
            return "true" if expr.value else "false"
        return str(expr.value)
    if isinstance(expr, BinaryExpression):
        if expr.op == "!=":
            return f"(NOT ({output_name(expr.left)} = {output_name(expr.right)}))"
        return f"({output_name(expr.left)} {_SQL_OPS.get(expr.op, expr.op)} {output_name(expr.right)})"
    if isinstance(expr, UnaryExpression):
        if expr.op == "not":
            return f"(NOT {output_name(expr.child)})"
        if expr.op == "neg":
            return f"(- {output_name(expr.child)})"
        if expr.op in ("isnull", "isnotnull"):
            return f"({output_name(expr.child)} IS {'NOT ' if expr.op == 'isnotnull' else ''}NULL)"
        return output_name(expr.child)
    if isinstance(expr, Cast):
        target = expr.target_type if isinstance(expr.target_type, str) else expr.target_type.simpleString()
        return f"{'TRY_CAST' if not expr.strict else 'CAST'}({output_name(expr.child)} AS {target.upper()})"
    if isinstance(expr, FunctionCall):
        if expr.name == "get_field":
            return f"{output_name(expr.children[0])}.{expr.options['name']}"
        if expr.name == "get_item":
            return f"{output_name(expr.children[0])}[{output_name(expr.children[1])}]"
        args = ["1" if isinstance(a, Star) else output_name(a) for a in expr.children if isinstance(a, Expression)]
        args += [str(v) for k, v in expr.options.items() if k in ("pattern", "format", "path", "scale")]
        return f"{expr.name}({', '.join(args)})"
    if isinstance(expr, CaseWhen):
        parts = " ".join(f"WHEN {output_name(c)} THEN {output_name(v)}" for c, v in expr.branches)
        tail = f" ELSE {output_name(expr.otherwise)}" if expr.otherwise is not None else ""
        return f"CASE {parts}{tail} END"
    if isinstance(expr, WindowExpression):
        return f"{output_name(expr.function)} OVER (...)"
    return repr(expr)


# --------------------------------------------------------------------------- #
# Grouping keys and ordering
# --------------------------------------------------------------------------- #


def group_key(values: Sequence[Any]) -> tuple:
    """A hashable key treating NaN as equal to NaN and -0.0 as 0.0, like Spark's grouping."""
    return tuple(_key_value(v) for v in values)


def _key_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ("NaN",)
        return 0.0 if value == 0 else value
    if isinstance(value, list):
        return ("list", tuple(_key_value(v) for v in value))
    if isinstance(value, dict):
        return ("map", tuple((_key_value(k), _key_value(v)) for k, v in value.items()))
    if isinstance(value, tuple):
        return ("struct", tuple(_key_value(v) for v in value))
    return value


def sort_comparator(orders: List[SortOrder]) -> Callable[[tuple, tuple], int]:
    """Compare two precomputed sort-key tuples with each order's direction and null placement."""

    def compare(left: tuple, right: tuple) -> int:
        for order, lv, rv in zip(orders, left, right):
            if lv is None or rv is None:
                if lv is None and rv is None:
                    continue
                null_cmp = -1 if lv is None else 1
                return null_cmp if order.nulls_first else -null_cmp
            result = spark_order(lv, rv, order.data_type)
            if result:
                return result if order.ascending else -result
        return 0

    return compare


def sort_rows(rows: List[tuple], orders: List[SortOrder], ctx: EvalContext) -> List[tuple]:
    """Stable sort of ``rows`` by resolved ``orders``."""
    keyed = [(tuple(eval_row(o.child, row, ctx) for o in orders), row) for row in rows]
    compare = sort_comparator(orders)
    keyed.sort(key=cmp_to_key(lambda a, b: compare(a[0], b[0])))
    return [row for _, row in keyed]


def window_values(window: WindowExpression, rows: List[tuple], ctx: EvalContext) -> List[Any]:
    """Compute a ranking window function (``row_number`` / ``rank`` / ``dense_rank``) for every row."""
    partitions: Dict[tuple, List[int]] = {}
    for index, row in enumerate(rows):
        key = group_key([eval_row(p, row, ctx) for p in window.partition_by])
        partitions.setdefault(key, []).append(index)
    order_keys = [tuple(eval_row(o.child, row, ctx) for o in window.order_by) for row in rows]
    compare = sort_comparator(window.order_by)
    result: List[Any] = [None] * len(rows)
    name = window.function.name
    for indices in partitions.values():
        ordered = sorted(indices, key=cmp_to_key(lambda a, b: compare(order_keys[a], order_keys[b])))
        rank = dense = 0
        previous: Optional[tuple] = None
        for position, index in enumerate(ordered, start=1):
            if previous is None or compare(previous, order_keys[index]) != 0:
                rank = position
                dense += 1
                previous = order_keys[index]
            result[index] = position if name == "row_number" else rank if name == "rank" else dense
    return result


def find_nodes(expr: Expression, predicate: Callable[[Expression], bool], stop_at: Tuple[type, ...] = ()) -> List:
    """All nodes in ``expr`` matching ``predicate`` (not descending into matches or ``stop_at`` types)."""
    found = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if predicate(node):
            found.append(node)
            continue
        if isinstance(node, stop_at):
            continue
        stack.extend(c for c in node.children if isinstance(c, Expression))
    return found


# --------------------------------------------------------------------------- #
# SQL predicate strings for filter("...")
# --------------------------------------------------------------------------- #

_SQL_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?i:BD|[LSYD])?(?![A-Za-z0-9_]))"
    r"|(?P<str>'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\")"
    r"|(?P<ident>`[^`]+`|[A-Za-z_][A-Za-z0-9_.]*)"
    r"|(?P<op><=>|<=|>=|<>|!=|==|\|\||[=<>()+\-*/,]))"
)
_COMPARISONS = {
    "=": "==",
    "==": "==",
    "<>": "!=",
    "!=": "!=",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "<=>": "<=>",
}
# Spark's ``unescapeSQLString`` single-character escapes; ``\%`` and ``\_`` keep their backslash for LIKE.
_SQL_ESCAPES = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a", "%": "\\%", "_": "\\_"}
_TYPED_LITERALS = {"DATE": DateType, "TIMESTAMP": TimestampType}


def parse_sql_predicate(text: str) -> Expression:
    """Parse a ``filter("...")`` SQL predicate with Spark's precedence, loosest first.

    ``OR``; ``AND``; ``NOT``; comparisons, ``IS [NOT] NULL|TRUE|FALSE|UNKNOWN|DISTINCT FROM``,
    ``[NOT] IN (...)``, ``[NOT] BETWEEN``, ``[NOT] LIKE|RLIKE|REGEXP``; ``+ - ||``; ``* /``; unary sign;
    then literals (typed numbers, escaped strings, ``DATE '...'``), ``CAST(x AS type)``, function
    calls and column names.
    """
    tokens: List[Tuple[str, str, int, int]] = []
    pos = 0
    while pos < len(text):
        match = _SQL_TOKEN_RE.match(text, pos)
        if match is None or match.end() == pos:
            if text[pos:].strip() == "":
                break
            not_implemented_yet(f"SQL predicate syntax in filter({text!r})")
        kind = match.lastgroup
        tokens.append((kind, match.group(kind), match.start(kind), match.end(kind)))
        pos = match.end()
    parser = _PredicateParser(tokens, text)
    expr = parser.disjunction()
    if parser.pos != len(tokens):
        parser.unsupported()
    return expr


def _sql_string(token: str) -> str:
    """A quoted SQL string literal's value, unescaped the way Spark's ``unescapeSQLString`` does (a doubled
    quote is one quote)."""
    quote = token[0]
    body, out, index = token[1:-1].replace(quote * 2, quote), [], 0
    while index < len(body):
        char = body[index]
        if char != "\\" or index + 1 == len(body):
            out.append(char)
            index += 1
            continue
        following = body[index + 1]
        if following == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", body[index + 2 : index + 6]):
            out.append(chr(int(body[index + 2 : index + 6], 16)))
            index += 6
        elif re.fullmatch(r"[0-3][0-7]{2}", body[index + 1 : index + 4]):
            out.append(chr(int(body[index + 1 : index + 4], 8)))
            index += 4
        else:
            out.append(_SQL_ESCAPES.get(following, following))
            index += 2
    return "".join(out)


def _sql_number(text: str, negative: bool) -> Literal:
    """A numeric literal typed as Spark types it: ``1`` int (bigint, then decimal when wider), ``1.5`` decimal,
    ``1e3`` double, and the ``L``/``S``/``Y``/``D``/``BD`` suffixes."""
    sign = "-" if negative else ""
    upper = text.upper()
    if upper.endswith("BD"):
        return Literal(Decimal(sign + text[:-2]))
    suffix = upper[-1] if upper[-1] in "LSYD" else ""
    digits = sign + (text[:-1] if suffix else text)
    if suffix == "D" or (not suffix and "E" in upper):
        return Literal(float(digits))
    if suffix:
        dtype = {"L": LongType, "S": ShortType, "Y": ByteType}[suffix]()
        value = int(digits)
        low, high = _INTEGRAL_RANGES[type(dtype)]
        if not low <= value <= high:
            raise ValueError(
                f"[INVALID_NUMERIC_LITERAL_RANGE] Numeric literal {digits} is outside the range of {dtype!r}."
            )
        return Literal(value, dtype)
    if "." in digits:
        return Literal(Decimal(digits))
    value = int(digits)
    return Literal(value) if -(2**63) <= value < 2**63 else Literal(Decimal(value))


class _PredicateParser:
    def __init__(self, tokens: List[Tuple[str, str, int, int]], text: str) -> None:
        self.tokens = tokens
        self.text = text
        self.pos = 0

    def unsupported(self) -> Any:
        not_implemented_yet(f"SQL predicate syntax in filter({self.text!r})")

    def _peek(self, kind: str, offset: int = 0) -> Optional[str]:
        index = self.pos + offset
        if index < len(self.tokens) and self.tokens[index][0] == kind:
            value = self.tokens[index][1]
            return value.upper() if kind == "ident" else value
        return None

    def _accept_word(self, *words: str) -> Optional[str]:
        word = self._peek("ident")
        if word in words:
            self.pos += 1
            return word
        return None

    def _accept_op(self, *ops: str) -> Optional[str]:
        op = self._peek("op")
        if op in ops:
            self.pos += 1
            return op
        return None

    def _expect_op(self, op: str) -> None:
        if self._accept_op(op) is None:
            self.unsupported()

    def disjunction(self) -> Expression:
        expr = self.conjunction()
        while self._accept_word("OR"):
            expr = BinaryExpression("or", expr, self.conjunction())
        return expr

    def conjunction(self) -> Expression:
        expr = self.negation()
        while self._accept_word("AND"):
            expr = BinaryExpression("and", expr, self.negation())
        return expr

    def negation(self) -> Expression:
        if self._accept_word("NOT"):
            return UnaryExpression("not", self.negation())
        return self.predicate()

    def predicate(self) -> Expression:
        left = self.additive()
        op = self._accept_op(*_COMPARISONS)
        if op is not None:
            return BinaryExpression(_COMPARISONS[op], left, self.additive())
        if self._accept_word("IS"):
            return self._is_predicate(left, negate=self._accept_word("NOT") is not None)
        negate = self._peek("ident") == "NOT" and self._peek("ident", 1) in (
            "IN",
            "BETWEEN",
            "LIKE",
            "RLIKE",
            "REGEXP",
        )
        if negate:
            self.pos += 1
        expr = self._keyword_predicate(left)
        if expr is None:
            return self.unsupported() if negate else left
        return UnaryExpression("not", expr) if negate else expr

    def _is_predicate(self, left: Expression, negate: bool) -> Expression:
        if self._accept_word("NULL"):
            return UnaryExpression("isnotnull" if negate else "isnull", left)
        if self._accept_word("DISTINCT"):
            if not self._accept_word("FROM"):
                self.unsupported()
            same = BinaryExpression("<=>", left, self.additive())
            return same if negate else UnaryExpression("not", same)
        word = self._accept_word("TRUE", "FALSE", "UNKNOWN")
        if word is None:
            self.unsupported()
        expr = (
            UnaryExpression("isnull", left)
            if word == "UNKNOWN"
            else BinaryExpression("<=>", left, Literal(word == "TRUE"))
        )
        return UnaryExpression("not", expr) if negate else expr

    def _keyword_predicate(self, left: Expression) -> Optional[Expression]:
        if self._accept_word("IN"):
            self._expect_op("(")
            items = [self.disjunction()]
            while self._accept_op(","):
                items.append(self.disjunction())
            self._expect_op(")")
            return FunctionCall("in", [left, *items])
        if self._accept_word("BETWEEN"):
            low = self.additive()
            if not self._accept_word("AND"):
                self.unsupported()
            return BinaryExpression(
                "and", BinaryExpression(">=", left, low), BinaryExpression("<=", left, self.additive())
            )
        word = self._accept_word("LIKE", "RLIKE", "REGEXP")
        if word is not None:
            pattern = self._peek("str")
            if pattern is None:
                self.unsupported()
            self.pos += 1
            return FunctionCall("like" if word == "LIKE" else "rlike", [left], {"pattern": _sql_string(pattern)})
        return None

    def additive(self) -> Expression:
        expr = self.multiplicative()
        while True:
            op = self._accept_op("+", "-", "||")
            if op is None:
                return expr
            right = self.multiplicative()
            expr = FunctionCall("concat", [expr, right]) if op == "||" else BinaryExpression(op, expr, right)

    def multiplicative(self) -> Expression:
        expr = self.unary()
        while True:
            op = self._accept_op("*", "/")
            if op is None:
                return expr
            expr = BinaryExpression(op, expr, self.unary())

    def unary(self) -> Expression:
        op = self._accept_op("-", "+")
        if op is None:
            return self.primary()
        number = self._peek("num")
        if number is not None:  # ``-1`` is a negative literal, not a negation
            self.pos += 1
            return _sql_number(number, negative=op == "-")
        child = self.unary()
        return UnaryExpression("neg", child) if op == "-" else UnaryExpression("pos", child)

    def primary(self) -> Expression:
        if self.pos >= len(self.tokens):
            return self.unsupported()
        kind, value, _, _ = self.tokens[self.pos]
        self.pos += 1
        if kind == "num":
            return _sql_number(value, negative=False)
        if kind == "str":
            text = _sql_string(value)
            while self._peek("str") is not None:  # adjacent literals concatenate: 'a' 'b' is 'ab'
                text += _sql_string(self.tokens[self.pos][1])
                self.pos += 1
            return Literal(text)
        if kind == "op" and value == "(":
            expr = self.disjunction()
            self._expect_op(")")
            return expr
        if kind != "ident":
            return self.unsupported()
        upper = value.upper()
        if upper in ("TRUE", "FALSE"):
            return Literal(upper == "TRUE")
        if upper == "NULL":
            return Literal(None)
        if upper in _TYPED_LITERALS and self._peek("str") is not None:
            literal = Literal(_sql_string(self.tokens[self.pos][1]))
            self.pos += 1
            return Cast(literal, _TYPED_LITERALS[upper](), strict=True)
        if self._peek("op") == "(":
            self.pos += 1
            return self._cast() if upper == "CAST" else self._call(value)
        return AttributeReference(value)

    def _cast(self) -> Expression:
        child = self.disjunction()
        if not self._accept_word("AS"):
            self.unsupported()
        start, depth = self.pos, 0
        while self.pos < len(self.tokens) and (depth or self.tokens[self.pos][1] != ")"):
            depth += {"(": 1, ")": -1}.get(self.tokens[self.pos][1], 0)
            self.pos += 1
        if self.pos == start or self.pos == len(self.tokens):
            self.unsupported()
        type_text = self.text[self.tokens[start][2] : self.tokens[self.pos - 1][3]]
        self.pos += 1
        return Cast(child, parse_ddl_type(type_text), strict=True)

    def _call(self, name: str) -> Expression:
        import sparkleframe.python.functions as F
        from sparkleframe.python.column import Column

        args: List[Expression] = []
        if self._accept_op(")") is None:
            args.append(self.disjunction())
            while self._accept_op(","):
                args.append(self.disjunction())
            self._expect_op(")")
        function = getattr(F, name.lower(), None)
        if name.startswith("_") or not callable(function):
            not_implemented_yet(f"function {name}() in filter({self.text!r})")
        return function(*(Column(arg) for arg in args))._expr
