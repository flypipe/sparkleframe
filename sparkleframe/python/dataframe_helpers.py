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
from sparkleframe.python.column_helpers import cast_value, spark_order, to_spark_string
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
        # Spark 4 stringifies whatever a string field is given (``True`` is ``"true"``).
        return value if isinstance(value, str) else to_spark_string(value, infer_type(value))
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
    first = data[0] if data else None
    if names is None:
        if isinstance(first, Row) and hasattr(first, "__fields__"):
            names = list(first.__fields__)
        elif isinstance(first, dict):
            names = list(first)
        else:
            names = [f"_{i + 1}" for i in range(len(first) if first is not None else 0)]
    fields = []
    for index, name in enumerate(names):
        dtype: Any = NullType()
        for row in data:
            value = row.get(name) if isinstance(row, dict) else row[index]
            dtype = _merge_types(dtype, infer_type(value), name)
        if isinstance(dtype, NullType) and data:
            raise ValueError(
                f"[CANNOT_DETERMINE_TYPE] Some of types cannot be determined after inferring, column `{name}` is "
                f"all null."
            )
        fields.append(StructField(name, dtype if not isinstance(dtype, NullType) else StringType(), True))
    return StructType(fields)


def _merge_types(left: Any, right: Any, name: str) -> Any:
    """PySpark's ``_merge_type`` for inference: nulls give way, a string absorbs other atomics, else it must agree."""
    if isinstance(left, NullType) or left == right:
        return right
    if isinstance(right, NullType):
        return left
    complex_types = (ArrayType, MapType, StructType)
    if (isinstance(left, StringType) or isinstance(right, StringType)) and not (
        isinstance(left, complex_types) or isinstance(right, complex_types)
    ):
        return StringType()
    raise TypeError(
        f'[CANNOT_MERGE_TYPE] Can not merge type "{left.simpleString()}" and "{right.simpleString()}" for column '
        f"`{name}`."
    )


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
    r"\s*(?:(?P<num>\d+\.\d*|\.\d+|\d+)|(?P<str>'(?:[^']|'')*')|(?P<ident>`[^`]+`|[A-Za-z_][A-Za-z0-9_.]*)"
    r"|(?P<op><=>|<=|>=|<>|!=|==|=|<|>|\(|\)))"
)


def parse_sql_predicate(text: str) -> Expression:
    """Parse the predicate grammar ``filter("...")`` supports: comparisons, IS [NOT] NULL, AND/OR/NOT."""
    tokens: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _SQL_TOKEN_RE.match(text, pos)
        if match is None or match.end() == pos:
            if text[pos:].strip() == "":
                break
            not_implemented_yet(f"SQL predicate syntax in filter({text!r})")
        kind = match.lastgroup
        tokens.append((kind, match.group(kind)))
        pos = match.end()
    parser = _PredicateParser(tokens, text)
    expr = parser.disjunction()
    if parser.pos != len(tokens):
        not_implemented_yet(f"SQL predicate syntax in filter({text!r})")
    return expr


class _PredicateParser:
    def __init__(self, tokens: List[Tuple[str, str]], text: str) -> None:
        self.tokens = tokens
        self.text = text
        self.pos = 0

    def _peek_word(self) -> Optional[str]:
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] == "ident":
            return self.tokens[self.pos][1].upper()
        return None

    def _next(self) -> Tuple[str, str]:
        if self.pos >= len(self.tokens):
            not_implemented_yet(f"SQL predicate syntax in filter({self.text!r})")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def disjunction(self) -> Expression:
        expr = self.conjunction()
        while self._peek_word() == "OR":
            self.pos += 1
            expr = BinaryExpression("or", expr, self.conjunction())
        return expr

    def conjunction(self) -> Expression:
        expr = self.negation()
        while self._peek_word() == "AND":
            self.pos += 1
            expr = BinaryExpression("and", expr, self.negation())
        return expr

    def negation(self) -> Expression:
        if self._peek_word() == "NOT":
            self.pos += 1
            return UnaryExpression("not", self.negation())
        return self.comparison()

    def comparison(self) -> Expression:
        left = self.operand()
        if self._peek_word() == "IS":
            self.pos += 1
            negate = self._peek_word() == "NOT"
            if negate:
                self.pos += 1
            if self._peek_word() != "NULL":
                not_implemented_yet(f"SQL predicate syntax in filter({self.text!r})")
            self.pos += 1
            return UnaryExpression("isnotnull" if negate else "isnull", left)
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] == "op" and self.tokens[self.pos][1] not in "()":
            op = self._next()[1]
            op = {"=": "==", "<>": "!="}.get(op, op)
            return BinaryExpression(op, left, self.operand())
        return left

    def operand(self) -> Expression:
        kind, value = self._next()
        if kind == "num":
            return Literal(Decimal(value) if "." in value else int(value))
        if kind == "str":
            return Literal(value[1:-1].replace("''", "'"))
        if kind == "op" and value == "(":
            expr = self.disjunction()
            if self._next() != ("op", ")"):
                not_implemented_yet(f"SQL predicate syntax in filter({self.text!r})")
            return expr
        if kind == "ident":
            upper = value.upper()
            if upper in ("TRUE", "FALSE"):
                return Literal(upper == "TRUE")
            if upper == "NULL":
                return Literal(None)
            return AttributeReference(value)
        not_implemented_yet(f"SQL predicate syntax in filter({self.text!r})")
