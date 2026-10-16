"""``pyspark.sql.functions`` surface for the pure-Python engine.

Each function is part of the **build** phase: it constructs an unresolved AST node (see
:mod:`sparkleframe.python.ast.expressions`) and makes no type decisions. Type resolution and
execution for every ``FunctionCall`` name live in
:mod:`sparkleframe.python.ast.function_registry`.

Argument conventions follow PySpark: a ``str`` passed where a column is expected names a column
(``F.lower("name")``), while non-column parameters — regex patterns, date formats, ``from_json``
schemas — are carried as node ``options`` rather than as columns.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from sparkleframe.python.ast.expressions import (
    AttributeReference,
    BinaryExpression,
    CaseWhen,
    FunctionCall,
    LambdaFunction,
    Literal,
    SortOrder,
    Star,
)
from sparkleframe.python.column import Column, _column_ref, _to_expression

ColumnOrName = Union[str, Column]


def _fn(name: str, *cols: Any, **options: Any) -> Column:
    """Build a :class:`Column` wrapping ``FunctionCall(name, [...])`` over column arguments."""
    return Column(FunctionCall(name, [_column_ref(c) for c in cols], options))


def _varargs(cols: tuple) -> list:
    """PySpark accepts ``f(a, b)`` and ``f([a, b])`` interchangeably for variadic functions."""
    if len(cols) == 1 and isinstance(cols[0], (list, tuple, set)):
        return list(cols[0])
    return list(cols)


def _literal_value(value: Any) -> Any:
    """A literal parameter given either as a Python value or as ``lit(value)``."""
    if isinstance(value, Column) and isinstance(value._expr, Literal):
        return value._expr.value
    return value


# --- Column builders ---------------------------------------------------------
def col(name: str) -> Column:
    if name == "*":
        return Column(Star())
    return Column(AttributeReference(name))


column = col


def lit(value: Any) -> Column:
    if isinstance(value, Column):
        return value
    return Column(Literal(value))


def when(condition: Any, value: Any) -> Column:
    """Start a multi-branch conditional; chain ``.when(...).otherwise(...)``."""
    return Column(CaseWhen([(_to_expression(condition), _to_expression(value))]))


def broadcast(df: Any) -> Any:
    """A join hint only — the engine has a single local plan, so the frame is returned unchanged."""
    return df


# --- Math ----------------------------------------------------------------------
def abs(col_name: ColumnOrName) -> Column:
    return _fn("abs", col_name)


def floor(col_name: ColumnOrName) -> Column:
    return _fn("floor", col_name)


def round(col_name: ColumnOrName, scale: int = 0) -> Column:
    return _fn("round", col_name, scale=_literal_value(scale))


def pow(base: Any, exponent: Any) -> Column:
    """``base ** exponent`` as a double; a ``str`` names a column, any other scalar is a literal.

    Builds the same ``**`` node as :meth:`Column.__pow__` (PySpark's ``Column.__pow__`` *is*
    ``pow``), so both share one analyze/evaluate path.
    """
    return Column(BinaryExpression("**", _column_ref(base), _column_ref(exponent)))


def isnan(col_name: ColumnOrName) -> Column:
    return _fn("isnan", col_name)


def try_divide(left: ColumnOrName, right: ColumnOrName) -> Column:
    return _fn("try_divide", left, right)


def greatest(*cols: ColumnOrName) -> Column:
    return _fn("greatest", *cols)


def least(*cols: ColumnOrName) -> Column:
    return _fn("least", *cols)


def coalesce(*cols: ColumnOrName) -> Column:
    return _fn("coalesce", *cols)


def nullif(e1: ColumnOrName, e2: ColumnOrName) -> Column:
    return _fn("nullif", e1, e2)


def isnull(col_name: ColumnOrName) -> Column:
    return _column(col_name).isNull()


# --- Strings ---------------------------------------------------------------------
def lower(col_name: ColumnOrName) -> Column:
    return _fn("lower", col_name)


def upper(col_name: ColumnOrName) -> Column:
    return _fn("upper", col_name)


def trim(col_name: ColumnOrName) -> Column:
    return _fn("trim", col_name)


def initcap(col_name: ColumnOrName) -> Column:
    return _fn("initcap", col_name)


def md5(col_name: ColumnOrName) -> Column:
    return _fn("md5", col_name)


def length(col_name: ColumnOrName) -> Column:
    return _fn("length", col_name)


def concat(*cols: ColumnOrName) -> Column:
    return _fn("concat", *cols)


def concat_ws(sep: str, *cols: ColumnOrName) -> Column:
    return Column(FunctionCall("concat_ws", [Literal(sep), *(_column_ref(c) for c in _varargs(cols))]))


def substring(col_name: ColumnOrName, pos: Any, len: Any) -> Column:
    return Column(FunctionCall("substring", [_column_ref(col_name), _to_expression(pos), _to_expression(len)]))


def split(col_name: ColumnOrName, pattern: str, limit: int = -1) -> Column:
    return _fn("split", col_name, pattern=_literal_value(pattern), limit=_literal_value(limit))


def regexp_replace(col_name: ColumnOrName, pattern: Any, replacement: Any) -> Column:
    return _fn("regexp_replace", col_name, pattern=_literal_value(pattern), replacement=_literal_value(replacement))


def regexp_extract(col_name: ColumnOrName, pattern: str, idx: int) -> Column:
    return _fn("regexp_extract", col_name, pattern=pattern, idx=idx)


# --- Dates and time ----------------------------------------------------------------
def current_date() -> Column:
    return _fn("current_date")


def current_timestamp() -> Column:
    return _fn("current_timestamp")


def now() -> Column:
    return _fn("current_timestamp")


def date_sub(start: ColumnOrName, days: Any) -> Column:
    return Column(FunctionCall("date_sub", [_column_ref(start), _to_expression(days)]))


def date_add(start: ColumnOrName, days: Any) -> Column:
    return Column(FunctionCall("date_add", [_column_ref(start), _to_expression(days)]))


def add_months(start: ColumnOrName, months: Any) -> Column:
    return Column(FunctionCall("add_months", [_column_ref(start), _to_expression(months)]))


def datediff(end: ColumnOrName, start: ColumnOrName) -> Column:
    return _fn("datediff", end, start)


def months_between(date1: ColumnOrName, date2: ColumnOrName, roundOff: bool = True) -> Column:
    return _fn("months_between", date1, date2, roundOff=roundOff)


def date_format(date: ColumnOrName, format: str) -> Column:
    return _fn("date_format", date, format=format)


def trunc(date: ColumnOrName, format: str) -> Column:
    return _fn("trunc", date, format=format)


def to_date(col_name: ColumnOrName, format: Optional[Any] = None) -> Column:
    return _fn("to_date", col_name, format=_literal_value(format))


def try_to_date(col_name: ColumnOrName, format: Optional[Any] = None) -> Column:
    return _fn("try_to_date", col_name, format=_literal_value(format))


def to_timestamp(col_name: ColumnOrName, format: Optional[Any] = None) -> Column:
    return _fn("to_timestamp", col_name, format=_literal_value(format))


def try_to_timestamp(col_name: ColumnOrName, format: Optional[Any] = None) -> Column:
    return _fn("try_to_timestamp", col_name, format=_literal_value(format))


# --- Complex types -------------------------------------------------------------------
def struct(*cols: ColumnOrName) -> Column:
    return _fn("struct", *_varargs(cols))


def array(*cols: ColumnOrName) -> Column:
    return _fn("array", *_varargs(cols))


def create_map(*cols: ColumnOrName) -> Column:
    return _fn("create_map", *_varargs(cols))


def map_from_entries(col_name: ColumnOrName) -> Column:
    return _fn("map_from_entries", col_name)


def map_keys(col_name: ColumnOrName) -> Column:
    return _fn("map_keys", col_name)


def map_values(col_name: ColumnOrName) -> Column:
    return _fn("map_values", col_name)


def map_entries(col_name: ColumnOrName) -> Column:
    return _fn("map_entries", col_name)


def element_at(col_name: ColumnOrName, extraction: Any) -> Column:
    """1-based array index / map key; ``extraction`` is a literal unless it is a Column (PySpark's ``lit``)."""
    return Column(FunctionCall("element_at", [_column_ref(col_name), _to_expression(extraction)]))


def try_element_at(col_name: ColumnOrName, extraction: ColumnOrName) -> Column:
    """Like ``element_at`` but null on a bad index; PySpark treats a ``str`` extraction as a column name."""
    return _fn("try_element_at", col_name, extraction)


def array_contains(col_name: ColumnOrName, value: Any) -> Column:
    return Column(FunctionCall("array_contains", [_column_ref(col_name), _to_expression(value)]))


def size(col_name: ColumnOrName) -> Column:
    return _fn("size", col_name)


def sort_array(col_name: ColumnOrName, asc: bool = True) -> Column:
    return _fn("sort_array", col_name, asc=asc)


def array_max(col_name: ColumnOrName) -> Column:
    return _fn("array_max", col_name)


def explode(col_name: ColumnOrName) -> Column:
    return _fn("explode", col_name)


def transform(col_name: ColumnOrName, f: Callable[..., Any]) -> Column:
    return Column(FunctionCall("transform", [_column_ref(col_name), LambdaFunction(f)]))


def filter(col_name: ColumnOrName, f: Callable[..., Any]) -> Column:
    return Column(FunctionCall("filter", [_column_ref(col_name), LambdaFunction(f)]))


# --- JSON ----------------------------------------------------------------------------
def get_json_object(col: ColumnOrName, path: str) -> Column:
    return _fn("get_json_object", col, path=path)


def from_json(col_name: ColumnOrName, schema: Any, options: Optional[dict] = None) -> Column:
    return _fn("from_json", col_name, schema=_literal_value(schema), options=options or {})


def to_json(col_name: ColumnOrName, options: Optional[dict] = None) -> Column:
    return _fn("to_json", col_name, options=options or {})


# --- Aggregates ----------------------------------------------------------------------
def count(col_name: ColumnOrName) -> Column:
    if isinstance(col_name, str) and col_name == "*":
        return Column(FunctionCall("count", [Star()]))
    return _fn("count", col_name)


def sum(col_name: ColumnOrName) -> Column:
    return _fn("sum", col_name)


def avg(col_name: ColumnOrName) -> Column:
    return _fn("avg", col_name)


mean = avg


def min(col_name: ColumnOrName) -> Column:
    return _fn("min", col_name)


def max(col_name: ColumnOrName) -> Column:
    return _fn("max", col_name)


def first(col_name: ColumnOrName, ignorenulls: bool = False) -> Column:
    return _fn("first", col_name, ignorenulls=ignorenulls)


def collect_list(col_name: ColumnOrName) -> Column:
    return _fn("collect_list", col_name)


def collect_set(col_name: ColumnOrName) -> Column:
    return _fn("collect_set", col_name)


# --- Window ranking --------------------------------------------------------------------
def row_number() -> Column:
    return _fn("row_number")


def rank() -> Column:
    return _fn("rank")


def dense_rank() -> Column:
    return _fn("dense_rank")


# --- Sort orders -------------------------------------------------------------------------
def _column(value: ColumnOrName) -> Column:
    return value if isinstance(value, Column) else Column(_column_ref(value))


def _sort(value: ColumnOrName, ascending: bool, nulls_first: bool) -> Column:
    return Column(SortOrder(_column(value)._expr, ascending, nulls_first))


def asc(column: ColumnOrName) -> Column:
    return _sort(column, True, True)


def asc_nulls_first(column: ColumnOrName) -> Column:
    return _sort(column, True, True)


def asc_nulls_last(column: ColumnOrName) -> Column:
    return _sort(column, True, False)


def desc(column: ColumnOrName) -> Column:
    return _sort(column, False, False)


def desc_nulls_first(column: ColumnOrName) -> Column:
    return _sort(column, False, True)


def desc_nulls_last(column: ColumnOrName) -> Column:
    return _sort(column, False, False)


# --- Non-deterministic -----------------------------------------------------------------------
def rand(seed: Optional[int] = None) -> Column:
    return _fn("rand", seed=seed)


def uuid() -> Column:
    return _fn("uuid")


def monotonically_increasing_id() -> Column:
    return _fn("monotonically_increasing_id")
