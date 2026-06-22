"""``pyspark.sql.functions`` surface for the pure-Python engine.

Each function is part of the **build** phase: it constructs an unresolved AST node (see
:mod:`sparkleframe.python.ast.expressions`) and makes no type decisions. A handful of
representative functions are wired to show the pattern across every node kind
(:class:`AttributeReference`, :class:`Literal`, :class:`FunctionCall`, :class:`CaseWhen`);
the remainder are declared with their PySpark signatures and raise — build the matching node
here, then teach the analyzer/evaluator to handle it (and drain the gate in
``sparkleframe/tests/parity/gaps.py``).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast.expressions import AttributeReference, CaseWhen, Expression, FunctionCall, Literal
from sparkleframe.python.column import Column, _column_ref, _to_expression


def _fn(name: str, *cols: Any) -> Column:
    """Build a :class:`Column` wrapping ``FunctionCall(name, [...])`` over column arguments."""
    return Column(FunctionCall(name, [_column_ref(c) for c in cols]))


# --- Column builders (build wired) -----------------------------------------
def col(name: str) -> Column:
    return Column(AttributeReference(name))


def lit(value: Any) -> Column:
    return Column(Literal(value))


class WhenBuilder:
    """Accumulates ``when(...).when(...)`` branches; ``otherwise`` finalizes into a :class:`CaseWhen`."""

    def __init__(self, condition: Any, value: Any) -> None:
        self._branches: list[tuple[Expression, Expression]] = [(_to_expression(condition), _to_expression(value))]

    def when(self, condition: Any, value: Any) -> "WhenBuilder":
        self._branches.append((_to_expression(condition), _to_expression(value)))
        return self

    def otherwise(self, value: Any) -> Column:
        return Column(CaseWhen(self._branches, _to_expression(value)))


def when(condition: Any, value: Any) -> WhenBuilder:
    """Start a multi-branch conditional; chain ``.when(...).otherwise(...)``."""
    return WhenBuilder(condition, value)


# --- A few representative wrappers (build wired) ---------------------------
def abs(col_name: Union[str, Column]) -> Column:
    return _fn("abs", col_name)


def lower(col_name: Union[str, Column]) -> Column:
    return _fn("lower", col_name)


def count(col_name: Union[str, Column]) -> Column:
    return _fn("count", col_name)


def coalesce(*cols: Union[str, Column]) -> Column:
    return _fn("coalesce", *cols)


# --- Remaining surface (slots — build the node when implementing) ----------
def get_json_object(col: Union[str, Column], path: str) -> Column:
    not_implemented_yet("functions.get_json_object")


def from_json(col_name: Union[str, Column], schema: Union[Any, str]) -> Column:
    not_implemented_yet("functions.from_json")


def to_json(col_name: Union[str, Column], options: Any = None) -> Column:
    not_implemented_yet("functions.to_json")


def least(*cols: Any) -> Column:
    not_implemented_yet("functions.least")


def greatest(*cols: Any) -> Column:
    not_implemented_yet("functions.greatest")


def sum(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.sum")


def mean(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.mean")


def min(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.min")


def max(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.max")


def first(col_name: Union[str, Column], ignorenulls: bool = False) -> Column:
    not_implemented_yet("functions.first")


def map_from_entries(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.map_from_entries")


def create_map(*cols: Any) -> Column:
    not_implemented_yet("functions.create_map")


def array(*cols: Any) -> Column:
    not_implemented_yet("functions.array")


def map_keys(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.map_keys")


def collect_list(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.collect_list")


def collect_set(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.collect_set")


def transform(col_name: Union[str, Column], func: Callable[[Column], Any]) -> Column:
    not_implemented_yet("functions.transform")


def round(col_name: Union[str, Column], scale: int = 0) -> Column:
    not_implemented_yet("functions.round")


def to_timestamp(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    not_implemented_yet("functions.to_timestamp")


def date_format(col_name: Union[str, Column], fmt: str) -> Column:
    not_implemented_yet("functions.date_format")


def regexp_replace(col_name: Union[str, Column], pattern: str, replacement: str) -> Column:
    not_implemented_yet("functions.regexp_replace")


def length(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.length")


def asc(column: Union[str, Column]) -> Column:
    not_implemented_yet("functions.asc")


def asc_nulls_first(column: Union[str, Column]) -> Column:
    not_implemented_yet("functions.asc_nulls_first")


def asc_nulls_last(column: Union[str, Column]) -> Column:
    not_implemented_yet("functions.asc_nulls_last")


def desc(column: Union[str, Column]) -> Column:
    not_implemented_yet("functions.desc")


def desc_nulls_first(column: Union[str, Column]) -> Column:
    not_implemented_yet("functions.desc_nulls_first")


def desc_nulls_last(column: Union[str, Column]) -> Column:
    not_implemented_yet("functions.desc_nulls_last")


def rank() -> Column:
    not_implemented_yet("functions.rank")


def dense_rank() -> Column:
    not_implemented_yet("functions.dense_rank")


def row_number() -> Column:
    not_implemented_yet("functions.row_number")


def floor(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.floor")


def pow(base: Any, exponent: Any) -> Column:
    not_implemented_yet("functions.pow")


def isnan(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.isnan")


def try_divide(left: Union[str, Column], right: Union[str, Column]) -> Column:
    not_implemented_yet("functions.try_divide")


def initcap(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.initcap")


def md5(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.md5")


def trim(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.trim")


def nullif(e1: Union[str, Column], e2: Union[str, Column]) -> Column:
    not_implemented_yet("functions.nullif")


def split(col_name: Union[str, Column], pattern: str, limit: int = -1) -> Column:
    not_implemented_yet("functions.split")


def substring(col_name: Union[str, Column], pos: int, length: int) -> Column:
    not_implemented_yet("functions.substring")


def now() -> Column:
    not_implemented_yet("functions.now")


def monotonically_increasing_id() -> Column:
    not_implemented_yet("functions.monotonically_increasing_id")


def current_timestamp() -> Column:
    not_implemented_yet("functions.current_timestamp")


def current_date() -> Column:
    not_implemented_yet("functions.current_date")


def date_sub(col_name: Union[str, Column], days: int) -> Column:
    not_implemented_yet("functions.date_sub")


def datediff(end: Union[str, Column], start: Union[str, Column]) -> Column:
    not_implemented_yet("functions.datediff")


def months_between(end: Union[str, Column], start: Union[str, Column]) -> Column:
    not_implemented_yet("functions.months_between")


def rand(seed: Optional[int] = None) -> Column:
    not_implemented_yet("functions.rand")


def broadcast(df: Any) -> Any:
    not_implemented_yet("functions.broadcast")


def sort_array(col_name: Union[str, Column], asc: bool = True) -> Column:
    not_implemented_yet("functions.sort_array")


def array_contains(col_name: Union[str, Column], value: Union[str, Column, Any]) -> Column:
    not_implemented_yet("functions.array_contains")


def size(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.size")


def filter(col_name: Union[str, Column], func: Callable[[Column], Any]) -> Column:
    not_implemented_yet("functions.filter")


def explode(col_name: Union[str, Column]) -> Column:
    not_implemented_yet("functions.explode")


def concat(*cols: Union[str, Column]) -> Column:
    not_implemented_yet("functions.concat")


def struct(*cols: Any) -> Column:
    not_implemented_yet("functions.struct")


def try_to_timestamp(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    not_implemented_yet("functions.try_to_timestamp")


def to_date(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    not_implemented_yet("functions.to_date")


def try_to_date(col_name: Union[str, Column], fmt: Optional[str] = None) -> Column:
    not_implemented_yet("functions.try_to_date")


def try_element_at(col_name: Union[str, Column], extraction: Union[str, int, Column]) -> Column:
    not_implemented_yet("functions.try_element_at")


def element_at(col_name: Union[str, Column], extraction: Union[str, int, Column]) -> Column:
    not_implemented_yet("functions.element_at")


def uuid() -> Column:
    not_implemented_yet("functions.uuid")
