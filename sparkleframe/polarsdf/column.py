from __future__ import annotations

import contextvars
import re
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Generator, Optional, Tuple, Union

import polars as pl

from sparkleframe.polarsdf.types import BooleanType, DataType, spark_type_name_to_polars

_polars_schema_ctx: contextvars.ContextVar[Optional[pl.Schema]] = contextvars.ContextVar(
    "sparkleframe_polars_schema", default=None
)


@contextmanager
def _polars_schema_for(schema: pl.Schema) -> Generator[None, None, None]:
    """Set the active Polars frame schema so Column.getItem can resolve struct/list types."""
    token = _polars_schema_ctx.set(schema)
    try:
        yield
    finally:
        _polars_schema_ctx.reset(token)


def _output_dtype_of_expr(expr: pl.Expr, schema: pl.Schema) -> Optional[pl.DataType]:
    try:
        return pl.LazyFrame(schema=schema).select(expr.alias("_x")).collect_schema()["_x"]
    except Exception:
        return None


def _resolve_expr_output_dtype(expr: pl.Expr) -> Optional[pl.DataType]:
    sch = _polars_schema_ctx.get()
    if sch is not None:
        d = _output_dtype_of_expr(expr, sch)
        if d is not None:
            return d
    try:
        meta = expr.meta
        if hasattr(meta, "output_dtype"):
            return meta.output_dtype()  # type: ignore[no-any-return]
    except Exception:
        pass
    return None


_NUMERIC_ORDER_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
)


def _is_numeric_polars_dtype(dt: Optional[pl.DataType]) -> bool:
    if dt is None:
        return False
    if isinstance(dt, pl.Decimal):
        return True
    return dt in _NUMERIC_ORDER_DTYPES


def _cmp_exprs(left: pl.Expr, right: pl.Expr, op: str) -> pl.Expr:
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    raise ValueError(op)


_DT_LIKE_STRING = re.compile(r"\d{4}.*[-/:T]")


def _parse_datetime_string_safe(v: str) -> Optional[datetime]:
    if not _DT_LIKE_STRING.search(v):
        return None
    try:
        s = pl.Series("_v", [v], dtype=pl.Utf8)
        parsed = s.str.to_datetime(strict=False)
        if parsed.is_null().all():
            return None
        ts = parsed.dt.replace_time_zone(None)[0]
        return ts  # type: ignore[no-any-return]
    except Exception:
        return None


def _series_to_string_compare(s: pl.Series) -> pl.Series:
    """Utf8 cast for comparisons; Object columns (e.g. ``getItem``) become ``str`` per cell."""
    if s.dtype == pl.Object:
        out: list[Any] = []
        for x in s.to_list():
            out.append(None if x is None else str(x))
        return pl.Series(s.name, out, dtype=pl.Utf8)
    return s.cast(pl.Utf8, strict=False)


def _expr_as_string_for_compare(e: pl.Expr) -> pl.Expr:
    return e.map_batches(_series_to_string_compare, return_dtype=pl.Utf8)


def _series_coerce_order_datetime(s: pl.Series) -> pl.Series:
    """Per-row datetime coercion for ordering (avoids Polars raising on ``str.to_datetime`` in ``when``)."""
    if s.len() == 0:
        return pl.Series(s.name, [], dtype=pl.Datetime("us"))
    out: list[Any] = []
    for v in s.to_list():
        if v is None:
            out.append(None)
        elif isinstance(v, datetime):
            ts = v
            if ts.tzinfo is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            out.append(ts)
        elif isinstance(v, date) and not isinstance(v, datetime):
            out.append(datetime(v.year, v.month, v.day))
        elif isinstance(v, str):
            out.append(_parse_datetime_string_safe(v))
        else:
            out.append(None)
    return pl.Series(s.name, out, dtype=pl.Datetime("us"))


def _coerce_expr_order_datetime(e: pl.Expr) -> pl.Expr:
    """Coerce to naive microsecond datetimes for Spark-like ordering (date / timestamp / ISO strings)."""
    return e.map_batches(_series_coerce_order_datetime, return_dtype=pl.Datetime("us"))


def _ordering_comparison(left_col: "Column", other: Any, op: str) -> "Column":
    """
    Spark-like ``< <= > >=``: numeric columns use numeric order; dates/timestamps/strings use
    temporal order when parsable (fixes ``datetime >= date_sub(current_date(), n)`` under
    string schemas); unknown dtypes prefer numeric parse then temporal then lexicographic string.
    """
    left = left_col.to_native()
    right = _to_expr(other)
    ld = _resolve_expr_output_dtype(left)
    rd = _resolve_expr_output_dtype(right)
    left_s = _expr_as_string_for_compare(left)
    right_s = _expr_as_string_for_compare(right)
    left_num = left_s.cast(pl.Float64, strict=False)
    right_num = right_s.cast(pl.Float64, strict=False)
    numeric_ok = left_num.is_not_null() & right_num.is_not_null()
    if _is_numeric_polars_dtype(ld) or _is_numeric_polars_dtype(rd):
        return Column(_cmp_exprs(left_num, right_num, op))

    left_dt = _coerce_expr_order_datetime(left)
    right_dt = _coerce_expr_order_datetime(right)
    temporal_ok = left_dt.is_not_null() & right_dt.is_not_null()
    left_str = left_s
    right_str = right_s
    if ld is not None or rd is not None:
        return Column(
            pl.when(temporal_ok)
            .then(_cmp_exprs(left_dt, right_dt, op))
            .when(numeric_ok)
            .then(_cmp_exprs(left_num, right_num, op))
            .otherwise(_cmp_exprs(left_str, right_str, op))
        )
    return Column(
        pl.when(numeric_ok)
        .then(_cmp_exprs(left_num, right_num, op))
        .when(temporal_ok)
        .then(_cmp_exprs(left_dt, right_dt, op))
        .otherwise(_cmp_exprs(left_str, right_str, op))
    )


def _apply_getitem_key(expr: pl.Expr, key: Union[str, int]) -> pl.Expr:
    """
    One Spark getItem step on ``expr`` (struct field, list index, or map fallbacks).
    Assumes the active :func:`_polars_schema_for` is set when resolving dtypes.
    """
    if isinstance(key, str):
        dtype = _resolve_expr_output_dtype(expr)
        if isinstance(dtype, pl.Struct):
            field_names = {f.name for f in dtype.fields}
            if key not in field_names:
                # Spark returns null when the struct schema has no such field (Polars raises).
                return pl.lit(None).cast(pl.String)
            return expr.struct.field(key)
        if isinstance(dtype, pl.List) and isinstance(getattr(dtype, "inner", None), pl.Struct):
            inner: pl.Struct = dtype.inner
            field_names = {f.name for f in inner.fields}
            # Physical struct field (e.g. ``key`` / ``value`` columns) beats map-by-name lookup.
            if key in field_names:
                return expr.list.eval(pl.element().struct.field(key))
            if "key" in field_names and "value" in field_names:
                return (
                    expr.list.eval(
                        pl.when(pl.element().struct.field("key") == pl.lit(key)).then(
                            pl.element().struct.field("value")
                        )
                    )
                    .list.drop_nulls()
                    .list.first()
                )

        def _extract_by_key(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, dict):
                return value.get(key)
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict) and entry.get("key") == key:
                        return entry.get("value")
                return None
            getter = getattr(value, "get", None)
            if callable(getter):
                try:
                    return getter(key)
                except Exception:
                    return None
            return None

        return expr.map_elements(_extract_by_key, return_dtype=pl.Object)
    if isinstance(key, int):
        dtype = _resolve_expr_output_dtype(expr)
        if isinstance(dtype, pl.List):
            return expr.list.get(key)

        def _index_at(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                if key < 0 or key >= len(v):
                    return None
                return v[key]
            return None

        return expr.map_elements(_index_at, return_dtype=pl.Object)
    raise TypeError(f"getItem key must be str or int, got {type(key).__name__}")


class Column:
    def __init__(
        self,
        expr_or_name: Union[str, pl.Expr, Any],
        *,
        getitem_chain: Tuple[Union[str, int], ...] = (),
        output_alias: Optional[str] = None,
    ):
        self._getitem_chain = getitem_chain
        self._output_alias = output_alias
        if isinstance(expr_or_name, str):
            self.expr = pl.col(expr_or_name)
        else:
            self.expr = expr_or_name
        self._broadcast_row_count_in_select: bool = False

    def _binary_arithmetic_float_operands(self, other: Any) -> tuple[pl.Expr, pl.Expr]:
        """
        Spark-like implicit numeric widening for ``+``, ``-``, ``*``, ``/`` (e.g. Utf8 decimals).
        """
        left = self.to_native().cast(pl.Float64, strict=False)
        right = _to_expr(other).cast(pl.Float64, strict=False)
        return left, right

    # Arithmetic operations
    def __mul__(self, other):
        # Preserve integral ``list.eval`` / ``transform`` behaviour for ``x * <int literal>`` (Spark).
        if isinstance(other, int) and not isinstance(other, bool):
            c = Column(self.to_native() * _to_expr(other))
            c._broadcast_row_count_in_select = bool(
                getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
            )
            return c
        left, right = self._binary_arithmetic_float_operands(other)
        c = Column(left * right)
        c._broadcast_row_count_in_select = bool(
            getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
        )
        return c

    def __add__(self, other):
        left, right = self._binary_arithmetic_float_operands(other)
        c = Column(left + right)
        c._broadcast_row_count_in_select = bool(
            getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
        )
        return c

    def __sub__(self, other):
        left, right = self._binary_arithmetic_float_operands(other)
        c = Column(left - right)
        c._broadcast_row_count_in_select = bool(
            getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
        )
        return c

    def __truediv__(self, other):
        left, right = self._binary_arithmetic_float_operands(other)
        c = Column(left / right)
        c._broadcast_row_count_in_select = bool(
            getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
        )
        return c

    def __radd__(self, other):
        left, right = _to_expr(other).cast(pl.Float64, strict=False), self.to_native().cast(pl.Float64, strict=False)
        c = Column(left + right)
        c._broadcast_row_count_in_select = bool(
            _operand_broadcasts_in_select(other) and getattr(self, "_broadcast_row_count_in_select", False)
        )
        return c

    def __rsub__(self, other):
        left, right = _to_expr(other).cast(pl.Float64, strict=False), self.to_native().cast(pl.Float64, strict=False)
        c = Column(left - right)
        c._broadcast_row_count_in_select = bool(
            _operand_broadcasts_in_select(other) and getattr(self, "_broadcast_row_count_in_select", False)
        )
        return c

    def __rmul__(self, other):
        if isinstance(other, int) and not isinstance(other, bool):
            c = Column(_to_expr(other) * self.to_native())
            c._broadcast_row_count_in_select = bool(
                _operand_broadcasts_in_select(other) and getattr(self, "_broadcast_row_count_in_select", False)
            )
            return c
        left, right = _to_expr(other).cast(pl.Float64, strict=False), self.to_native().cast(pl.Float64, strict=False)
        c = Column(left * right)
        c._broadcast_row_count_in_select = bool(
            _operand_broadcasts_in_select(other) and getattr(self, "_broadcast_row_count_in_select", False)
        )
        return c

    def __rtruediv__(self, other):
        left, right = _to_expr(other).cast(pl.Float64, strict=False), self.to_native().cast(pl.Float64, strict=False)
        c = Column(left / right)
        c._broadcast_row_count_in_select = bool(
            _operand_broadcasts_in_select(other) and getattr(self, "_broadcast_row_count_in_select", False)
        )
        return c

    # Comparison operations
    def _numeric_comparison_operands(self, other):
        # Object columns cannot cast to Float/String directly; map Python values to Utf8 first.
        left_str = _expr_as_string_for_compare(self.to_native())
        right_str = _expr_as_string_for_compare(_to_expr(other))
        left = left_str.cast(pl.Float64, strict=False)
        right = right_str.cast(pl.Float64, strict=False)
        return left, right, left_str, right_str

    def __eq__(self, other):
        left_num, right_num, left_str, right_str = self._numeric_comparison_operands(other)
        numeric_valid = left_num.is_not_null() & right_num.is_not_null()
        return Column(pl.when(numeric_valid).then(left_num == right_num).otherwise(left_str == right_str))

    def __ne__(self, other):
        left_num, right_num, left_str, right_str = self._numeric_comparison_operands(other)
        numeric_valid = left_num.is_not_null() & right_num.is_not_null()
        return Column(pl.when(numeric_valid).then(left_num != right_num).otherwise(left_str != right_str))

    def __lt__(self, other):
        return _ordering_comparison(self, other, "lt")

    def __le__(self, other):
        return _ordering_comparison(self, other, "le")

    def __gt__(self, other):
        return _ordering_comparison(self, other, "gt")

    def __ge__(self, other):
        return _ordering_comparison(self, other, "ge")

    # Logical operations
    def __and__(self, other):
        return Column(self.to_native() & _to_expr(other))

    def __rand__(self, other):
        return Column(_to_expr(other) & self.to_native())

    def __or__(self, other):
        return Column(self.to_native() | _to_expr(other))

    def __ror__(self, other):
        return Column(_to_expr(other) | self.to_native())

    def __invert__(self):
        c = Column(~self.to_native())
        c._broadcast_row_count_in_select = bool(getattr(self, "_broadcast_row_count_in_select", False))
        return c

    def alias(self, name: str) -> Column:
        """
        Mimics pyspark.sql.Column.alias

        Args:
            name (str): Alias name for the column expression

        Returns:
            Column: A new Column with the alias applied
        """
        c = Column(self.expr, getitem_chain=self._getitem_chain, output_alias=name)
        c._broadcast_row_count_in_select = bool(getattr(self, "_broadcast_row_count_in_select", False))
        return c

    def asc(self) -> "Column":
        base = self.to_native()
        column_ = Column(base.sort(descending=False, nulls_last=False))
        column_._sort_col = base
        column_._sort_descending = False
        column_._sort_nulls_last = False
        return column_

    def desc(self) -> "Column":
        base = self.to_native()
        column_ = Column(base.sort(descending=True, nulls_last=True))
        column_._sort_col = base
        column_._sort_descending = True
        column_._sort_nulls_last = True
        return column_

    def desc_nulls_last(self) -> "Column":
        return self.desc()

    def asc_nulls_last(self) -> "Column":
        base = self.to_native()
        column_ = Column(base.sort(descending=False, nulls_last=True))
        column_._sort_col = base
        column_._sort_descending = False
        column_._sort_nulls_last = True
        return column_

    def cast(self, data_type: DataType) -> Column:
        """
        Mimics pyspark.sql.Column.cast using Polars' cast().

        Args:
            data_type (DataType): A sparkleframe-defined DataType object.

        Returns:
            Column: A new Column with the expression casted.
        """
        if not isinstance(data_type, DataType):
            raise TypeError(f"cast() expects a DataType, got {type(data_type)}")
        if isinstance(data_type, BooleanType):
            # Polars does not support strict Utf8->Boolean casting directly.
            # Parse common Spark-like boolean string values first.
            base = self.to_native()
            string_expr = base.cast(pl.String, strict=False).str.strip_chars().str.to_lowercase()
            parsed_bool = (
                pl.when(base.is_null())
                .then(pl.lit(None, dtype=pl.Boolean))
                .when(string_expr.is_in(["true", "t", "1", "yes", "y"]))
                .then(pl.lit(True))
                .when(string_expr.is_in(["false", "f", "0", "no", "n"]))
                .then(pl.lit(False))
                .otherwise(pl.lit(None, dtype=pl.Boolean))
            )
            return Column(parsed_bool)
        # Use Polars `strict=False` so invalid values become null per row, like Spark 4
        # (ANSI) casts. `strict=True` in Polars fails the whole expression for any bad row.
        c = Column(self.to_native().cast(data_type.to_native(), strict=False))
        c._broadcast_row_count_in_select = bool(getattr(self, "_broadcast_row_count_in_select", False))
        return c

    def try_cast(self, data_type: Union[DataType, str]) -> "Column":
        """
        Mimics pyspark.sql.Column.try_cast (Spark 4+).

        Attempts to cast the column to the target type; returns null instead
        of raising an error when the value cannot be converted.

        Args:
            data_type (DataType or str): Target type as a sparkleframe DataType
                or a Spark type-name string (e.g. "int", "double", "string").

        Returns:
            Column: A new Column with the non-strict cast applied.
        """
        simple_type_name = None
        if isinstance(data_type, DataType):
            simple_type_name = data_type.simpleString().lower()
            native = data_type.to_native()
        elif isinstance(data_type, str):
            simple_type_name = data_type.strip().lower()
            native = spark_type_name_to_polars(data_type)
        elif hasattr(data_type, "simpleString"):
            simple_type_name = str(data_type.simpleString()).lower()
            native = spark_type_name_to_polars(simple_type_name)
        else:
            raise TypeError(f"try_cast() expects a DataType or str, got {type(data_type)}")

        if simple_type_name in {"bool", "boolean"}:
            true_values = {"true", "t", "1", "yes", "y"}
            false_values = {"false", "f", "0", "no", "n"}

            def _parse_bool(value: Any):
                if value is None:
                    return None
                lowered = str(value).strip().lower()
                if lowered in true_values:
                    return True
                if lowered in false_values:
                    return False
                return None

            if isinstance(self.expr, pl.Series):
                return Column(self.expr.map_elements(_parse_bool, return_dtype=pl.Boolean))

            base = self.to_native()
            string_expr = base.cast(pl.String, strict=False).str.strip_chars().str.to_lowercase()
            parsed_bool = (
                pl.when(base.is_null())
                .then(pl.lit(None, dtype=pl.Boolean))
                .when(string_expr.is_in(list(true_values)))
                .then(pl.lit(True))
                .when(string_expr.is_in(list(false_values)))
                .then(pl.lit(False))
                .otherwise(pl.lit(None, dtype=pl.Boolean))
            )
            return Column(parsed_bool)
        return Column(self.to_native().cast(native, strict=False))

    def isin(self, *values) -> Column:
        """
        Mimics pyspark.sql.Column.isin and supports both:
            col("x").isin("a", "b") and col("x").isin(["a", "b"])

        Args:
            *values: A list of values or individual arguments.

        Returns:
            Column: A Column representing a boolean expression.
        """
        # If a single iterable (non-str) is passed, use that directly
        if len(values) == 1 and isinstance(values[0], Iterable) and not isinstance(values[0], str):
            value_list = list(values[0])
        else:
            value_list = list(values)

        return Column(self.to_native().is_in(value_list))

    def isNotNull(self) -> Column:
        """
        Mimics pyspark.sql.Column.isNotNull

        Returns:
            Column: A Column representing the non-null condition.
        """
        return Column(self.to_native().is_not_null())

    def isNull(self) -> Column:
        """
        Mimics pyspark.sql.Column.isNull

        Returns:
            Column: A Column representing the null condition.
        """
        return Column(self.to_native().is_null())

    def rlike(self, pattern: str) -> Column:
        """
        Mimics pyspark.sql.Column.rlike using Polars' regex matching.

        Args:
            pattern (str): Regular expression pattern to match.

        Returns:
            Column: A new Column representing a boolean expression.
        """
        if not isinstance(pattern, str):
            raise TypeError(f"rlike() expects a string pattern, got {type(pattern)}")

        return Column(self.to_native().str.contains(pattern))

    def getItem(self, key: Union[str, int]) -> "Column":
        """
        Spark-like Column.getItem:
          - If `key` is a string, select a field from a Struct (also works for MapType materialized as Struct).
          - If `key` is an int, select an element from a List/Array column at that index.

        Indexing is applied lazily in :meth:`to_native` so the active DataFrame schema
        (set during ``select`` / ``withColumn``) can be used to pick struct vs list paths.

        Examples:
            col("s").getItem("a")        # struct field 'a'
            col("arr").getItem(0)        # list element at index 0
            col("col").getItem("key").getItem("key2")  # nested map-as-struct
        """
        if not isinstance(key, (str, int)):
            raise TypeError(f"getItem expects str or int, got {type(key).__name__}")
        return Column(self.expr, getitem_chain=(*self._getitem_chain, key), output_alias=None)

    def __getitem__(self, key: Union[str, int]) -> "Column":
        """
        Support PySpark-style indexing syntax on Column expressions.

        Examples:
            F.split(F.col("partner_name_variation"), "-")[0]
            F.col("struct_col")["field_name"]
        """
        return self.getItem(key)

    def _to_native_getitem_only(self) -> pl.Expr:
        """Expression after applying the deferred :meth:`getItem` chain, without user ``alias``."""
        e: pl.Expr = self.expr
        for k in self._getitem_chain:
            e = _apply_getitem_key(e, k)
        return e

    def to_native(self) -> pl.Expr:
        e = self._to_native_getitem_only()
        if self._output_alias is not None:
            e = e.alias(self._output_alias)
        return e

    def contains(self, substring: str) -> "Column":
        """
        Mimics pyspark.sql.Column.contains.

        Checks if the string column contains the given substring (literal match, case-sensitive).

        Args:
            substring (str): The substring to search for (treated as a literal, not a regex).

        Returns:
            Column: A boolean Column: True if substring is found, False if not, and null for null inputs.
        """
        if not isinstance(substring, str):
            raise TypeError(f"contains() expects a string substring, got {type(substring).__name__}")
        return Column(self.to_native().str.contains(substring, literal=True))


def _to_expr(value):
    if isinstance(value, Column):
        return value.to_native()
    elif isinstance(value, pl.Expr):
        return value
    else:
        return pl.lit(value)


def _operand_broadcasts_in_select(other: Any) -> bool:
    """True if ``other`` is a row-aligned literal operand for Spark-style ``select``."""
    if isinstance(other, Column):
        return bool(getattr(other, "_broadcast_row_count_in_select", False))
    if isinstance(other, (int, float, str, bool, type(None))):
        return True
    if isinstance(other, pl.Expr):
        return False
    return False
