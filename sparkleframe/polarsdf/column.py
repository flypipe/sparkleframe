from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional, Tuple, Union

import polars as pl

from sparkleframe.polarsdf.column_helpers import (
    _apply_getitem_key,
    _assert_complex_compare_supported,
    _cmp_exprs,
    _coerce_expr_order_datetime,
    _expr_as_string_for_compare,
    _is_complex_polars_dtype,
    _is_numeric_polars_dtype,
    _native_compare_when_complex,
    _resolve_expr_output_dtype,
    _spark_numeric_widened_type,
    _validated_arithmetic_expr,
    _validated_float64_expr,
    _validated_pow_expr,
)
from sparkleframe.polarsdf.types import BooleanType, DataType, spark_type_name_to_polars


def _ordering_comparison(left_col: "Column", other: Any, op: str) -> "Column":
    """
    Spark-like ``< <= > >=``: numeric columns use numeric order; dates/timestamps/strings use
    temporal order when parsable (fixes ``datetime >= date_sub(current_date(), n)`` under
    string schemas); unknown dtypes prefer numeric parse then temporal then lexicographic string.

    Ordering on List / Struct / Array / Map dtypes is not yet supported (Polars has no
    lexicographic compare for nested dtypes); raises :class:`NotImplementedError` with a
    clear message at build time when the dtype is known, or at evaluation time otherwise.
    """
    left = left_col.to_native()
    right = _to_expr(other)
    ld = _resolve_expr_output_dtype(left)
    rd = _resolve_expr_output_dtype(right)
    _assert_complex_compare_supported(op, ld, rd)
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
    coerced_expr = (
        pl.when(numeric_ok)
        .then(_cmp_exprs(left_num, right_num, op))
        .when(temporal_ok)
        .then(_cmp_exprs(left_dt, right_dt, op))
        .otherwise(_cmp_exprs(left_str, right_str, op))
    )
    return Column(_native_compare_when_complex(left, right, op, coerced_expr))


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

    def _spark_arithmetic_operands(self, other: Any) -> tuple[pl.Expr, pl.Expr]:
        """
        Spark-like type promotion for ``+``, ``-``, ``*``.
        Rejects non-numeric types; widens to the larger numeric type when
        operands differ (e.g. int + long -> long); preserves the type when
        both sides match (e.g. int + int -> int).
        """
        left = self.to_native()
        right = _to_expr(other)
        ld = _resolve_expr_output_dtype(left)
        rd = _resolve_expr_output_dtype(right)
        left = _validated_arithmetic_expr(left, ld)
        right = _validated_arithmetic_expr(right, rd)
        if ld is not None and rd is not None:
            target = _spark_numeric_widened_type(ld, rd)
            if target is not None:
                left = left.cast(target, strict=False)
                right = right.cast(target, strict=False)
        return left, right

    def _spark_float64_operands(self, other: Any) -> tuple[pl.Expr, pl.Expr]:
        """
        For ``/`` which always promotes to Float64 in Spark and rejects
        non-numeric operands (string, boolean, binary, date/timestamp, complex).
        """
        left = self.to_native()
        right = _to_expr(other)
        ld = _resolve_expr_output_dtype(left)
        rd = _resolve_expr_output_dtype(right)
        return _validated_float64_expr(left, ld), _validated_float64_expr(right, rd)

    def _spark_pow_operands(self, other: Any) -> tuple[pl.Expr, pl.Expr]:
        """
        For ``**`` which always promotes to Float64 in Spark. Spark's ``pow``
        auto-casts string operands, so we are more lenient here than for ``/``.
        """
        left = self.to_native()
        right = _to_expr(other)
        ld = _resolve_expr_output_dtype(left)
        rd = _resolve_expr_output_dtype(right)
        return _validated_pow_expr(left, ld), _validated_pow_expr(right, rd)

    # Arithmetic operations
    def __mul__(self, other):
        if isinstance(other, int) and not isinstance(other, bool):
            c = Column(self.to_native() * _to_expr(other))
            c._broadcast_row_count_in_select = bool(
                getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
            )
            return c
        left, right = self._spark_arithmetic_operands(other)
        c = Column(left * right)
        c._broadcast_row_count_in_select = bool(
            getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
        )
        return c

    def __add__(self, other):
        left, right = self._spark_arithmetic_operands(other)
        c = Column(left + right)
        c._broadcast_row_count_in_select = bool(
            getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
        )
        return c

    def __sub__(self, other):
        left, right = self._spark_arithmetic_operands(other)
        c = Column(left - right)
        c._broadcast_row_count_in_select = bool(
            getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
        )
        return c

    def __truediv__(self, other):
        left, right = self._spark_float64_operands(other)
        c = Column(left / right)  # both sides already Float64
        c._broadcast_row_count_in_select = bool(
            getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
        )
        return c

    def __radd__(self, other):
        right_expr = self.to_native()
        left_expr = _to_expr(other)
        rd = _resolve_expr_output_dtype(right_expr)
        ld = _resolve_expr_output_dtype(left_expr)
        right_expr = _validated_arithmetic_expr(right_expr, rd)
        left_expr = _validated_arithmetic_expr(left_expr, ld)
        if ld is not None and rd is not None:
            target = _spark_numeric_widened_type(ld, rd)
            if target is not None:
                left_expr = left_expr.cast(target, strict=False)
                right_expr = right_expr.cast(target, strict=False)
        c = Column(left_expr + right_expr)
        c._broadcast_row_count_in_select = bool(
            _operand_broadcasts_in_select(other) and getattr(self, "_broadcast_row_count_in_select", False)
        )
        return c

    def __rsub__(self, other):
        right_expr = self.to_native()
        left_expr = _to_expr(other)
        rd = _resolve_expr_output_dtype(right_expr)
        ld = _resolve_expr_output_dtype(left_expr)
        right_expr = _validated_arithmetic_expr(right_expr, rd)
        left_expr = _validated_arithmetic_expr(left_expr, ld)
        if ld is not None and rd is not None:
            target = _spark_numeric_widened_type(ld, rd)
            if target is not None:
                left_expr = left_expr.cast(target, strict=False)
                right_expr = right_expr.cast(target, strict=False)
        c = Column(left_expr - right_expr)
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
        right_expr = self.to_native()
        left_expr = _to_expr(other)
        rd = _resolve_expr_output_dtype(right_expr)
        ld = _resolve_expr_output_dtype(left_expr)
        right_expr = _validated_arithmetic_expr(right_expr, rd)
        left_expr = _validated_arithmetic_expr(left_expr, ld)
        if ld is not None and rd is not None:
            target = _spark_numeric_widened_type(ld, rd)
            if target is not None:
                left_expr = left_expr.cast(target, strict=False)
                right_expr = right_expr.cast(target, strict=False)
        c = Column(left_expr * right_expr)
        c._broadcast_row_count_in_select = bool(
            _operand_broadcasts_in_select(other) and getattr(self, "_broadcast_row_count_in_select", False)
        )
        return c

    def __rtruediv__(self, other):
        right_expr = self.to_native()
        left_expr = _to_expr(other)
        rd = _resolve_expr_output_dtype(right_expr)
        ld = _resolve_expr_output_dtype(left_expr)
        left_expr = _validated_float64_expr(left_expr, ld)
        right_expr = _validated_float64_expr(right_expr, rd)
        c = Column(left_expr / right_expr)
        c._broadcast_row_count_in_select = bool(
            _operand_broadcasts_in_select(other) and getattr(self, "_broadcast_row_count_in_select", False)
        )
        return c

    def __pow__(self, other):
        """Spark-like ``col ** exponent`` (same semantics as :func:`~sparkleframe.polarsdf.functions.pow`)."""
        left, right = self._spark_pow_operands(other)
        c = Column(left.pow(right))
        c._broadcast_row_count_in_select = bool(
            getattr(self, "_broadcast_row_count_in_select", False) and _operand_broadcasts_in_select(other)
        )
        return c

    def __rpow__(self, other):
        """``scalar ** col`` (Spark / PySpark ``Column`` supports reflected power)."""
        right_expr = self.to_native()
        left_expr = _to_expr(other)
        rd = _resolve_expr_output_dtype(right_expr)
        ld = _resolve_expr_output_dtype(left_expr)
        left_expr = _validated_pow_expr(left_expr, ld)
        right_expr = _validated_pow_expr(right_expr, rd)
        c = Column(left_expr.pow(right_expr))
        c._broadcast_row_count_in_select = bool(
            _operand_broadcasts_in_select(other) and getattr(self, "_broadcast_row_count_in_select", False)
        )
        return c

    # Comparison operations
    def _numeric_comparison_operands(self, other):
        left_str = _expr_as_string_for_compare(self.to_native())
        right_str = _expr_as_string_for_compare(_to_expr(other))
        left = left_str.cast(pl.Float64, strict=False)
        right = right_str.cast(pl.Float64, strict=False)
        return left, right, left_str, right_str

    def _equality_comparison(self, other: Any, equal: bool) -> "Column":
        """
        Spark-like ``==`` / ``!=`` semantics: cross-type equality coerces through
        string / numeric (so ``col(int) == lit('1')`` matches Spark), but
        list / struct / array operands fall back to Polars' native equality.

        Raises :class:`NotImplementedError` with a clear message for MapType operands
        (Polars stores maps as ``List(Struct([key, value]))`` so we can't replicate
        Spark's map equality semantics yet).
        """
        left = self.to_native()
        right = _to_expr(other)
        ld = _resolve_expr_output_dtype(left)
        rd = _resolve_expr_output_dtype(right)
        op = "eq" if equal else "ne"
        _assert_complex_compare_supported(op, ld, rd)
        if _is_complex_polars_dtype(ld) or _is_complex_polars_dtype(rd):
            return Column((left == right) if equal else (left != right))
        left_num, right_num, left_str, right_str = self._numeric_comparison_operands(other)
        numeric_valid = left_num.is_not_null() & right_num.is_not_null()
        if equal:
            coerced = pl.when(numeric_valid).then(left_num == right_num).otherwise(left_str == right_str)
        else:
            coerced = pl.when(numeric_valid).then(left_num != right_num).otherwise(left_str != right_str)
        if ld is not None and rd is not None:
            return Column(coerced)
        return Column(_native_compare_when_complex(left, right, "eq" if equal else "ne", coerced))

    def __eq__(self, other):
        return self._equality_comparison(other, equal=True)

    def __ne__(self, other):
        return self._equality_comparison(other, equal=False)

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

    def __neg__(self):
        """Unary minus (PySpark ``-col``), e.g. ``F.pow(1 + rate, -number_of_periods)`` when ``number_of_periods`` is a column."""
        c = Column(-self.to_native())
        c._broadcast_row_count_in_select = bool(getattr(self, "_broadcast_row_count_in_select", False))
        return c

    def __pos__(self):
        """Unary plus (PySpark ``+col``)."""
        c = Column(+self.to_native())
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
