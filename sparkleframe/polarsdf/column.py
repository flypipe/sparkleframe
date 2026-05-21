from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional, Tuple, Union

import polars as pl

from sparkleframe.polarsdf.column_helpers import (
    _apply_getitem_key,
    _equality_comparison_expr,
    _ordering_comparison_expr,
    _resolve_expr_output_dtype,
    _spark_numeric_widened_type,
    _validated_arithmetic_expr,
    _validated_float64_expr,
    _validated_pow_expr,
)
from sparkleframe.polarsdf.types import BooleanType, DataType, spark_type_name_to_polars


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
            return Column(self.to_native() * _to_expr(other))
        left, right = self._spark_arithmetic_operands(other)
        return Column(left * right)

    def __add__(self, other):
        left, right = self._spark_arithmetic_operands(other)
        return Column(left + right)

    def __sub__(self, other):
        left, right = self._spark_arithmetic_operands(other)
        return Column(left - right)

    def __truediv__(self, other):
        left, right = self._spark_float64_operands(other)
        return Column(left / right)  # both sides already Float64

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
        return Column(left_expr + right_expr)

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
        return Column(left_expr - right_expr)

    def __rmul__(self, other):
        if isinstance(other, int) and not isinstance(other, bool):
            return Column(_to_expr(other) * self.to_native())
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
        return Column(left_expr * right_expr)

    def __rtruediv__(self, other):
        right_expr = self.to_native()
        left_expr = _to_expr(other)
        rd = _resolve_expr_output_dtype(right_expr)
        ld = _resolve_expr_output_dtype(left_expr)
        left_expr = _validated_float64_expr(left_expr, ld)
        right_expr = _validated_float64_expr(right_expr, rd)
        return Column(left_expr / right_expr)

    def __pow__(self, other):
        """Spark-like ``col ** exponent`` (same semantics as :func:`~sparkleframe.polarsdf.functions.pow`)."""
        left, right = self._spark_pow_operands(other)
        return Column(left.pow(right))

    def __rpow__(self, other):
        """``scalar ** col`` (Spark / PySpark ``Column`` supports reflected power)."""
        right_expr = self.to_native()
        left_expr = _to_expr(other)
        rd = _resolve_expr_output_dtype(right_expr)
        ld = _resolve_expr_output_dtype(left_expr)
        left_expr = _validated_pow_expr(left_expr, ld)
        right_expr = _validated_pow_expr(right_expr, rd)
        return Column(left_expr.pow(right_expr))

    def __eq__(self, other):
        return Column(_equality_comparison_expr(self.to_native(), _to_expr(other), equal=True))

    def __ne__(self, other):
        return Column(_equality_comparison_expr(self.to_native(), _to_expr(other), equal=False))

    def __lt__(self, other):
        return Column(_ordering_comparison_expr(self.to_native(), _to_expr(other), "lt"))

    def __le__(self, other):
        return Column(_ordering_comparison_expr(self.to_native(), _to_expr(other), "le"))

    def __gt__(self, other):
        return Column(_ordering_comparison_expr(self.to_native(), _to_expr(other), "gt"))

    def __ge__(self, other):
        return Column(_ordering_comparison_expr(self.to_native(), _to_expr(other), "ge"))

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
        return Column(~self.to_native())

    def __neg__(self):
        """Unary minus (PySpark ``-col``), e.g. ``F.pow(1 + rate, -number_of_periods)`` when ``number_of_periods`` is a column."""
        return Column(-self.to_native())

    def __pos__(self):
        """Unary plus (PySpark ``+col``)."""
        return Column(+self.to_native())

    def alias(self, name: str) -> Column:
        """
        Mimics pyspark.sql.Column.alias

        Args:
            name (str): Alias name for the column expression

        Returns:
            Column: A new Column with the alias applied
        """
        return Column(self.expr, getitem_chain=self._getitem_chain, output_alias=name)

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
        return Column(self.to_native().cast(data_type.to_native(), strict=False))

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
