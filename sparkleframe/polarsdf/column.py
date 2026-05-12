from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional, Tuple, Union

import polars as pl

from sparkleframe.polarsdf.column_helpers import (
    _apply_getitem_key,
    _coerce_mixed_arithmetic_operands,
    _equality_comparison_expr,
    _has_decimal_operand,
    _ordering_comparison_expr,
    _parse_bool_string,
    _resolve_expr_output_dtype,
    _spark_decimal_div_result_type,
    _spark_numeric_widened_type,
    _string_to_bool_expr,
    _string_to_bool_expr_strict,
    _validated_arithmetic_expr,
    _validated_decimal_div_expr,
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

    def _spark_arithmetic_operands(self, other: Any, op: str = "+") -> tuple[pl.Expr, pl.Expr]:
        """
        Spark-like type promotion for ``+``, ``-``, ``*``.
        Rejects non-numeric types; widens to the larger numeric type when
        operands differ (e.g. int + long -> long); preserves the type when
        both sides match (e.g. int + int -> int).

        Cross-type coercion (numeric + string, int + date) is attempted before
        per-operand validation so mixed-type operations match Spark behaviour.
        """
        left = self.to_native()
        right = _to_expr(other)
        ld = _resolve_expr_output_dtype(left)
        rd = _resolve_expr_output_dtype(right)
        coerced = _coerce_mixed_arithmetic_operands(left, right, ld, rd, op)
        if coerced is not None:
            return coerced
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
        For ``/`` which promotes to Float64 in Spark for most types, but preserves
        Decimal arithmetic when at least one operand is Decimal.
        Cross-type numeric+string coercion is applied first.
        """
        left = self.to_native()
        right = _to_expr(other)
        ld = _resolve_expr_output_dtype(left)
        rd = _resolve_expr_output_dtype(right)
        coerced = _coerce_mixed_arithmetic_operands(left, right, ld, rd, "/")
        if coerced is not None:
            return coerced[0].cast(pl.Float64, strict=False), coerced[1].cast(pl.Float64, strict=False)
        if _has_decimal_operand(ld, rd):
            left = _validated_decimal_div_expr(left, ld)
            right = _validated_decimal_div_expr(right, rd)
            return left, right
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
        left, right = self._spark_arithmetic_operands(other, op="*")
        return Column(left * right)

    def __add__(self, other):
        left, right = self._spark_arithmetic_operands(other, op="+")
        return Column(left + right)

    def __sub__(self, other):
        left, right = self._spark_arithmetic_operands(other, op="-")
        return Column(left - right)

    def __truediv__(self, other):
        left, right = self._spark_float64_operands(other)
        result = left / right
        ld = _resolve_expr_output_dtype(self.to_native())
        rd = _resolve_expr_output_dtype(_to_expr(other))
        if _has_decimal_operand(ld, rd):
            result = result.cast(_spark_decimal_div_result_type(ld, rd), strict=False)
        return Column(result)

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
        """Unary minus (PySpark ``-col``)."""
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

        Follows Spark 4's default ``spark.sql.ansi.enabled=true`` semantics: raises
        ``CAST_INVALID_INPUT`` on malformed input (e.g. ``"Bob".cast(int)``,
        ``"maybe".cast(boolean)``) instead of silently returning null. Use
        :meth:`try_cast` for the lenient variant that returns null.

        Numeric / boolean coercions that Spark accepts (e.g. ``int -> bool`` via
        nonzero -> true) still flow through Polars' native cast.

        Args:
            data_type (DataType): A sparkleframe-defined DataType object.

        Returns:
            Column: A new Column with the expression casted.
        """
        if not isinstance(data_type, DataType):
            raise TypeError(f"cast() expects a DataType, got {type(data_type)}")
        native = self.to_native()
        if isinstance(data_type, BooleanType):
            # The helper dispatches at runtime: string source -> strict literal
            # parse (CAST_INVALID_INPUT on unknown); numeric / bool source -> native
            # Polars cast (which already matches Spark for those types).
            return Column(_string_to_bool_expr_strict(native))
        # ``strict=True`` mirrors Spark 4 ANSI: Polars raises (wrapped) when any
        # row fails the cast, where Spark raises ``CAST_INVALID_INPUT``.
        return Column(native.cast(data_type.to_native(), strict=True))

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
            if isinstance(self.expr, pl.Series):
                return Column(self.expr.map_elements(_parse_bool_string, return_dtype=pl.Boolean))
            return Column(_string_to_bool_expr(self.to_native()))
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
