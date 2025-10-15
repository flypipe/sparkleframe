from __future__ import annotations

from collections.abc import Iterable
from typing import Union

import polars as pl

from sparkleframe.polarsdf.types import DataType


class Column:
    def __init__(self, expr_or_name):
        if isinstance(expr_or_name, str):
            self.expr = pl.col(expr_or_name)
        else:
            self.expr = expr_or_name

    # Arithmetic operations
    def __mul__(self, other):
        return Column(self.expr * _to_expr(other))

    def __add__(self, other):
        return Column(self.expr + _to_expr(other))

    def __sub__(self, other):
        return Column(self.expr - _to_expr(other))

    def __truediv__(self, other):
        return Column(self.expr / _to_expr(other))

    def __radd__(self, other):
        return Column(_to_expr(other) + self.expr)

    def __rsub__(self, other):
        return Column(_to_expr(other) - self.expr)

    def __rmul__(self, other):
        return Column(_to_expr(other) * self.expr)

    def __rtruediv__(self, other):
        return Column(_to_expr(other) / self.expr)

    # Comparison operations
    def __eq__(self, other):
        return Column(self.expr == _to_expr(other))

    def __ne__(self, other):
        return Column(self.expr != _to_expr(other))

    def __lt__(self, other):
        return Column(self.expr < _to_expr(other))

    def __le__(self, other):
        return Column(self.expr <= _to_expr(other))

    def __gt__(self, other):
        return Column(self.expr > _to_expr(other))

    def __ge__(self, other):
        return Column(self.expr >= _to_expr(other))

    # Logical operations
    def __and__(self, other):
        return Column(self.expr & _to_expr(other))

    def __rand__(self, other):
        return Column(_to_expr(other) & self.expr)

    def __or__(self, other):
        return Column(self.expr | _to_expr(other))

    def __ror__(self, other):
        return Column(_to_expr(other) | self.expr)

    def alias(self, name: str) -> Column:
        """
        Mimics pyspark.sql.Column.alias

        Args:
            name (str): Alias name for the column expression

        Returns:
            Column: A new Column with the alias applied
        """
        return Column(self.expr.alias(name))

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
        return Column(self.expr.cast(data_type.to_native()))

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

        return Column(self.expr.is_in(value_list))

    def isNotNull(self) -> Column:
        """
        Mimics pyspark.sql.Column.isNotNull

        Returns:
            Column: A Column representing the non-null condition.
        """
        return Column(self.expr.is_not_null())

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

        return Column(self.expr.str.contains(pattern))

    def getItem(self, key: Union[str, int]) -> "Column":
        """
        Spark-like Column.getItem:
          - If `key` is a string, select a field from a Struct (also works for MapType materialized as Struct).
          - If `key` is an int, select an element from a List/Array column at that index.

        Examples:
            col("s").getItem("a")        # struct field 'a'
            col("arr").getItem(0)        # list element at index 0
            col("col").getItem("key").getItem("key2")  # nested map-as-struct
        """
        if isinstance(key, str):
            # struct field access (our MapType columns are Structs, so this covers maps too)
            return Column(self.expr.struct.field(key))
        elif isinstance(key, int):
            # list/array index access
            return Column(self.expr.list.get(key))
        else:
            raise TypeError(f"getItem expects str or int, got {type(key).__name__}")

    def to_native(self) -> pl.Expr:
        return self.expr


def _to_expr(value):
    if isinstance(value, Column):
        return value.to_native()
    elif isinstance(value, pl.Expr):
        return value
    else:
        return pl.lit(value)
