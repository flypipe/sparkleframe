from __future__ import annotations

import polars as pl

from sparkleframe.polarsdf.types import DataType


class Column:
    def __init__(self, expr_or_name):
        if isinstance(expr_or_name, str):
            self.expr = pl.col(expr_or_name)
        else:
            self.expr = expr_or_name

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

    def isin(self, values: list) -> Column:
        """
        Mimics pyspark.sql.Column.isin

        Args:
            values (list): A list of values to check for inclusion.

        Returns:
            Column: A Column representing a boolean expression.
        """
        return Column(self.expr.is_in(values))

    def isNotNull(self) -> Column:
        """
        Mimics pyspark.sql.Column.isNotNull

        Returns:
            Column: A Column representing the non-null condition.
        """
        return Column(self.expr.is_not_null())

    def otherwise(self, value) -> Column:
        """
        Finalize a conditional column by providing the fallback (else) value.
        """
        return Column(self.expr.otherwise(_to_expr(value)))

    def to_native(self) -> pl.Expr:
        return self.expr

def _to_expr(value):
    if isinstance(value, Column):
        return value.to_native()
    elif isinstance(value, pl.Expr):
        return value
    else:
        return pl.lit(value)
