"""Shared, private helpers for ``dataframe.py`` (pure-Python engine).

Per the repo convention (CLAUDE.md), shared/private logic for the public ``dataframe`` surface
lives here rather than in ``dataframe.py``. These helpers bridge the stored row tuples + PySpark
schema into the dict form the analyze/evaluate phases consume, and reassemble the output frame.
"""

from __future__ import annotations

from typing import Any, List

from sparkleframe.python.ast.expressions import Alias, AttributeReference, BinaryExpression, Expression


def field_names(schema: Any) -> List[str]:
    """Column names of a PySpark ``StructType`` schema."""
    return [field.name for field in schema.fields]


def rows_as_dicts(rows: List[tuple], schema: Any) -> List[dict]:
    """Pair each stored row tuple with the schema's column names → list of dicts."""
    names = field_names(schema)
    return [dict(zip(names, row)) for row in rows]


def output_name(resolved: Expression) -> str:
    """Derive the output column name of a resolved expression, Spark-style.

    ``Alias`` and ``AttributeReference`` carry an explicit name; other nodes fall back to a
    generated label (Spark would generate one too, e.g. ``(a + b)``).
    """
    if isinstance(resolved, Alias):
        return resolved.name
    if isinstance(resolved, AttributeReference):
        return resolved.name
    if isinstance(resolved, BinaryExpression):
        return f"({output_name(resolved.left)} {resolved.op} {output_name(resolved.right)})"
    return repr(resolved)
