"""Analyze phase: resolve an unresolved :class:`~sparkleframe.python.ast.expressions.Expression`
against a schema.

This is the middle phase of build → analyze → evaluate (see ``docs/design/python-engine-ast.md``).
Because resolution happens *here* — where the schema is in hand — the "dtype unknown at build
time" state that limits the Polars engine cannot occur.

:func:`analyze` walks the unresolved tree and returns a **new** tree of the same node classes
with each node's ``data_type`` slot filled and coercion :class:`Cast` nodes inserted where Spark
would. The original build tree is left untouched; "resolved" means every returned node has
``data_type is not None``.

This is the seam where per-feature type resolution lands. Implementing a feature here (plus
:mod:`sparkleframe.python.ast.evaluator`) is what lets its id be deleted from
``PYTHON_NOT_IMPLEMENTED`` in ``sparkleframe/tests/parity/gaps.py``.
"""

from __future__ import annotations

from typing import Any

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast import coercion
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    Cast,
    Expression,
    Literal,
)


def analyze(expr: Expression, schema: Any) -> Expression:
    """Resolve ``expr``'s output types against ``schema`` and return the resolved tree.

    Args:
        expr: The unresolved expression produced by the build phase.
        schema: The PySpark ``StructType`` of the frame the expression is applied to.
    """
    if isinstance(expr, AttributeReference):
        resolved = AttributeReference(expr.name)
        resolved.data_type = _field_type(schema, expr.name)
        return resolved

    if isinstance(expr, Literal):
        return Literal(expr.value, data_type=expr.data_type or coercion.infer_literal_type(expr.value))

    if isinstance(expr, Alias):
        child = analyze(expr.child, schema)
        resolved = Alias(child, expr.name)
        resolved.data_type = child.data_type
        return resolved

    if isinstance(expr, BinaryExpression) and expr.op in ("+", "-", "*"):
        left = analyze(expr.left, schema)
        right = analyze(expr.right, schema)
        result_dt, left_cast, right_cast = coercion.coerce_arithmetic(expr.op, left.data_type, right.data_type)
        resolved = BinaryExpression(expr.op, _with_cast(left, left_cast), _with_cast(right, right_cast))
        resolved.data_type = result_dt
        return resolved

    detail = f"{type(expr).__name__} {getattr(expr, 'op', '')}".strip()
    not_implemented_yet(f"the analyze phase for {detail}")


def _with_cast(resolved: Expression, target_dt: Any) -> Expression:
    """Wrap ``resolved`` in a strict :class:`Cast` to ``target_dt`` (or return it unchanged)."""
    if target_dt is None:
        return resolved
    cast = Cast(resolved, target_dt, strict=True)
    cast.data_type = target_dt
    return cast


def _field_type(schema: Any, name: str) -> Any:
    """Look up a column's declared PySpark ``DataType`` in the frame schema."""
    if schema is None:
        raise KeyError(f"Cannot resolve column {name!r}: no schema in hand")
    return schema[name].dataType
