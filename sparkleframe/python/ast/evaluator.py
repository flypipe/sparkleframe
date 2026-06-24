"""Evaluate phase: run a resolved :class:`~sparkleframe.python.ast.expressions.Expression`
against data.

This is the final phase of build → analyze → evaluate (see ``docs/design/python-engine-ast.md``).
The evaluator consumes an expression already resolved by :mod:`sparkleframe.python.ast.analyzer`
(so every node has a known ``data_type``) and produces the resulting column values.

:func:`evaluate` works **column-wise**: given the frame's rows (as dicts) it returns one value
per row for the expression. It dispatches on node kind plus ``op`` and computes Python values
directly.

This is the seam where per-feature execution lands. Implementing a feature here (plus
:mod:`sparkleframe.python.ast.analyzer`) is what lets its id be deleted from
``PYTHON_NOT_IMPLEMENTED`` in ``sparkleframe/tests/parity/gaps.py``.
"""

from __future__ import annotations

from typing import Any, List

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    Cast,
    Expression,
    Literal,
)

try:  # pragma: no cover - exercised only with the real pyspark installed
    from pyspark.sql.types import (
        ByteType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
    )
except Exception:  # pragma: no cover - mock pyspark (under activate) has no real types
    pass


def evaluate(expr: Expression, rows: List[dict]) -> List[Any]:
    """Run a resolved ``expr`` against ``rows`` and return one value per row.

    Args:
        expr: An expression already resolved by :func:`sparkleframe.python.ast.analyzer.analyze`.
        rows: The frame's rows as dicts keyed by column name.
    """
    if isinstance(expr, AttributeReference):
        return [row[expr.name] for row in rows]

    if isinstance(expr, Literal):
        return [expr.value for _ in rows]

    if isinstance(expr, Alias):
        return evaluate(expr.child, rows)

    if isinstance(expr, Cast):
        return [_cast_value(v, expr.target_type) for v in evaluate(expr.child, rows)]

    if isinstance(expr, BinaryExpression) and expr.op in ("+", "-", "*"):
        left = evaluate(expr.left, rows)
        right = evaluate(expr.right, rows)
        return [_arithmetic(expr.op, lv, rv) for lv, rv in zip(left, right)]

    detail = f"{type(expr).__name__} {getattr(expr, 'op', '')}".strip()
    not_implemented_yet(f"the evaluate phase for {detail}")


def _arithmetic(op: str, left: Any, right: Any) -> Any:
    """Apply ``+ - *`` element-wise; any null operand yields null (Spark semantics)."""
    if left is None or right is None:
        return None
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    return left * right


def _cast_value(value: Any, target_type: Any) -> Any:
    """Cast a single Python value to ``target_type`` (a PySpark ``DataType``).

    Covers the casts this slice inserts: string → double (via ``float``) and numeric → numeric.
    A null stays null. ANSI-strict behavior comes for free — ``float("x")`` raises on bad input.
    """
    if value is None:
        return None
    if isinstance(target_type, (FloatType, DoubleType)):
        return float(value)
    if isinstance(target_type, (ByteType, ShortType, IntegerType, LongType)):
        return int(value)
    if isinstance(target_type, StringType):
        return str(value)
    not_implemented_yet(f"casting to {type(target_type).__name__}")
