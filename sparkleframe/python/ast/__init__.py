"""The AST core of the pure-Python engine: build → analyze → evaluate.

This sub-package is the engine's heart, kept separate from the public API surface
(``column``/``functions``/``dataframe``). It holds the unresolved expression nodes produced by
the **build** phase (:mod:`~sparkleframe.python.ast.expressions`), the **analyze** phase that
resolves types against a schema (:mod:`~sparkleframe.python.ast.analyzer`), and the
**evaluate** phase that runs a resolved expression (:mod:`~sparkleframe.python.ast.evaluator`).
See ``docs/design/python-engine-ast.md``.
"""

# ruff: noqa: F401
from sparkleframe.python.ast.analyzer import analyze
from sparkleframe.python.ast.evaluator import evaluate
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    CaseWhen,
    Cast,
    Expression,
    FunctionCall,
    Literal,
    UnaryExpression,
)

__all__ = [
    "analyze",
    "evaluate",
    "Alias",
    "AttributeReference",
    "BinaryExpression",
    "CaseWhen",
    "Cast",
    "Expression",
    "FunctionCall",
    "Literal",
    "UnaryExpression",
]
