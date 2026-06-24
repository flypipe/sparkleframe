"""Unresolved expression AST for the pure-Python engine — the **build** phase.

This module is the backbone of the build → analyze → evaluate flow described in
``docs/design/python-engine-ast.md``. Constructing an expression (e.g. ``col("a") + lit(1)``)
produces an unresolved AST node here and makes **zero type decisions** — the schema is not
known at build time. Type resolution happens later in :mod:`sparkleframe.python.ast.analyzer`
(analyze), and execution in :mod:`sparkleframe.python.ast.evaluator` (evaluate).

The node classes are deliberately generic (Catalyst-style): a small set covers the whole
public API surface, so the analyzer and evaluator dispatch on the node kind plus an
``op`` / ``name`` string rather than needing one class per operator or function.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


class Expression:
    """Base class for every unresolved AST node.

    Pure structure: holds child sub-expressions and, after the analyze phase, an optional
    resolved :attr:`data_type`. The **build** phase must never set :attr:`data_type` — that
    is the analyzer's job once a schema is in hand (see the design doc).
    """

    def __init__(self, children: Optional[List["Expression"]] = None) -> None:
        self.children: List[Expression] = list(children) if children else []
        # Filled in by analyzer.analyze(); ``None`` means "not resolved yet".
        self.data_type: Any = None

    def __repr__(self) -> str:
        inner = ", ".join(repr(c) for c in self.children)
        return f"{type(self).__name__}({inner})"


class Literal(Expression):
    """A constant, e.g. ``lit(1)`` or ``lit(None)``."""

    def __init__(self, value: Any, data_type: Any = None) -> None:
        super().__init__()
        self.value = value
        # A literal's type may be supplied at build time (e.g. a typed null); otherwise the
        # analyzer infers it from ``value``.
        self.data_type = data_type

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


class AttributeReference(Expression):
    """An unresolved column reference, e.g. ``col("x")``."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"AttributeReference({self.name!r})"


class BinaryExpression(Expression):
    """A two-operand op: arithmetic / comparison / logical.

    ``op`` is the operator token, e.g. ``"+"``, ``"-"``, ``"*"``, ``"/"``, ``"**"``,
    ``"=="``, ``"!="``, ``"<"``, ``"<="``, ``">"``, ``">="``, ``"and"``, ``"or"``.
    """

    def __init__(self, op: str, left: Expression, right: Expression) -> None:
        super().__init__([left, right])
        self.op = op

    @property
    def left(self) -> Expression:
        return self.children[0]

    @property
    def right(self) -> Expression:
        return self.children[1]

    def __repr__(self) -> str:
        return f"BinaryExpression({self.op!r}, {self.left!r}, {self.right!r})"


class UnaryExpression(Expression):
    """A single-operand op, e.g. ``"not"``, ``"neg"``, ``"pos"``, ``"isnull"``, ``"isnotnull"``."""

    def __init__(self, op: str, child: Expression) -> None:
        super().__init__([child])
        self.op = op

    @property
    def child(self) -> Expression:
        return self.children[0]

    def __repr__(self) -> str:
        return f"UnaryExpression({self.op!r}, {self.child!r})"


class Cast(Expression):
    """A type cast. ``strict=True`` is ANSI ``cast`` (raises on bad input); ``strict=False`` is ``try_cast``.

    Keeping the strict flag on the node (rather than implementing one as a delegate to the
    other) honors the ANSI strict-vs-lenient invariant in ``CLAUDE.md``.
    """

    def __init__(self, child: Expression, target_type: Any, strict: bool) -> None:
        super().__init__([child])
        self.target_type = target_type
        self.strict = strict

    @property
    def child(self) -> Expression:
        return self.children[0]

    def __repr__(self) -> str:
        return f"Cast({self.child!r}, {self.target_type!r}, strict={self.strict})"


class FunctionCall(Expression):
    """A named function applied to argument expressions, e.g. ``abs(col("x"))`` -> ``FunctionCall("abs", [...])``."""

    def __init__(self, name: str, args: List[Expression]) -> None:
        super().__init__(list(args))
        self.name = name

    @property
    def args(self) -> List[Expression]:
        return self.children

    def __repr__(self) -> str:
        inner = ", ".join(repr(a) for a in self.args)
        return f"FunctionCall({self.name!r}, [{inner}])"


class CaseWhen(Expression):
    """A ``when(...).when(...).otherwise(...)`` chain.

    ``branches`` is a list of ``(condition, value)`` expression pairs; ``otherwise`` is the
    fallback expression or ``None``.
    """

    def __init__(
        self,
        branches: List[Tuple[Expression, Expression]],
        otherwise: Optional[Expression] = None,
    ) -> None:
        flat: List[Expression] = []
        for cond, val in branches:
            flat.extend([cond, val])
        if otherwise is not None:
            flat.append(otherwise)
        super().__init__(flat)
        self.branches = list(branches)
        self.otherwise = otherwise

    def __repr__(self) -> str:
        parts = ", ".join(f"({c!r} -> {v!r})" for c, v in self.branches)
        return f"CaseWhen([{parts}], otherwise={self.otherwise!r})"


class Alias(Expression):
    """A renamed expression, e.g. ``col("x").alias("y")``."""

    def __init__(self, child: Expression, name: str) -> None:
        super().__init__([child])
        self.name = name

    @property
    def child(self) -> Expression:
        return self.children[0]

    def __repr__(self) -> str:
        return f"Alias({self.child!r}, {self.name!r})"
