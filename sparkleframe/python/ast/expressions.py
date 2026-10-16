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

from typing import Any, Callable, Dict, List, Optional, Tuple


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
    """A column reference, e.g. ``col("x")``, or ``df["x"]`` qualified by the frame's ``plan_id``.

    The analyzer resolves it to a position in the frame's row tuples (:attr:`index`).
    """

    def __init__(self, name: str, plan_id: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.plan_id = plan_id
        self.index: Optional[int] = None

    def __repr__(self) -> str:
        return f"AttributeReference({self.name!r})"


class Star(Expression):
    """``col("*")`` / ``select("*")`` — expands to every column of the frame."""


class BinaryExpression(Expression):
    """A two-operand op: arithmetic / comparison / logical.

    ``op`` is the operator token, e.g. ``"+"``, ``"-"``, ``"*"``, ``"/"``, ``"**"``,
    ``"=="``, ``"!="``, ``"<"``, ``"<="``, ``">"``, ``">="``, ``"<=>"``, ``"and"``, ``"or"``.
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
    """A named function applied to argument expressions, e.g. ``abs(col("x"))`` -> ``FunctionCall("abs", [...])``.

    ``options`` carries the non-expression parameters a function takes (a regex pattern, a date
    format, a ``from_json`` schema, ...), so they never masquerade as columns.
    """

    def __init__(self, name: str, args: List[Expression], options: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(list(args))
        self.name = name
        self.options: Dict[str, Any] = dict(options) if options else {}

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


class SortOrder(Expression):
    """A sort key: ``col.asc()``, ``col.desc_nulls_last()``, ... (only meaningful in ``orderBy`` / windows)."""

    def __init__(self, child: Expression, ascending: bool, nulls_first: bool) -> None:
        super().__init__([child])
        self.ascending = ascending
        self.nulls_first = nulls_first

    @property
    def child(self) -> Expression:
        return self.children[0]

    def __repr__(self) -> str:
        return f"SortOrder({self.child!r}, ascending={self.ascending}, nulls_first={self.nulls_first})"


class WindowExpression(Expression):
    """``function.over(window)``: a ranking function evaluated over partitioned, ordered rows."""

    def __init__(self, function: Expression, partition_by: List[Expression], order_by: List[SortOrder]) -> None:
        super().__init__([function, *partition_by, *order_by])
        self.function = function
        self.partition_by = list(partition_by)
        self.order_by = list(order_by)

    def __repr__(self) -> str:
        return f"WindowExpression({self.function!r}, partition_by={self.partition_by!r}, order_by={self.order_by!r})"


class LambdaVariable(Expression):
    """A lambda parameter (the ``x`` in ``transform(col, lambda x: x + 1)``).

    The analyzer types it from the enclosing higher-order function and assigns its :attr:`slot`
    in the lambda scope the evaluator keeps per row.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.slot: Optional[int] = None

    def __repr__(self) -> str:
        return f"LambdaVariable({self.name!r})"


class LambdaFunction(Expression):
    """A Python callable passed to a higher-order function; its body is built at analyze time.

    The body can only be built once the parameter types are known, so the build phase keeps the
    callable and the analyzer calls it with :class:`LambdaVariable` columns.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        super().__init__()
        self.func = func

    def __repr__(self) -> str:
        return f"LambdaFunction({getattr(self.func, '__name__', 'lambda')})"
