"""Shared 'not implemented yet' helper for the pure-Python engine scaffolding.

The engine is a walking skeleton: the build phase (constructing the expression AST) is wired,
but the analyze and evaluate phases — and every DataFrame action — raise. Use this helper so
the message is consistent and always points a future implementer at the design doc and the
build → analyze → evaluate flow.
"""

from __future__ import annotations

from typing import NoReturn


def not_implemented_yet(what: str) -> NoReturn:
    """Raise a uniform ``NotImplementedError`` for an unimplemented engine behavior.

    Args:
        what: A short feature id, e.g. ``"DataFrame.select"`` or ``"functions.split"``.
    """
    raise NotImplementedError(
        f"The pure-Python engine does not implement {what} yet. Build the AST node here, then "
        f"implement its analyze (type resolution) and evaluate (execution) phases, and drain the "
        f"matching id from PYTHON_NOT_IMPLEMENTED in sparkleframe/tests/parity/gaps.py. "
        f"See docs/design/python-engine-ast.md."
    )
