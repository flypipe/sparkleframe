"""Per-engine "not supported yet" gates for the shared parity suite.

A parity test is tagged with ``@pytest.mark.feature("<id>")``. If that id is in an
engine's gate set, the test is ``xfail`` for that engine (it may fail without
turning CI red). When an engine implements the behavior, **delete the id from its
set in the same change** — the matching parity tests then become required-to-pass.

This is the TDD ratchet: the Python set starts large (the engine is empty) and
drains to zero as the engine lands; deleting an entry is how a feature gets
"promoted" from optional to required. Each entry that the Python set still holds
but the Polars set does not is a polars gap the Python engine is expected to fix.
"""

from __future__ import annotations

# Polars engine — documented in docs/known_gaps.md. Type coercion is decided at
# expression-build time without a schema, so these cross-type cases raise instead
# of coercing the way Spark does.
POLARS_NOT_SUPPORTED = {
    "arithmetic.numeric_plus_string",
    "arithmetic.int_plus_date",
}

# Python engine — nothing is implemented yet, so every behavior under test is
# gated. Seed this from the parity tests as they are written; drain it as the
# build → analyze → evaluate engine lands (#102).
PYTHON_NOT_IMPLEMENTED = {
    "arithmetic.same_type",
    "arithmetic.numeric_plus_string",
    "arithmetic.int_plus_date",
}

GATES = {
    "polars": POLARS_NOT_SUPPORTED,
    "python": PYTHON_NOT_IMPLEMENTED,
}

# Ratchet: the Python gate may only shrink. This is a committed high-water mark —
# lower it (never raise it) when you delete entries. The ratchet test enforces
# that the live set never exceeds this number, so a regression can't quietly
# re-gate a behavior instead of being fixed.
PYTHON_GATE_HIGH_WATER_MARK = 3
