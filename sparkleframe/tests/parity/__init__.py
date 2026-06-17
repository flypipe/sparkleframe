"""Shared, engine-parametrized Spark-parity harness.

Both engines (Polars and the upcoming pure-Python engine) exist to mimic the
*same* spec: PySpark 4. So a parity case — "this expression must equal what real
Spark returns" — is engine-independent and should be written **once** and run
against **every** engine.

This package provides:

- ``engines`` — a small adapter per engine (build a frame, extract records,
  expose the ``functions`` module). The Python adapter is ``None`` until the
  ``sparkleframe.python`` package is scaffolded, so its parametrization skips.
- ``oracle`` — one engine-agnostic comparison against real Spark, reusing the
  existing normalization in ``sparkleframe.tests.utils`` (single source of truth
  for *how* we compare).
- ``gaps`` — per-engine "not supported yet" sets. A parity test for a gated
  behavior is ``xfail`` for that engine; deleting an entry (when the feature
  lands) makes the matching tests required-to-pass.
"""
