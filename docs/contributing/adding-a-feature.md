# Adding or changing an API

The end-to-end checklist for adding a `functions` / `column` / `dataframe` method (or changing an
existing one). It ties together the other guides; follow the links for detail.

## 1. Locate the right layer

- Implementation goes in the engine package under `sparkleframe/<engine>/` (today: `polarsdf`).
- Public modules (`functions.py`, `column.py`, `dataframe.py`) are **thin facades**. Put shared or
  private logic in the matching `*_helpers.py` (`functions_helpers.py`, `column_helpers.py`,
  `dataframe_helpers.py`). See [ARCHITECTURE.md](../../ARCHITECTURE.md) for the layering.

## 2. Implement to Spark 4 semantics

- Match **Spark 4.x** behavior. Do not preserve Spark 3 semantics or pre–Spark 4 lenient defaults.
- For cast/parse/conversion, follow the strict/lenient pattern: one shared helper with a `strict`
  flag; strict raises, lenient returns `null`. **Never** implement a strict API as a delegate to its
  `try_*` sibling. See [spark4-semantics.md](spark4-semantics.md).
- Add type hints on all parameters and return types. Leave no dead code.

## 3. Write parity tests (valid AND malformed)

- Every new `functions` / `column` / `dataframe` method needs **Spark parity tests** in the matching
  `*_test.py`, asserting against real PySpark with `assert_matches_spark`.
- Fixtures must include **malformed / edge-case inputs**, not just well-formed data — otherwise
  strict error paths never run. For strict/lenient pairs, cover both the raising and the
  null-returning sides. Full patterns and the `engine` fixture vs `ENGINES[Engine.POLARS]` choice are
  in [testing.md](testing.md).

## 4. Update the gate if you promoted a behavior

- If a parity test now passes on an engine that previously gated it, delete its
  `@pytest.mark.feature` id from that engine's set in `gaps.py` and lower
  `PYTHON_GATE_HIGH_WATER_MARK` accordingly — **in the same commit**. See [engines.md](engines.md).
- If you changed Polars behavior that's documented in [../known_gaps.md](../known_gaps.md), update
  that doc too.

## 5. Verify — Definition of Done

Run everything in Docker (the container provides env the Spark fixture needs):

```bash
make black                                   # format
make lint                                    # ruff
make test f=<the test file(s) you touched>   # fast inner loop
make pr-check                                # black + lint + coverage — the gate
```

A change is **done** when:

- [ ] Logic lives in the right layer (facade thin, helper holds the logic).
- [ ] Spark 4 semantics; strict/lenient via a shared `strict` flag, no `try_*` delegation.
- [ ] Parity tests cover **valid and malformed** inputs and pass against real Spark.
- [ ] `gaps.py` and `PYTHON_GATE_HIGH_WATER_MARK` updated if a behavior was promoted;
      `known_gaps.md` updated if a documented Polars gap changed.
- [ ] `make pr-check` is green.
- [ ] Type hints present; no dead code; docs touched if behavior/coverage changed.
