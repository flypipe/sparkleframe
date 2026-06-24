# Engines and the parity gate

SparkleFrame runs the PySpark API on a **pluggable engine**. This guide covers the engine
abstraction and the TDD "gaps ratchet" that tracks per-engine coverage. For the big-picture map,
see [ARCHITECTURE.md](../../ARCHITECTURE.md).

## The `Engine` enum

`sparkleframe/engine.py` declares the engines. Each member is `(module, class_prefix)`:

```python
class Engine(Enum):
    POLARS = ("polarsdf", "Polars")
    PYTHON = ("python", "Python")
```

- `module` is the package under `sparkleframe/` that implements the engine.
- `class_prefix` is stripped when binding to the `pyspark` namespace (`PolarsDataFrame` →
  `DataFrame`), via `clean_class_name`.

`activate(engine=Engine.POLARS)` (in `sparkleframe/activate.py`) imports the engine's package and
rebinds `sys.modules["pyspark.sql"]` (and submodules) to it, so existing `import pyspark` code runs
on the engine. Selecting an engine that isn't scaffolded raises `NotImplementedError`.

The **Polars** engine (`sparkleframe/polarsdf`) is the mature engine. The **`python`** engine
(`sparkleframe/python`) is scaffolded and landing incrementally via its build → analyze → evaluate
flow — the first vertical slice (numeric arithmetic, incl. the `numeric + string` coercion) is in,
and its gate drains as more behaviors land. See [#102](https://github.com/flypipe/sparkleframe/issues/102)
and the design in [../design/python-engine-ast.md](../design/python-engine-ast.md).

## The parity harness

Everything in `sparkleframe/tests/parity/` is engine-independent:

- **`engines.py`** — `EngineAdapter` is the *only* place a test touches a concrete engine. It bundles
  the engine's `functions` module, a `build_df(rows, spark_schema)`, a `to_records`, and a `dtype`
  translator. `ENGINES` is a dict keyed by the `Engine` enum; an engine that isn't scaffolded yet is
  `None`.
- **`oracle.py`** — `assert_matches_spark` is the single definition of "matches real Spark." There
  must be exactly one — duplicating normalization per engine gives false confidence.
- **`conftest.py`** — the `engine` fixture parametrizes every parity test over all engines (`[POLARS]`,
  `[PYTHON]`, …); unscaffolded engines `skip`, gated behaviors `xfail`.

## The gaps ratchet

`sparkleframe/tests/parity/gaps.py` holds a per-engine set of feature ids that the engine **does not
support yet**:

```python
POLARS_NOT_SUPPORTED = { "arithmetic.numeric_plus_string", ... }
PYTHON_NOT_IMPLEMENTED = { ...behaviors not yet drained as the engine lands... }

GATES = {"polars": POLARS_NOT_SUPPORTED, "python": PYTHON_NOT_IMPLEMENTED}

PYTHON_GATE_HIGH_WATER_MARK = 165   # committed high-water mark; only ever lowered
```

A parity test tagged `@pytest.mark.feature("<id>")` is `xfail`ed for an engine when `<id>` is in that
engine's gate set. So a behavior an engine can't do yet fails *without turning CI red*, and a behavior
it *can* do is **required to pass**.

`gate_ratchet_test.py` enforces two things:

- `test_python_gate_only_shrinks` — the gate may not grow (you can't quietly re-gate a regression).
- `test_high_water_mark_is_not_stale` — `PYTHON_GATE_HIGH_WATER_MARK` must equal the current gate
  size, so the mark drops as the engine progresses.

### Promoting a behavior (the workflow)

When an engine learns to do something, promote it from optional to required **in the same commit**:

1. Make the parity test pass on that engine (implement the behavior in `sparkleframe/<engine>/`).
2. Ensure the test is tagged `@pytest.mark.feature("<id>")` ([testing.md](testing.md) covers id
   naming).
3. **Delete `<id>`** from the engine's gate set in `gaps.py` (e.g. from `PYTHON_NOT_IMPLEMENTED`).
4. If you removed N ids from the Python gate, **lower `PYTHON_GATE_HIGH_WATER_MARK` by N** so
   `test_high_water_mark_is_not_stale` passes.
5. Run `make test f=sparkleframe/tests/parity` — the test now runs as required (no `xfail`).

Going the other direction (adding *new* parity coverage that an engine can't satisfy yet) means
adding ids to a gate and raising the high-water mark deliberately — that's the one time the mark goes
up, and it should be an intentional, reviewed change.

## `known_gaps.md` ↔ `gaps.py`

[../known_gaps.md](../known_gaps.md) is the human-readable explanation of the **Polars** gaps —
root causes and workarounds. `POLARS_NOT_SUPPORTED` in `gaps.py` is the machine-enforced counterpart.
Entries in `PYTHON_NOT_IMPLEMENTED` that are *not* in `POLARS_NOT_SUPPORTED` are behaviors Polars
already handles and the Python engine is expected to reach.

## Adding a new engine (sketch)

1. Create `sparkleframe/<engine>/` mirroring `polarsdf/` (session, dataframe, column, functions,
   types, group, window) with the engine's `class_prefix` on public classes.
2. Add the member to `Engine` and a `{"label", "module"}` entry to `docs/generate_supported_api.py`.
3. Add an adapter in `engines.py` (return `None` until the package exists) and a gate set + the
   `GATES` entry in `gaps.py`.
4. Drain the gate as behaviors land, following the ratchet workflow above.
