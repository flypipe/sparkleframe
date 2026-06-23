# Architecture

> A bird's-eye codemap of SparkleFrame: where things live and how the coarse pieces
> relate. It names files and types but does not link them — use symbol search to jump
> to anything named here. It is revisited a few times a year, not kept in lock-step with
> the code, so it deliberately avoids line numbers and per-module implementation detail.
> For *how to work in* an area, follow the guides under `docs/contributing/`.

## The problem

PySpark's DataFrame API is the lingua franca for data pipelines, but running it requires a
Spark/JVM cluster — heavy to start, slow for small or low-latency recomputation. SparkleFrame
re-implements that API surface so the **same PySpark code** runs on a lightweight local
**engine** instead. It is a partial shim aimed at Spark 4.x: behavior matches Spark where
implemented, verified against real Spark.

## Codemap

### Engine selection — the outermost seam

`sparkleframe/engine.py` defines `Engine`, the enum of engines. Each member carries a package
name and a class-name prefix (`POLARS = ("polarsdf", "Polars")`, `PYTHON = ("python", "Python")`).
`Engine` is re-exported from the package root (`sparkleframe/__init__.py`).

`sparkleframe/activate.py` is how `import pyspark` starts resolving to SparkleFrame. `activate()`
installs a mock `pyspark` into `sys.modules`, then rebinds `pyspark.sql` (and submodules like
`pyspark.sql.functions`) to the selected engine's package, stripping the engine's class prefix
(`PolarsDataFrame` → `DataFrame`). `deactivate()` restores the real modules; `activate_context()`
is the scoped form. Users can also import an engine package directly and skip activation entirely.

### An engine package — the API surface

An engine lives under `sparkleframe/<engine>/`. Polars is the only one implemented today
(`sparkleframe/polarsdf`); `sparkleframe/python` is scaffolded (#102). Within a package, the public
modules are **thin facades** and all real logic sits in a matching `*_helpers.py`. That split is
an invariant, not a style preference (see Invariants below).

| File (in `sparkleframe/polarsdf/`) | Responsibility |
|---|---|
| `__init__.py` | Exports `Column`, `DataFrame`, `SparkSession`, `Window`, `WindowSpec`, `functions` |
| `session.py` | `SparkSession` — builder + `createDataFrame`; the entry point |
| `dataframe.py` | `DataFrame` — wraps one engine frame; `select`/`filter`/`join`/`groupBy`/… |
| `dataframe_helpers.py` | Join mapping, expression-join execution, value conversion |
| `column.py` | `Column` — wraps one engine expression; arithmetic, comparison, cast, getItem |
| `column_helpers.py` | Type validation/coercion, comparison guards, strict/lenient cast helpers |
| `functions.py` | `pyspark.sql.functions` surface (`col`, `lit`, `to_timestamp`, `element_at`, …) |
| `functions_helpers.py` | Date/time parsing, JSON/struct, array/map helpers — each with a `strict` flag |
| `group.py` | `GroupedData` — aggregations after `groupBy` |
| `window.py` | `Window` / `WindowSpec` — partition/order/frame for `over()` |
| `types.py` | DataType hierarchy + `StructType`/`ArrayType`/`MapType` and name↔type registry |
| `types_utils.py` | Map↔Struct materialization, schema-cast application |

`sparkleframe/base/dataframe.py` holds the engine-independent `DataFrame` base (e.g.
`to_native_df`).

### How a call flows

`Column` wraps a **single** engine expression; `DataFrame` wraps a **single** engine frame. A
PySpark call like `col("a") + col("b")` builds a `Column` around the engine's expression; the
public operator delegates to a `column_helpers` routine that applies Spark's type rules, and the
result is evaluated when the `DataFrame` operation (`select`/`filter`/`withColumn`) runs. This
wrapper pair is the **seam where Spark semantics meet engine primitives** — it is where Spark
coercion and ANSI-strictness are imposed on top of the underlying engine.

### The parity harness — proving it matches Spark

`sparkleframe/tests/parity/` is engine-independent and is the second half of the architecture.

- `engines.py` — `EngineAdapter`, the *only* place tests touch a concrete engine, plus the
  `ENGINES` dict keyed by `Engine`. An engine that isn't scaffolded yet is `None`.
- `oracle.py` — `assert_matches_spark`, the single definition of "does this match real Spark?"
  It reads results through the adapter, normalizes, and compares (multiset by default).
- `conftest.py` — the `engine` fixture parametrizes every parity test over all engines; gated
  behaviors `xfail` and unscaffolded engines skip.
- `gaps.py` — per-engine "not supported yet" gate sets and the `PYTHON_GATE_HIGH_WATER_MARK`
  ratchet; `gate_ratchet_test.py` enforces that the gate only shrinks.

The mechanics and the gate-shrinking workflow live in `docs/contributing/engines.md` and
`docs/contributing/testing.md`.

## Codemap diagram

```mermaid
flowchart TD
    user["user code: import pyspark.sql / sparkleframe.&lt;engine&gt;"]
    activate["activate.py — rebinds sys.modules to the chosen engine"]
    engineenum["engine.py — Engine enum (POLARS, PYTHON)"]
    pkg["sparkleframe/&lt;engine&gt;/ (polarsdf today, python scaffolded)"]
    facade["public facades: functions.py / column.py / dataframe.py"]
    helpers["*_helpers.py — coercion, parsing, strict/lenient logic"]
    prim["engine primitives (Column wraps one expr, DataFrame one frame)"]

    user --> activate --> engineenum
    activate --> pkg --> facade --> helpers --> prim

    subgraph parity["tests/parity (engine-independent)"]
      test["*_parity_test.py"] --> adapter["EngineAdapter / ENGINES"]
      adapter --> pkg
      test --> oracle["oracle.assert_matches_spark"]
      oracle --> spark["real PySpark (truth)"]
      gaps["gaps.py gates + ratchet"] -.-> test
    end
```

## Invariants (what must stay true)

- **The shared, test, and oracle layers never import a concrete engine package directly.** They
  reach an engine only through `EngineAdapter` / the `Engine` enum. This is what keeps a second
  engine viable. (Noted exception: the oracle currently borrows normalization helpers from
  `tests/utils`, which transitively touch `polarsdf` — see the comment in `oracle.py`.)
- **Public modules carry no engine-coercion logic inline.** `functions`/`column`/`dataframe` are
  facades; coercion, parsing, and validation live in `*_helpers.py`. The helper is the boundary.
- **No Spark 3 semantics.** ANSI-strict is the default; strict APIs never delegate to their
  `try_*` siblings.
- **The parity gate only shrinks.** A behavior is "promoted" from optional to required by deleting
  its id from a gate set — never by re-gating a regression.

## Cross-cutting concerns

- **Spark 4 ANSI strict vs lenient.** Strict/lenient pairs share one helper with a `strict` flag.
  The rules and the `element_at`/`try_element_at` map-key subtlety (SPARK-48766) are in
  `docs/contributing/spark4-semantics.md`.
- **Type coercion timing — the present-day constraint.** The Polars engine resolves column dtypes
  at *expression-build time*, before the frame schema is known. Cross-type coercion (e.g.
  `col(float) + col(string)`) therefore can't fire for bare column references, which is the root
  cause behind most of `docs/known_gaps.md`. Engines are free to resolve types differently behind
  the `EngineAdapter` / `Engine` boundary. The scaffolded `python` engine resolves this differently —
  see `docs/design/python-engine-ast.md` and #104. (No design detail here, by intent.)
- **Multiple engines.** The canonical name for the scaffolded pure-Python engine is **`python`**
  (package `sparkleframe/python/`), consistent across `engine.py`, `gaps.py`, and the docs
  generator. Note: issues #102/#104 and the `feature/pythondf-backend` branch use `pythondf`; those
  external references should be reconciled to `python`.
