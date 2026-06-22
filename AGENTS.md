# SparkleFrame — agent guide

> This is the lean entry point. It states what is *always* true and points to the
> detailed guides. Read the linked guide before doing work in its area.
> (`CLAUDE.md` is a symlink to this file — one source of truth.)

## What SparkleFrame is

SparkleFrame implements the **PySpark DataFrame API** so pipelines run **without a Spark
cluster**. It is a **partial shim** targeting the **PySpark / Apache Spark 4.x** DataFrame API:
not every operation is supported, but behavior matches Spark where implemented.

It runs on a **pluggable engine**. Polars is the engine today (`sparkleframe/polarsdf`); a
pure-Python engine is planned (`sparkleframe/python`, #102). The engine is selected via the
`Engine` enum and bound by `activate()`. See [ARCHITECTURE.md](ARCHITECTURE.md) for the map.

## Hard invariants (always true — don't violate without explicit instruction)

- **Spark 4 only.** Do not preserve Spark 3 semantics, compatibility shims, or pre–Spark 4
  lenient defaults unless Spark 4 still defines them.
- **Engine-agnostic by layer.** Engine implementations live under `sparkleframe/<engine>/`
  (e.g. `polarsdf`). Public API modules (`functions.py`, `column.py`, `dataframe.py`) are thin;
  shared/private logic goes in the matching `*_helpers.py`. Never bake one engine's assumptions
  into the shared, test, or docs layers — the parity harness reaches engines only through the
  `EngineAdapter` / `Engine` enum.
- **ANSI-strict by default.** Spark 4 sets `spark.sql.ansi.enabled=true`: malformed casts/parses
  **raise**, they don't silently null. Implement strict and lenient (`try_*`) APIs as a shared
  helper with a `strict` flag. **Never implement a strict API as a one-line delegate to its
  `try_*` sibling.** See [docs/contributing/spark4-semantics.md](docs/contributing/spark4-semantics.md).
- **Parity tests are mandatory.** Every new method on **functions**, **column**, or **dataframe**
  needs **Spark parity tests** that exercise **valid AND malformed** inputs. See
  [docs/contributing/testing.md](docs/contributing/testing.md).
- **Always run tests in Docker** via `make test` (not bare `pytest`) — the container sets env
  vars (e.g. `TIMEZONE`) the Spark fixture depends on. Validate with `make black` and `make lint`.
  The full pre-PR gate is `make pr-check`.

## Local development essentials

- Type-hint all parameters and return types. No dead code or dead methods.
- Run affected tests: `make test f=sparkleframe/polarsdf/functions_test.py`.
- Run the whole PR gate before finishing: `make pr-check` (black + lint + coverage).

## Guide map

| When you're… | Read |
|---|---|
| Orienting / need the big picture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Adding or changing an API | [docs/contributing/adding-a-feature.md](docs/contributing/adding-a-feature.md) |
| Writing tests | [docs/contributing/testing.md](docs/contributing/testing.md) |
| Touching cast/parse/strict-vs-lenient | [docs/contributing/spark4-semantics.md](docs/contributing/spark4-semantics.md) |
| Working on engines / the parity gate | [docs/contributing/engines.md](docs/contributing/engines.md) |
| Curious why some edge cases fail | [docs/known_gaps.md](docs/known_gaps.md) |
| Designing the Python engine | [docs/design/python-engine-ast.md](docs/design/python-engine-ast.md) (#104) |
