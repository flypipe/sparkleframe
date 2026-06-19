# Design: AST-based type resolution for the Python engine

> This describes a design for the planned pure-Python engine (`sparkleframe/python`,
> [#102](https://github.com/flypipe/sparkleframe/issues/102)). It is forward-looking and
> will change as the engine lands. It is intentionally kept out of `ARCHITECTURE.md`, which
> maps what exists today. Tracking issue: [#104](https://github.com/flypipe/sparkleframe/issues/104).

## The problem

SparkleFrame's Polars engine makes type-dependent decisions at **expression-build time** — the
moment `col("a") + col("b")` runs. But the information needed to make those decisions (the column
dtypes) only exists **later**, when the expression meets a DataFrame inside `select()` /
`filter()` / `withColumn()`. The schema isn't in hand at build time, so cross-type coercion can't
fire for bare column references.

This single timing mismatch is the root cause behind most entries in
[known_gaps.md](../known_gaps.md). For example, Spark implicitly casts a string operand to numeric:

```python
# Spark: works — casts "3.14" -> 3.14, returns 4.14
spark_df.select((F.col("float_col") + F.col("string_col")).alias("r"))

# Polars engine: raises at evaluation time — the dtypes weren't known when `+` was built,
# so the coercion logic never ran.
sparkle_df.select((PF.col("float_col") + PF.col("string_col")).alias("r"))
```

## The design — build → analyze → evaluate

The Python engine should mirror Spark's Catalyst flow and split expression handling into three
phases instead of deciding types eagerly:

1. **Build** — constructing `col("a") + col("b")` produces an **unresolved AST node**. Zero type
   decisions are made; the node just records "add these two sub-expressions."
2. **Analyze** — when the expression meets a schema-bearing DataFrame (in `select`/`filter`/
   `withColumn`), walk the tree top-down, resolve each node's output type from the schema, and
   apply Spark's coercion rules (e.g. string + numeric → cast string to numeric).
3. **Evaluate** — run the now fully-resolved plan against the data.

## Why it works by construction

Because type resolution **only** happens in the analyze phase — which by definition has the schema
in hand — the "dtype is unknown" state that breaks the Polars engine **cannot occur**. The gaps in
`known_gaps.md` that stem from build-time resolution are eliminated structurally, not patched case
by case.

This is also the Python engine's path to **draining `PYTHON_NOT_IMPLEMENTED`**: each behavior the
analyze/evaluate phases correctly reproduce lets its feature id be removed from the gate (see
[engines.md](../contributing/engines.md) for the ratchet workflow).

## Scope notes

- The canonical engine name is **`python`** (package `sparkleframe/python/`). Issues #102/#104 and
  the `feature/pythondf-backend` branch use `pythondf`; reconcile those to `python`.
- Out of scope for the first pass (per #102): distributed execution, performance optimization, and
  100% API coverage. The pure-Python engine doubles as a readable reference implementation of
  expected PySpark behavior.

## Related

- [#104](https://github.com/flypipe/sparkleframe/issues/104) — this design
- [#102](https://github.com/flypipe/sparkleframe/issues/102) — the Python engine
- [known_gaps.md](../known_gaps.md) — the gaps this design removes
- [contributing/engines.md](../contributing/engines.md) — engines + the parity gate
