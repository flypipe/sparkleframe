# Known Parity Gaps

SparkleFrame aims to match PySpark / Spark 4.x behaviour, but some edge cases
are not yet supported.  This page lists the known gaps so contributors can
prioritise fixes and users can work around them.

> These gaps are specific to the **Polars** engine, whose root cause is build-time
> type resolution (see "Mixed-type column arithmetic" below). The pure-Python engine
> (`sparkleframe/python`, #102) resolves types in an *analyze* phase once the schema is
> known, which removes this whole class structurally — it already handles
> `Numeric + String`. See [design/python-engine-ast.md](design/python-engine-ast.md).

## Mixed-type column arithmetic

**Affected operations:** `+`, `-`, `*`, `/` between two columns of different
types.

SparkleFrame resolves column dtypes at *expression build time* (when `col("a")
+ col("b")` executes).  Because Polars expressions are lazy, the underlying
dtypes are not yet known at that point — the DataFrame schema is only available
later, inside `select()` / `withColumn()` / `filter()`.

This means the cross-type coercion logic (`_coerce_mixed_arithmetic_operands`)
never fires for bare column references, and Polars raises
`InvalidOperationError` at evaluation time instead of coercing.

### Numeric + String (12 test cases)

Spark implicitly casts the string column to a numeric type (Float64) when the
other operand is numeric.  SparkleFrame raises `InvalidOperationError`.

Affected pair labels: `float_x_string`, `double_x_string`, `string_x_decimal`
across all four arithmetic operators.

```python
# Spark: works (casts "3.14" → 3.14, returns 4.14)
spark_df.select((F.col("float_col") + F.col("string_col")).alias("r"))

# SparkleFrame: raises InvalidOperationError
sparkle_df.select((PF.col("float_col") + PF.col("string_col")).alias("r"))
```

### Integer + Date (3 test cases)

Spark treats the integer as a day offset and adds it to the date column.
SparkleFrame raises `InvalidOperationError`.

Affected pair labels: `byte_x_date`, `short_x_date`, `int_x_date` for `+`.

```python
# Spark: works (adds 5 days to the date)
spark_df.select((F.col("int_col") + F.col("date_col")).alias("r"))

# SparkleFrame: raises InvalidOperationError
sparkle_df.select((PF.col("int_col") + PF.col("date_col")).alias("r"))
```

### Root cause

`_resolve_expr_output_dtype` returns `None` for bare `pl.col("name")`
references when no schema context is active.  The coercion code in
`_coerce_mixed_arithmetic_operands` correctly handles both cases but bails out
early when either dtype is `None`.

### Possible fix paths

1. **Schema-aware Columns** — propagate the DataFrame schema into Column
   objects so dtypes are resolvable at build time.  Requires changes to the
   `Column` constructor, `DataFrame.select()`, and anywhere columns are
   created.

2. **Runtime `map_batches` fallback** — when both dtypes are unresolvable,
   wrap the two operands in `pl.struct().map_batches()` to inspect actual
   dtypes at evaluation time and apply coercion.  Must be scoped narrowly to
   the unknown-dtype path to avoid regressions in same-type operations (a
   previous broad attempt was reverted for this reason).

## `transform` with struct-producing lambdas

**Affected operations:** `F.transform(col, lambda x: F.struct(...))`

Polars' `list.eval` does not support `pl.struct` expressions, so SparkleFrame
falls back to `map_batches`.  The batch callback infers the struct dtype at
runtime from the data, and the `return_dtype` hint is constructed from the
struct field names (defaulting unknown types to `Utf8`).

This means:

- **`_struct_parts` metadata propagation is required:** `WhenBuilder`
  (`when/otherwise`) propagates `_struct_parts` from the struct branch so that
  `transform` can build the return dtype.  If this metadata is lost (e.g.
  through a chain not yet covered), the fallback parses `.alias("name")`
  patterns from the expression's string representation.

- **Performance:** `map_batches` processes one row at a time in the struct
  path.  This is acceptable for typical transform+struct patterns (e.g. UTM
  parsing) but will be slower than native `list.eval` for very large datasets.

## `getItem` resolution outside schema context

**Affected operations:** `col("arr").getItem(0)` used inside `filter()` or
other contexts where `to_native()` is called before a schema is available.

SparkleFrame stores `getItem` operations as a lazy chain (`_getitem_chain`)
resolved during `to_native()`.  When `to_native()` runs outside a schema
context (e.g. inside `isNotNull()` before `filter()` evaluates, or when a
function like `size()`/`struct()`/`when()` eagerly resolves a `Column`
argument built from a dotted path such as `col("a.b")` — see
`functions.py`'s ubiquitous `_to_expr(col_name) if isinstance(col_name,
Column) else pl.col(col_name)` pattern), the dtype resolver returns `None`,
forcing a UDF fallback.

The UDF fallbacks (`_extract_by_key`, `_index_at`) handle `pl.Series` inputs
correctly and return the *actual* extracted value (not a stringified copy),
via `map_batches` with no forced `return_dtype` — Polars infers the real
output dtype (`Float64`, `List(Struct(...))`, etc.) from the values returned,
which SparkleFrame's always-eager evaluation model makes safe. This keeps
downstream consumers (`F.size()`, `F.struct()`, `F.when()`, arithmetic, …)
working on the real value instead of silently getting a stringified
`None`/wrong result. The resolution is still less efficient than the native
`list.get` / `struct.field` paths used when dtype is known.

Letting Polars infer the dtype has one edge case: if *every* row in the
batch extracts to `None` (the source container is null in every row — no
real data to infer a dtype from at all), Polars infers `pl.Null` for the
fallback's output Series, and several downstream ops (`.str.*`, `.list.*`, …)
reject `Null` input outright (`SchemaError: expected 'String', got 'null'`).
Both fallbacks special-case this via `_series_from_extracted_values`, casting
an all-null result to `pl.String` — an arbitrary but harmless choice since
every value is null anyway, and it matches what the old (stringifying)
fallback always produced.

## `element_at` with `F.lit(int)` indices

**Affected operations:** `element_at(col, F.lit(n))`, `try_element_at(col, F.lit(n))`

`F.lit(value)` wraps the value in `pl.repeat(value, pl.len())`, which is a
`pl.Expr` rather than a plain Python `int`.  SparkleFrame resolves this by
evaluating the expression against a dummy DataFrame to extract the integer.
This works but adds a small overhead per call.

## `when/otherwise` with `F.lit(None)` type coercion

**Affected operations:** `F.when(cond, expr).otherwise(F.lit(None))`

PySpark infers the result type from the non-null branch; SparkleFrame now
uses `pl.Null` dtype for `lit(None)`, which Polars correctly super-types with
the non-null branch.  This is resolved for standard types but may still
surface for complex nested types not yet tested.

## Equality comparison dtype resolution

**Affected operations:** `==`, `!=` between columns where both dtypes are
unresolvable at expression build time.

SparkleFrame includes a same-family short-circuit (both string, both numeric,
or both boolean) that uses native Polars comparisons.  For cross-type
comparisons (e.g. `col(int) == lit(string)`), the full coercion logic applies,
which may use `map_batches` with a UDF.

## Map type detection

**Affected operations:** all comparisons and arithmetic involving `MapType`
columns.

Polars stores maps as `List(Struct([key, value]))`, which is indistinguishable
from a regular list-of-structs at the dtype level.  SparkleFrame cannot detect
whether a column is a true Spark `MapType` and therefore cannot replicate
Spark's map-specific behaviour (e.g. rejecting all comparisons on maps).

This is a fundamental Polars representation issue with no straightforward fix.

## Ordering on nested types

**Affected operations:** `<`, `<=`, `>`, `>=` on `ArrayType` and `StructType`
columns.

Polars does not implement lexicographic ordering for `List` or `Struct` dtypes.
Spark supports element-wise ordering on arrays and field-wise ordering on
structs, but replicating this would require manually exploding and comparing
elements — a non-trivial implementation.
