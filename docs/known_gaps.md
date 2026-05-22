# Known Parity Gaps

SparkleFrame aims to match PySpark / Spark 4.x behaviour, but some edge cases
are not yet supported.  This page lists the known gaps so contributors can
prioritise fixes and users can work around them.

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
