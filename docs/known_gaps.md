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
context (e.g. inside `isNotNull()` before `filter()` evaluates), the dtype
resolver returns `None`, forcing a UDF fallback.

The UDF fallbacks (`_extract_by_key`, `_index_at`) now handle `pl.Series`
inputs correctly, but the resolution is less efficient than the native
`list.get` / `struct.field` paths used when dtype is known.

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

## `functions.pow` operand types and Java NaN semantics

**Affected operations:** `F.pow(base, exponent)` (the `Column` `**` / reflected `**` operators
are not affected for operand types — they cast both sides to `Float64` first).

Spark's `pow` casts both operands to `double` before computing, so it accepts any numeric,
decimal, string (ANSI cast, whitespace trimmed) or null operand and always returns `double`.
The Polars engine's `F.pow` hands the raw operands to Polars' `pow`, which:

- rejects a negative integer exponent (`2 ** -1` fails converting `-1` to `u32`) and returns an
  integer dtype for integer operands;
- rejects `Decimal`, `Null` (e.g. `F.lit(None)`) and `String` operands outright.

Separately, Polars follows C99 `pow`, whereas Spark uses Java's `Math.pow`: `1 ** NaN` and
`(±1) ** ±inf` are `NaN` in Spark but `1.0` in Polars.

**Workaround:** cast operands to `double` first, e.g. `F.pow(F.col("a").cast("double"), 2)`.
The pure-Python engine implements all of these (see `spark_pow` in
`sparkleframe/python/functions_helpers.py`).

## String and non-boolean operands in comparisons and logical operators

**Affected operations:** `==`, `!=`, `<`, `<=`, `>`, `>=` with a string operand on one side, and
`&`, `|`, `~` on non-boolean operands.

Spark 4 (ANSI) compares a string with another type by casting the string strictly: to `bigint`
against integral types, to `double` against float/double/decimal, and to the other type against
boolean, date, timestamp and binary. A malformed string raises `CAST_INVALID_INPUT`. This runs
into the same build-time dtype limitation as "Mixed-type column arithmetic" above, so the
Polars engine raises a type mismatch instead of coercing (`"7" == 7`, `"2024-01-15" < date`).

The logical operators accept only boolean operands in Spark. A string operand is cast to boolean
(`"yes"`, `"t"`, `"1"`, … are true), a `null` literal is typed as boolean, and anything else is a
`DATATYPE_MISMATCH` analysis error. The Polars engine rejects string operands and, for integer
operands, silently computes a **bitwise** `and`/`or`/`not` instead of raising.

Arrays and structs with null elements (or NaN inside them) are covered by "Ordering on nested
types".

**Workaround:** cast explicitly, e.g. `F.col("s").cast("long") == F.col("n")` or
`F.col("flag").cast("boolean") & F.col("b")`. The pure-Python engine implements all of these.

## `getItem` with a Column index on arrays

**Affected operations:** `col("arr")[col("i")]`, `col("arr").getItem(col("i"))`.

A Column key is supported on map columns (`create_map(...)[col("status")]`), where it looks the
value up as a map key and returns null when absent, as Spark does. Spark also reads a Column key on
an array as a 0-based index; sparkleframe raises `NotImplementedError` for that case rather than
returning wrong values. Use `element_at(col("arr"), col("i") + 1)` instead.

## Outer struct nulls in `createDataFrame`

**Affected operations:** `spark.createDataFrame(rows, schema)` where a `StructType` column holds
`None`.

The outer null is lost: `None` becomes a struct whose fields are all null, so `isNull()` on that
column returns `False` where Spark returns `True`. Structs produced by `from_json` keep their outer
nulls and are not affected.

## Projection naming and `withColumns` resolution

**Affected operations:** `select(...)` with an unaliased expression; `withColumns({...})` where a later
entry reads a column an earlier entry replaces.

Spark names an unaliased expression after its SQL text (`F.col("salary") * 1.1` is `(salary * 1.1)`).
The Polars engine names it after the root column (`salary`), so the output can even collide with an
existing column name.

`withColumns` evaluates every entry against the **input** frame. A later entry may reference a column
an earlier entry *adds* (lateral column alias, Spark 3.4+). A name that already exists in the input,
though, keeps meaning the input value even if an earlier entry of the same call replaces it:
`withColumns({"x": col("x") + 1, "z": col("x") * 2})` computes `z` from the original `x`. The Polars
engine applies the entries one after another, so `z` sees the new `x`.

**Workaround:** alias expressions explicitly, and split a `withColumns` that reads a column it also
replaces into two calls. The pure-Python engine implements both.

## Joins on same-named Column keys and cross joins

**Affected operations:** `left.join(right, on=F.col("id"))` (or `[F.col("id")]`) when both sides
have `id`; `left.join(right)` without `on`; `crossJoin`.

In Spark, `F.col("id")` in a join condition is an unqualified reference that matches a column on
both sides, so the join fails with `AMBIGUOUS_REFERENCE`. Pass the key as a string (`on="id"`), or
compare qualified columns (`left["id"] == right["id"]`). The Polars engine treats `F.col("id")` as
`on="id"` instead.

A cross join keeps both sides' columns, so the result has two `id` columns: a bare `"id"` is
ambiguous and `left["id"]` / `right["id"]` pick a side. The Polars engine raises on `join` without
`on`, has no `crossJoin`, and cannot tell the two `id` columns apart.

**Workaround:** join on a string key, or rename one side's key before a Column-condition join. The
pure-Python engine implements Spark's behavior.

## `explode` on null, empty and non-collection input

**Affected operations:** `F.explode(col)`.

Spark's `explode` emits no row for a null or empty array (`explode_outer` emits one row with a
null), and rejects a column that is not an array or map. The Polars engine keeps a null row for both
null and empty arrays, i.e. it behaves like `explode_outer`, and does not reject non-collection
input.

**Workaround:** filter `F.size(col) > 0` before exploding. The pure-Python engine implements Spark's
behavior.

## `months_between` rounding

**Affected operations:** `F.months_between(end, start)` (default `roundOff=True`).

Spark rounds the result to 8 decimal places. The Polars engine returns full precision
(`2.16129032258` where Spark returns `2.16129032`).

**Workaround:** `F.round(F.months_between(end, start), 8)`. The pure-Python engine rounds as Spark
does.

## String promotion in `union`

**Affected operations:** `union` / `unionByName` of a string column with a numeric one.

Spark 4 (ANSI) widens a string next to an integral type to `bigint`, and next to a fractional type
(`float`, `double`, `decimal`) to `double`. It then casts the string side strictly, so `"20"` becomes
`20` and `"abc"` raises `CAST_INVALID_INPUT`. The Polars engine fails to stack the two columns
(`SchemaError: type String is incompatible with expected type Int64`).

**Workaround:** cast the string column to the numeric type before the union. The pure-Python engine
implements Spark's widening.

## Decimal scale in `createDataFrame`

**Affected operations:** `spark.createDataFrame(rows, schema)` with a `DecimalType(p, s)` field and
`Decimal` values whose scale differs from `s`.

Spark stores each value at the field's scale, rounded HALF_UP (`Decimal("0.065")` in `decimal(5,4)`
is `0.0650`), and rejects a value with more integral digits than the precision allows. The Polars engine
keeps the value's own scale, so a later `cast("string")` or `to_json` prints `0.065`, and it accepts
the over-wide value.

**Workaround:** quantize the values to the field's scale before building the frame. The pure-Python
engine rescales as Spark does.

## `to_json` decimal and timestamp output

**Affected operations:** `F.to_json(...)` over structs, maps or arrays that contain `DecimalType`,
`TimestampType` or `DateType` values.

Spark writes a decimal as a JSON **number** at the column's scale (`{"apr":0.0650}`). A timestamp is
written as a string in the `timestampFormat` option, by default `yyyy-MM-dd'T'HH:mm:ss.SSSXXX` in the
session time zone (`"2026-10-01T15:32:10.000Z"` under UTC), and a date in `dateFormat` (default
`yyyy-MM-dd`). The Polars engine writes decimals as JSON **strings** (`{"apr":"0.0650"}`) and timestamps
as `yyyy-MM-dd HH:mm:ss`, and it rejects every option other than `ignoreNullFields`.

**Workaround:** none that reproduces Spark's output exactly. Cast decimals to `double` if the consumer
accepts numbers without a fixed scale. The pure-Python engine matches Spark.

## Polars deviations found by the nba parity sweep

The parity tests written for the Python engine (`Engine.PYTHON`), which datahub's nba function
uses, found these Polars deviations from Spark 4. The pure-Python engine matches Spark on all of them.
The items are grouped by area; the parity ids in parentheses are gated in `POLARS_NOT_SUPPORTED`.

**Casts and arithmetic**

- Decimal result types: `+ - *` widen to `decimal(38, s)` instead of Spark's precision rules
  (`column.arithmetic.decimal_result_type`). `decimal / decimal` returns `double`
  (`column.divide.decimal_result_type`). `sum(decimal(10,2))` stays `decimal(10,2)`; Spark gives
  `decimal(20,2)` (`dataframe.groupby.sum_decimal_result_type`).
- Division by zero returns null or `inf` instead of raising `DIVIDE_BY_ZERO`
  (`column.divide.by_zero_raises`, `column.divide.integral_returns_double`,
  `functions.try_divide.by_zero_null`).
- String → number/decimal casts do not trim whitespace (`" 12 "`) (`column.cast.decimal_to_string_keeps_scale`,
  `column.cast.to_decimal_overflow_raises`, `column.try_cast.malformed_string_matrix`,
  `column.try_cast.to_decimal_overflow_null`).
- Rounding to a decimal scale truncates or rounds half-even instead of HALF_UP, and decimal overflow under
  `try_cast` does not become null (`column.cast.to_decimal_rounds_half_up`, `column.try_cast.ddl_string_target`).
- String → date takes only a bare date (Spark takes the date part of `"2024-01-15T10:30:45"`), and string →
  timestamp rejects a date-only string (`column.try_cast.malformed_string_matrix`, `column.try_cast.ddl_string_target`).
- Timestamp → string always prints 6 fraction digits; Spark drops trailing zeros (`column.cast.timestamp_to_string`).
- `F.round` rounds half-even and keeps the input decimal scale (`functions.round.half_up_double_and_decimal`).
- A double cast to string prints `1e-9`; Spark prints `1.0E-9` (`functions.greatest.mixed_numeric_types`).

**Comparisons and predicates**

- `date <=> string` fails instead of casting the string (`column.eq_null_safe.mixed_types`).
- `isin` does not widen mixed-type candidates, and a null candidate gives `false` instead of null when nothing
  matches (`column.isin.coerces_candidates`, `column.isin.null_semantics`).
- `contains(None)` raises, and a Column needle is unsupported (`column.contains.literal_and_null`).
- Regex: Rust's `$` does not match before a trailing `\n`, and its `\d` matches non-ASCII digits
  (`column.rlike.nba_patterns`).
- A string compared with an int skips the ANSI cast, so `"abc"` does not raise (`dataframe.is_empty`).

**Nested values**

- `getField` on a missing field or a non-struct does not raise (`column.get_field.struct`).
- An out-of-range `arr[i]` returns null instead of raising `INVALID_ARRAY_INDEX`; a `map<int,string>` column
  cannot be built (`column.getitem.array_out_of_range_null`, `column.getitem.map_missing_key_null`).
- An empty `create_map()` cannot fill more than one row, and a null map key does not raise `NULL_MAP_KEY`
  (`functions.create_map.empty`, `functions.create_map.null_key_raises`).
- `map_entries` returns the map unchanged (`functions.map_entries`).

**Ordering and windows**

- `asc_nulls_first` / `desc_nulls_first` are missing. `orderBy` cannot mix a plain column with a `.desc()` key.
  The requested null ordering is ignored, so nulls sort first (`column.sort.nulls_ordering`,
  `window.dense_rank.ties`, `window.order_by.desc_nulls_last`).
- A ranking window with no `partitionBy` fails instead of ranking the whole frame
  (`window.row_number.without_order_raises`).

**JSON**

- `from_json`: malformed JSON gives null where Spark gives a struct of nulls. A `map<string,string>` keeps
  nested objects as Python text and booleans as `"True"` (`functions.from_json.malformed_permissive`,
  `functions.from_json.map_and_array_schemas`).
- `get_json_object` rejects single-quoted JSON and raises on malformed paths (`$.`, `$[`, `a`) instead of
  returning null (`functions.get_json_object.malformed_json_null`).
- `to_json` accepts no option other than `ignoreNullFields` (`functions.to_json.options`).

**Dates**

- `try_to_date` returns null for valid `M/d/yyyy`, `M/d/yy`, `d/M/yy`, `dd/MM/yy` and `MM/yy` dates. It
  accepts `2024-1-5` for `yyyy-MM-dd` and `2/9/2024` for `MM/dd/yyyy`, and it reads `02/29/24` under
  `MM/dd/yyyy` as year 24 (`functions.try_to_date.nba_patterns`). **The nba function uses all of these
  patterns.**

**DataFrame**

- `createDataFrame` does not reject `True` in a bigint field, an `int` in a double field, or a string in a
  boolean field. It stores `True` in a string field as `"True"` instead of `"true"`. Inference does not reject a
  column that mixes bigint and double (`dataframe.create.value_type_mismatch_raises`,
  `dataframe.create.names_only_schema_infers_types`).
- `distinct` / `dropDuplicates` accept map columns, which Spark rejects. The `drop_duplicates` alias is missing
  (`dataframe.distinct.basic`, `dataframe.drop_duplicates.all_columns`).
- Column-condition joins rename the right side's duplicate `x` to `x_right`, support only inner/left, and do not
  raise `AMBIGUOUS_REFERENCE` for an unqualified duplicate name (`dataframe.join.column_condition_qualified_refs`,
  `dataframe.join.ambiguous_reference_raises`, `dataframe.join.left_anti`).
- `first(..., ignorenulls=True)` has no whole-frame `DataFrame.agg` (`functions.first.ignorenulls`).
- `union` does not reject a boolean column unioned with a bigint one (`dataframe.union.positional`).
- `toPandas` gives `datetime64[us]` timestamps (Spark: `[ns]`), and an empty string column is `str`
  (Spark: `object`) (`dataframe.to_pandas.dtypes`).

## Polars deviations found by the Python engine gap work

Parity tests written while closing the Python engine's gaps against Polars (`PYTHON_ENGINE_GAPS.md`) found
these Polars deviations from Spark 4. The pure-Python engine matches Spark on all of them; the parity ids in
parentheses are gated in `POLARS_NOT_SUPPORTED`.

**DataFrame.alias**

- `alias` stores the name but nothing resolves it: `F.col("e.id")`, `select("e.*")` and `filter("e.id > 1")` fail
  with `ColumnNotFoundError`, so a self-join cannot tell its sides apart (`dataframe.alias.qualified_references`).

**createDataFrame**

- A pandas frame goes through Arrow instead of PySpark's `to_records` path, so a mixed `object` column
  (`[1, "x"]`) raises `ArrowInvalid` instead of becoming `string` (`dataframe.create.from_pandas`).
- `dtypes` prints nested types without their element types (`array`, not `array<bigint>`), and an Arrow map
  column reports `array` (`dataframe.create.from_arrow`, `dataframe.create.from_arrow_stream`).
- A single-`DataType` schema treats a list item as a row, so `createDataFrame([[1, 2]], ArrayType(...))`
  raises instead of giving one `value` (`dataframe.create.single_datatype_schema`).
- A string field stores Python's `str` without PySpark's bool lowercasing (`"True"`) and renders dicts as
  JSON (`dataframe.create.string_field_python_str`).
- Empty data with only column names builds an empty frame; Spark raises `CANNOT_INFER_EMPTY_SCHEMA`
  (`dataframe.create.empty_inference_raises`).
- A DDL string schema (`"a int, b string"`) fails to build rows that a `StructType` builds
  (`dataframe.create.ddl_schema`).

**toArrow**

- There is no `toArrow` (PySpark 4's name). `to_arrow` returns Polars' own Arrow export, whose types differ from
  Spark's (`large_string`, `large_list`) (`dataframe.to_arrow`).

**Column**

- `Column.like` is not implemented (`column.like`).
- A number cast to timestamp is read as microseconds, not seconds, since the epoch
  (`column.cast.numeric_to_timestamp`), and a timestamp does not cast to a number
  (`column.cast.timestamp_to_numeric`).
- Casts that Spark 4 ANSI rejects at analysis succeed: number/boolean → date, boolean → timestamp,
  date → number/boolean and timestamp → boolean (`column.cast.ansi_datetime_mismatch_raises`).
- NaN / ±inf cast to decimal raises; Spark gives null even under ANSI (`column.cast.non_finite_to_decimal_null`).

**filter / where SQL strings**

- Predicate strings go to `pl.sql_expr`, whose SQL dialect is not Spark's
  (`dataframe.filter.sql_string_expressions`):
  - typed number literals (`1e0`, `2S`, `1Y`), double-quoted strings and `\'` escapes are not understood;
  - string escapes (`\u0031`, `\061`, `\d`) are not unescaped, and adjacent literals (`'a' '1'`) don't concatenate;
  - `LIKE` escapes, `IS [NOT] TRUE/FALSE/UNKNOWN`, `NOT IN` with a null candidate and `round` give different rows;
  - several Spark functions (`date_format`, `datediff`) and `IN` lists with expressions don't resolve;
  - `LIKE 'a\\'` (escape at the end) and an out-of-range `128Y` don't raise.

**lit**

- `lit([1, 2.5])` fails instead of widening the elements to `double`, and `lit(dict)` / `lit(tuple)` build
  struct/list literals where Spark raises `UNSUPPORTED_FEATURE.LITERAL_TYPE` (`functions.lit.list`).

**from_json**

- A DDL schema string must be one of a few simple forms: `struct<...>` and field lists with nested types raise
  `Unsupported Spark type name`, a top-level `name: type` list is accepted (Spark's `fromDDL` rejects it), and a
  non-container type (`"int"`) is accepted instead of raising `INVALID_JSON_SCHEMA`. An unparseable record
  gives null instead of a struct of nulls (`functions.from_json.ddl_schema`).
