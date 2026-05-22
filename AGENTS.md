# SparkleFrame — agent guide

## Project identity

SparkleFrame implements the PySpark DataFrame API so pipelines run on
[Polars](https://docs.pola.rs/api/python/stable/reference/index.html) without a Spark cluster. It is a **partial shim**
aimed at the **PySpark / Apache Spark 4.x** DataFrame API: not every operation is supported, but behavior matches Spark
where implemented. See [README.md](README.md) for motivation and usage.

**Spark 4 only.** Do not preserve Spark 3 semantics, compatibility shims, or pre–Spark 4 lenient defaults unless Spark
4 still defines them.

## Code layout

Public API lives in thin modules that mirror PySpark:

- `sparkleframe/polarsdf/functions.py`
- `sparkleframe/polarsdf/column.py`
- `sparkleframe/polarsdf/dataframe.py`

**Helpers:** put new private or shared implementation logic in the matching `*_helpers.py` file, not in the public
modules:

- `sparkleframe/polarsdf/column_helpers.py`
- `sparkleframe/polarsdf/dataframe_helpers.py`
- `sparkleframe/polarsdf/functions_helpers.py`

## Spark 4 cast and parse semantics

Spark 4 defaults to ANSI-strict behavior (`spark.sql.ansi.enabled=true`). Malformed casts and parses **throw** (e.g.
`CAST_INVALID_INPUT`), not silent null. Spark 4 added `try_*` and `try_cast` as lenient variants that return null on
error.

| API family | Strict (raises on bad input) | Lenient (returns `null`) |
|------------|--------------------------------|--------------------------|
| Cast | `Column.cast` | `Column.try_cast` |
| Timestamps | `to_timestamp` | `try_to_timestamp` |
| Dates | `to_date` (where strict) | `try_to_date` |
| Boolean from string | `cast` / strict helpers | `try_cast` / lenient helpers |
| Array index / map lookup | `element_at` | `try_element_at` |

When adding or changing behavior:

- Implement strict and lenient pairs with a shared helper and a `strict` flag where applicable (see
  `functions_helpers._to_datetime_column`, `_to_timestamp_no_format_column`).
- Strict string→boolean casting uses `_string_to_bool_expr_strict`; lenient uses `_string_to_bool_expr` in
  `column_helpers.py`.
- Do not reintroduce Spark 3–style silent nulls for strict APIs.
- **Never implement a strict API as a one-line delegate to its `try_*` sibling** (e.g. `element_at` must not call
  `try_element_at`). Spark 4 ANSI strictness is the default; the lenient function is the exception.
- Shared logic belongs in `*_helpers.py` with a `strict` flag (see `element_at_column` in `functions_helpers.py`).
- **`element_at` vs `try_element_at` string `extraction`:** for maps, Spark documents different semantics (SPARK-48766):
  `element_at(col, "key")` uses the literal string `"key"`; `try_element_at(col, "key")` uses the column named `key`.
  Use `lit("key")` (or a `Column` wrapping a literal) when a lenient map lookup needs a literal key.

## Testing requirements

Every new method on **functions**, **column**, or **dataframe** must include **Spark parity tests** in the
corresponding `*_test.py`.

### Assertion helper

Use `assert_sparkle_spark_frame_are_equal` from `sparkleframe/tests/utils.py` whenever Spark produces a result
DataFrame. Pass one SparkleFrame `DataFrame` and one PySpark `DataFrame` (different types); the helper JSON-compares
normalized row records.

```python
from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal

sf_result = sparkle_df.select(...)
spark_result = spark_df.select(...)
assert_sparkle_spark_frame_are_equal(sf_result, spark_result)
```

New tests should use this helper rather than `assert_pyspark_df_equal`.

### Cover valid and malformed inputs

Parity tests must exercise **more than well-formed data**. A common mistake is cast or timestamp tests where every
string parses successfully, so strict error paths (`CAST_INVALID_INPUT`, etc.) are never evaluated.

For cast, parse, and conversion APIs, fixtures **must** include malformed or edge-case values, for example:

- Invalid timestamp or date strings (`"not-a-date"`, wrong pattern for the format)
- Unparseable strings for numeric or boolean casts
- `null` mixed with bad values (see `test_try_to_timestamp_malformed_returns_null` in `functions_test.py`)

What to assert depends on the API:

1. **Strict APIs** (`cast`, `to_timestamp`, …): include malformed input where **Spark raises**; confirm Spark fails,
   then assert SparkleFrame **also raises** on evaluation (`pytest.raises`). See `TestColumnSparkParity` in
   `column_test.py`.
2. **Lenient APIs** (`try_cast`, `try_to_timestamp`, `try_to_date`, …): include malformed rows; bad values become
   `null`, then compare the full result frame to Spark with `assert_sparkle_spark_frame_are_equal`.
3. **Valid inputs:** still required — same pipeline on Spark and SparkleFrame, then frame parity.
4. **Strict/lenient pairs:** add at least one test where Spark **raises** on the strict API (confirm on Spark first,
   then `pytest.raises` on SparkleFrame evaluation), and a matching test that the lenient API returns `null` on the
   same input. See `TestElementAt::test_array_oob_raises_like_spark` in `functions_test.py`.

```python
import pytest
import polars as pl
from pyspark.sql.functions import to_timestamp as spark_to_timestamp
from pyspark.sql.functions import try_to_timestamp as spark_try_to_timestamp

from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.functions import to_timestamp, try_to_timestamp
from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal

# Valid inputs
df = pl.DataFrame({"ts": ["2023-01-01 12:34:56", None]})
sparkle_df = DataFrame(df)
spark_df = spark.createDataFrame(df.to_pandas())
sf_result = sparkle_df.select(to_timestamp("ts").alias("result"))
spark_result = spark_df.select(spark_to_timestamp("ts").alias("result"))
assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

# Malformed input — strict API must fail (confirm Spark first)
bad_df = pl.DataFrame({"ts": ["not-a-date"]})
sparkle_bad = DataFrame(bad_df)
spark_bad = spark.createDataFrame(bad_df.to_pandas())
with pytest.raises(Exception):
    spark_bad.select(spark_to_timestamp("ts")).collect()
with pytest.raises(Exception):
    sparkle_bad.select(to_timestamp("ts")).to_native_df()

# Malformed input — lenient API returns null, then frame parity
mixed_df = pl.DataFrame({"ts": ["2023-01-01 12:34:56", "not-a-date", None]})
sparkle_mixed = DataFrame(mixed_df)
spark_mixed = spark.createDataFrame(mixed_df.to_pandas())
sf_result = sparkle_mixed.select(try_to_timestamp("ts").alias("result"))
spark_result = spark_mixed.select(spark_try_to_timestamp("ts").alias("result"))
assert_sparkle_spark_frame_are_equal(sf_result, spark_result)
```

## Local development

When editing Python in this repo:

- Add type hints on all function parameters and return types.
- Run `make black` and `make lint` to validate code structure.
- Avoid leaving dead code or dead methods.
- Run affected tests, e.g. `make test f=sparkleframe/polarsdf/functions_test.py`.
