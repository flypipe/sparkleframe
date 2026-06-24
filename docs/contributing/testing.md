# Testing

Every new method on **functions**, **column**, or **dataframe** must include **Spark parity
tests** in the corresponding `*_test.py`. A parity test runs the *same pipeline* on SparkleFrame
and on real PySpark and asserts the results match.

## Run tests in Docker

Always run tests via `make`, not bare `pytest` — the Docker container sets env vars (e.g.
`TIMEZONE`) the Spark fixture depends on.

```bash
make test                                            # whole suite
make test f=sparkleframe/polarsdf/functions_test.py  # one file
make test f=sparkleframe/tests/parity                # the engine-parametrized parity suite
make pr-check                                         # black + lint + coverage (the PR gate)
```

## The assertion helper

Use `assert_matches_spark` from `sparkleframe/tests/parity/oracle.py` whenever Spark produces a
result DataFrame. It compares an engine frame against a real PySpark frame **via the engine
adapter** (engine-agnostic), with order-insensitive multiset compare by default.

For **co-located, polars-only tests** (the `sparkleframe/polarsdf/*_test.py` files), import the
`ENGINES` dict and pass the polars adapter, keyed by `Engine.POLARS`:

```python
from sparkleframe.engine import Engine
from sparkleframe.tests.parity.engines import ENGINES
from sparkleframe.tests.parity.oracle import assert_matches_spark

sf_result = sparkle_df.select(...)
spark_result = spark_df.select(...)
assert_matches_spark(sf_result, spark_result, ENGINES[Engine.POLARS])
```

**Engine-parametrized** parity tests in `sparkleframe/tests/parity/` use the `engine` fixture
instead (see [Parity-test conventions](#parity-test-conventions) below). Pass `check_row_order=True`
only when the ordering itself is under test (e.g. `orderBy` / `sort`).

## Cover valid and malformed inputs

Parity tests must exercise **more than well-formed data**. A common mistake is cast or timestamp
tests where every string parses successfully, so strict error paths (`CAST_INVALID_INPUT`, etc.)
are never evaluated.

For cast, parse, and conversion APIs, fixtures **must** include malformed or edge-case values:

- Invalid timestamp or date strings (`"not-a-date"`, wrong pattern for the format)
- Unparseable strings for numeric or boolean casts
- `null` mixed with bad values (see `test_try_to_timestamp_malformed_returns_null` in
  `functions_test.py`)

What to assert depends on the API:

1. **Strict APIs** (`cast`, `to_timestamp`, …): include malformed input where **Spark raises**;
   confirm Spark fails, then assert SparkleFrame **also raises** on evaluation (`pytest.raises`).
   See `TestColumnSparkParity` in `column_test.py`.
2. **Lenient APIs** (`try_cast`, `try_to_timestamp`, `try_to_date`, …): include malformed rows; bad
   values become `null`, then compare the full result frame to Spark with
   `assert_matches_spark(sf, spark, ENGINES[Engine.POLARS])`.
3. **Valid inputs:** still required — same pipeline on Spark and SparkleFrame, then frame parity.
4. **Strict/lenient pairs:** add at least one test where Spark **raises** on the strict API (confirm
   on Spark first, then `pytest.raises` on SparkleFrame evaluation), and a matching test that the
   lenient API returns `null` on the same input. See `TestElementAt::test_array_oob_raises_like_spark`
   in `functions_test.py`.

See [spark4-semantics.md](spark4-semantics.md) for which APIs are strict vs lenient and why.

```python
import pytest
import polars as pl
from pyspark.sql.functions import to_timestamp as spark_to_timestamp
from pyspark.sql.functions import try_to_timestamp as spark_try_to_timestamp

from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.functions import to_timestamp, try_to_timestamp
from sparkleframe.engine import Engine
from sparkleframe.tests.parity.engines import ENGINES
from sparkleframe.tests.parity.oracle import assert_matches_spark

# Valid inputs
df = pl.DataFrame({"ts": ["2023-01-01 12:34:56", None]})
sparkle_df = DataFrame(df)
spark_df = spark.createDataFrame(df.to_pandas())
sf_result = sparkle_df.select(to_timestamp("ts").alias("result"))
spark_result = spark_df.select(spark_to_timestamp("ts").alias("result"))
assert_matches_spark(sf_result, spark_result, ENGINES[Engine.POLARS])

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
assert_matches_spark(sf_result, spark_result, ENGINES[Engine.POLARS])
```

## Test layering: who owns what

SparkleFrame has five overlapping test layers. Each layer owns a *different property* of the
same code. When writing a new test, decide which layer should own the assertion and resist
restating it elsewhere — duplicated assertions add maintenance cost without adding signal.

| Layer | Lives in | Owns |
|---|---|---|
| AST phase units | `sparkleframe/python/ast/*_test.py` | What a single phase produces: coercion table, analyze inserts `Cast`, evaluate computes values from a resolved tree. Pure functions; no DataFrame. |
| Engine parity | `sparkleframe/tests/parity/` | Result *values* match real Spark across `[POLARS]` and `[PYTHON]`. The oracle compares records as a sorted multiset and does **not** assert schema/dtypes. |
| Colocated DataFrame units | `sparkleframe/<engine>/dataframe_test.py` | Orchestration contract for the DataFrame surface: projection, output order, output names, *resolved output dtype*, argument handling, unimplemented-slot raising. **Not values** — parity owns those. |
| Colocated Column units | `sparkleframe/<engine>/column_test.py` | AST shape produced by dunder methods: which node type, which operator string, which operand is on which side (especially for reflected ops). |
| Colocated functions units | `sparkleframe/<engine>/functions_test.py` | AST shape produced by each `F.*` factory: node type, name, arg structure, branch order for `when/otherwise`. |

The shorthand: **parity owns values; colocated units own structure.** If a colocated test
asserts a numeric result that parity already covers, either delete the assertion or replace it
with the structural property it was implicitly probing (typically: output schema/dtype).

## Smells to avoid in colocated tests

These are the common smells that have shown up in review. Fix them at write-time, not at
review-time.

### 1. Re-asserting values that parity already covers

```python
# BAD — parity already proves 11/22 against real Spark; this is strictly weaker.
out = df.select((F.col("a") + F.col("b")).alias("r"))
assert out.to_records() == [{"r": 11}, {"r": 22}]
```

If the only unique thing the test proves is the resolved dtype, keep just that and drop the
value assertion:

```python
# GOOD — parity owns the value; this owns the schema/dtype claim parity ignores.
out = df.select((F.col("i") + F.col("l")).alias("r"))
assert isinstance(out._schema["r"].dataType, LongType)
```

### 2. Fixtures that can't fail the property

```python
# BAD — single-column input; ``select("a")`` passes even if select ignored its arg.
df = DataFrame([(1,), (2,)], StructType([StructField("a", IntegerType())]))
assert df.select("a").to_records() == [{"a": 1}, {"a": 2}]
```

A test must use a fixture that lets the property fail. For projection, that means **≥2
columns**; for ordering, distinct names; for dtype resolution, operands whose result type
differs from the inputs.

### 3. Mocking AST phases ("did analyze run?")

```python
# BAD — couples to call structure; passes even if the frame is assembled wrong.
with patch("sparkleframe.python.ast.analyzer.analyze") as m:
    df.select(F.col("a") + F.col("b"))
    m.assert_called_once()
```

A behavioral assertion on the output (schema + records, or resolved dtype) already proves
analyze fired *and* checks the contract. Mocks here lock in implementation details and bypass
the actual property under test.

### 4. Import-smoke tests

```python
# BAD — passes whenever the module exists; zero information.
def test_helpers_module_importable():
    assert isinstance(helpers.__name__, str)
```

Trust the import system. If you need a single canary that the package layout is intact, write
**one** dedicated test, not one per submodule.

### 5. Runtime "type" assertions that duplicate mypy

```python
# BAD — mypy and ruff catch this in CI.
assert isinstance(out, DataFrame)
```

The meaningful runtime analog is asserting the resolved output `StructField` dtype, which mypy
*can't* see.

### 6. Reflected-operator tests that only check the operator

```python
# BAD — passes even if ``__rsub__`` swaps operands.
result = 1 - c
assert result._expr.op == "-"
```

Non-commutative reflected dunders (`__rsub__`, `__rtruediv__`, `__rpow__`) must also assert
operand placement — otherwise a left/right swap regression slips through.

### 7. Substring repr smoke tests

```python
# BAD — passes for any repr that contains the class name.
assert "Column(" in repr(col)
```

Either assert the repr *delegates* (contains the wrapped name/expression), or delete the test.
"Class name appears in its own repr" is a tautology.

## Parity-test conventions

The shared parity suite (`sparkleframe/tests/parity/`) runs each test against **every engine**.

- **The `engine` fixture** (from `conftest.py`) parametrizes a test over all engines in `ENGINES`,
  so one test runs as `...[POLARS]` and `...[PYTHON]`. Build frames and expressions through the
  fixture, never by importing an engine directly:

  ```python
  @pytest.mark.feature("functions.abs.integer")
  def test_abs_integer(engine, spark):
      F = engine.functions
      actual = engine.build_df(rows, schema).select(F.abs(F.col("x")).alias("r"))
      expected = spark.createDataFrame(rows, schema).select(SF.abs(SF.col("x")).alias("r"))
      assert_matches_spark(actual, expected, engine)
  ```

- **`@pytest.mark.feature("<id>")`** tags the Spark behavior under test. The id is what the
  per-engine gate in `gaps.py` keys on. Use dotted `module.area.case` naming, matching existing
  ids — e.g. `functions.to_timestamp.with_format`, `column.cast.int_to_boolean`,
  `dataframe.join.polars_joins`.
- **Gating** is automatic: if the id is in an engine's gate set, the test `xfail`s for that engine;
  unscaffolded engines skip. You don't add `xfail` markers by hand — see [engines.md](engines.md).
- **`check_row_order=True`** only when ordering is the thing under test.
