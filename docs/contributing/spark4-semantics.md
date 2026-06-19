# Spark 4 cast and parse semantics

Spark 4 defaults to ANSI-strict behavior (`spark.sql.ansi.enabled=true`). Malformed casts and
parses **throw** (e.g. `CAST_INVALID_INPUT`), they do not silently return `null`. Spark 4 added
`try_*` and `try_cast` as lenient variants that return `null` on error.

| API family | Strict (raises on bad input) | Lenient (returns `null`) |
|------------|--------------------------------|--------------------------|
| Cast | `Column.cast` | `Column.try_cast` |
| Timestamps | `to_timestamp` | `try_to_timestamp` |
| Dates | `to_date` (where strict) | `try_to_date` |
| Boolean from string | `cast` / strict helpers | `try_cast` / lenient helpers |
| Array index / map lookup | `element_at` | `try_element_at` |

## Rules when adding or changing behavior

- Implement strict and lenient pairs with a **shared helper and a `strict` flag** where applicable
  (see `_to_datetime_column`, `_to_timestamp_no_format_column`, and `element_at_column` in
  `functions_helpers.py`).
- Strict string→boolean casting uses `_string_to_bool_expr_strict`; lenient uses
  `_string_to_bool_expr` (in `column_helpers.py`).
- Do **not** reintroduce Spark 3–style silent nulls for strict APIs.
- **Never implement a strict API as a one-line delegate to its `try_*` sibling** (e.g. `element_at`
  must not call `try_element_at`). Spark 4 ANSI strictness is the default; the lenient function is
  the exception.
- Shared logic belongs in `*_helpers.py` with a `strict` flag.

## `element_at` vs `try_element_at` — string extraction on maps

For maps, Spark documents different semantics (SPARK-48766):

- `element_at(col, "key")` uses the **literal string** `"key"`.
- `try_element_at(col, "key")` uses the **column named** `key`.

So when a lenient map lookup needs a literal key, use `lit("key")` (or a `Column` wrapping a
literal). The shared helper `element_at_column` encodes this difference behind its `strict` flag.

## Testing the pairs

Strict/lenient behavior must be covered on **both** sides — a test where Spark raises on the strict
API (confirm on Spark, then `pytest.raises` on SparkleFrame), and a matching test that the lenient
API returns `null` on the same input. The full patterns and examples are in
[testing.md](testing.md).
