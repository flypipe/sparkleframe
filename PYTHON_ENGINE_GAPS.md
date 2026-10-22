# Python engine ← Polars gaps

Tracks behaviors the Polars engine (`sparkleframe/polarsdf`) supports that the pure-Python engine
(`sparkleframe/python`) does not yet. Stacked on #114 (`feat-nba-python-engine`).

Every public Polars name already exists on the Python engine (directly or as an alias), and the
Python parity gate is drained (`PYTHON_GATE_HIGH_WATER_MARK = 0`), so the gaps below are
**behavioral**: argument forms, input types and code paths the Python engine rejects or mishandles.
Where Polars itself deviates from Spark, the target is **Spark 4**, not Polars.

Each item lands as its own commit with engine-parametrized parity tests (valid **and** malformed
inputs) in `sparkleframe/tests/parity/`.

## Status

| # | Gap | Status |
|---|---|---|
| 1 | `createDataFrame` / `DataFrame(...)` from pandas, polars and arrow frames; bare `DataType` schema | done |
| 2 | `from_json` with a DDL field-list schema (`"a int, b string"`) | done |
| 3a | `Column.like` (missing in both engines; needed by SQL `LIKE`) | done |
| 3 | `filter`/`where` SQL strings: negative literals, arithmetic, `IN`, `BETWEEN`, `LIKE`, `RLIKE`, function calls, `CAST` | done |
| 3b | `%` / modulo with ANSI remainder semantics (`Column.__mod__` and SQL `%`, missing in both engines) | pending |
| 4 | `lit(list)` / `lit(dict)` literals (Spark: list is `array`, dict/tuple raise) | done |
| 5 | Numeric ↔ timestamp casts (epoch seconds); ANSI-rejected datetime casts raise at analysis | done |
| 6 | `DataFrame.alias` with `alias.col` / `alias.*` resolution (self-joins) | done |
| 7 | `DataFrame.toArrow` (PySpark 4) and its `to_arrow` alias, with Spark's Arrow types | done |
| 8 | Datetime pattern letters (`Q`, `K`, `k`, `z`, `VV`, `O`, `F`, `G`, `D` parsing) and Spark's pattern errors | done |
| 9 | `get_json_object` `[*]` wildcard paths (and Jackson's number rendering) | done |
| 10 | Time-only string → timestamp cast | pending |
| 11 | `Window.rangeBetween` / `rowsBetween` bound validation | pending |
| 12 | `fillna(subset=[...])` with a column that doesn't exist | pending |

## Notes per item

1. **createDataFrame from frames.** `DataFrame.__init__` runs `list(data)`, so a pandas frame
   silently becomes rows of its column names. Polars frames raise `LENGTH_SHOULD_BE_THE_SAME`, and
   arrow tables raise `NotImplementedError`. `createDataFrame([1, 2], IntegerType())` raises `TypeError`.
2. **from_json DDL.** `_json_schema` in `ast/function_registry.py` calls `parse_ddl_type` without
   the `struct<...>` wrap that `session.py` applies to field lists.
3. **SQL predicates.** `_PredicateParser` in `dataframe_helpers.py` only knows comparisons,
   `IS [NOT] NULL`, `AND`/`OR`/`NOT` and parentheses.
4. **lit(list/dict).** Literal type inference (`ast/coercion.py`) raises on `list`/`dict`.
5. **Numeric → timestamp.** `cast_value` raises `NotImplementedError`; Polars returns wrong values
   (it reads ints as microseconds).
6. **alias.** Python raises `NotImplementedError`; Polars stores the alias.
7. **to_arrow.** Python raises `NotImplementedError`; Polars returns a `pyarrow.Table`.
8. **Pattern letters.** Python raises `NotImplementedError` for unknown letters; Polars echoes them.
   Parsing reads `D` into `fields["day_of_year"]` but never uses it.
9. **get_json_object wildcards.** Python raises; Polars returns only the first match.
10. **Time-only strings.** `cast`/`try_cast` of `"10:30:00"` to timestamp raises `NotImplementedError`.
11. **Window bounds.** Polars raises `TypeError` for non-int bounds; Python accepts anything.
12. **fillna subset.** Python raises a bare `KeyError`; confirm Spark's behavior first.

## Out of scope (missing in both engines)

- Aggregates over a window (`F.sum(...).over(w)`) and applying frame bounds.
