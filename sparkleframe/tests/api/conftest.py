"""Shared fixtures for backend-parameterized API parity tests.

Every test in this directory runs against both polarsdf and pythondf, plus a
native PySpark equivalent for parity. Tests should take ``(session, F, spark)``
and assert equality via ``assert_sparkle_spark_frame_are_equal``.

``XFAIL_POLARSDF`` records known polarsdf-vs-PySpark divergences so existing
polarsdf bugs don't block pythondf progress.
"""
from __future__ import annotations

import importlib
from typing import Iterable

import pytest


BACKENDS = ("polarsdf", "pythondf")


# Registry of (test node id substring, backend) -> reason.
# Example: ("test_decimal_division", "polarsdf"): "polarsdf rounds at 18 vs Spark 38 digits".
XFAIL_POLARSDF: dict[tuple[str, str], str] = {
    # polarsdf.Column does not implement __mod__/__rmod__ at all (Spark's `%` operator).
    ("test_arithmetic_col_col[polarsdf-mod", "polarsdf"): "polarsdf Column has no __mod__",
    ("test_reverse_arithmetic_literal_col[polarsdf-rmod", "polarsdf"): "polarsdf Column has no __rmod__",
    ("test_int_nulls[polarsdf-mod", "polarsdf"): "polarsdf Column has no __mod__",
    ("test_double_nulls[polarsdf-mod", "polarsdf"): "polarsdf Column has no __mod__",
    ("test_arithmetic_col_col_typed[polarsdf-mod", "polarsdf"): "polarsdf Column has no __mod__",
    # polarsdf Column.cast rejects str type names (only DataType); only try_cast accepts str.
    ("test_and_with_lit_null[polarsdf]", "polarsdf"): "polarsdf Column.cast rejects str type names",
    # polarsdf delegates isin to Polars is_in, which treats null-in-list as a regular value and
    # returns False/True instead of Spark's null-on-non-match semantics.
    ("test_isin_list_has_null[polarsdf]", "polarsdf"): "polarsdf isin does not apply Spark null-in-list semantics",
    ("test_isin_string_list_has_null[polarsdf]", "polarsdf"): "polarsdf isin does not apply Spark null-in-list semantics",
    # polarsdf Column has no .between() method.
    ("test_between_int[polarsdf]", "polarsdf"): "polarsdf Column has no .between()",
    ("test_between_string[polarsdf]", "polarsdf"): "polarsdf Column has no .between()",
    ("test_between_with_null_value[polarsdf]", "polarsdf"): "polarsdf Column has no .between()",
    ("test_between_inclusive_bounds[polarsdf]", "polarsdf"): "polarsdf Column has no .between()",
    # polarsdf Column has no .startswith() / .endswith() / .like() methods.
    ("TestStartswith", "polarsdf"): "polarsdf Column has no .startswith()",
    ("TestEndswith", "polarsdf"): "polarsdf Column has no .endswith()",
    ("TestLike", "polarsdf"): "polarsdf Column has no .like()",
    # Batch 5 (cast / try_cast) — polarsdf vs PySpark divergences.
    # polarsdf Column.cast only accepts DataType instances; Spark also accepts
    # type-name strings like "int", "double", "decimal(10,2)".
    ("test_cast_accepts_string_type_name[polarsdf]", "polarsdf"): "polarsdf cast rejects str type names",
    ("test_cast_accepts_decimal_string_type_name[polarsdf]", "polarsdf"): "polarsdf cast rejects str type names",
    # polars strict_cast(Int32) on " 10 " fails — Spark trims whitespace.
    ("test_string_to_numeric_valid[polarsdf", "polarsdf"): "polarsdf does not trim whitespace when casting str to numeric",
    # polars strict_cast(Datetime) cannot parse 'YYYY-MM-DD HH:MM:SS' (only ISO 'T' form).
    ("test_string_to_timestamp_via_string_roundtrip[polarsdf]", "polarsdf"): "polarsdf cannot cast 'YYYY-MM-DD HH:MM:SS' to timestamp",
    # Batch 7 (getItem / getField / __getitem__) — polarsdf gaps vs PySpark.
    # polarsdf Column has no .getField() method (Spark alias for struct field access).
    ("test_get_field_struct[polarsdf]", "polarsdf"): "polarsdf Column has no .getField()",
    # polarsdf's map-as-Struct getItem returns an empty list (drops the row)
    # for a missing key instead of returning null per Spark map semantics.
    ("test_map_missing_key_returns_null[polarsdf]", "polarsdf"): "polarsdf map getItem drops rows on missing key (row-count divergence)",
    # polars list.get raises on out-of-bounds index instead of returning null.
    ("test_array_out_of_bounds_returns_null[polarsdf]", "polarsdf"): "polarsdf raises ComputeError on out-of-bounds list.get",
    # Batch 9 (ordering / slicing) — polarsdf vs PySpark divergences.
    # polarsdf.orderBy(Column) requires the Column to have sort metadata attached
    # via .asc()/.desc(); a bare F.col("x") raises AttributeError on `_sort_col`.
    ("test_order_by_column_default_asc[polarsdf]", "polarsdf"): "polarsdf orderBy requires .asc()/.desc() on bare Column",
    # polarsdf's str-key orderBy hard-codes nulls_last=True, which contradicts
    # PySpark's default of NULLS FIRST for ASC.
    ("test_order_by_str_default_nulls_first[polarsdf]", "polarsdf"): "polarsdf str orderBy uses NULLS LAST; Spark default is NULLS FIRST for ASC",
    # polarsdf Column has no asc_nulls_first / desc_nulls_first variants.
    ("test_order_by_asc_nulls_first_explicit[polarsdf]", "polarsdf"): "polarsdf Column has no .asc_nulls_first()",
    ("test_order_by_desc_nulls_first[polarsdf]", "polarsdf"): "polarsdf Column has no .desc_nulls_first()",
    # polarsdf DataFrame has no .limit() method.
    ("TestLimit", "polarsdf"): "polarsdf DataFrame has no .limit()",
    # polarsdf DataFrame has no .head() / .take() / .first() methods.
    ("TestHead", "polarsdf"): "polarsdf DataFrame has no .head()",
    ("TestTake", "polarsdf"): "polarsdf DataFrame has no .take()",
    ("TestFirst", "polarsdf"): "polarsdf DataFrame has no .first()",
    # polarsdf DataFrame has no .count() method (used as parity baseline here).
    ("test_drop_duplicates_subset_single[polarsdf]", "polarsdf"): "polarsdf DataFrame has no .count()",
    ("test_drop_duplicates_subset_multi[polarsdf]", "polarsdf"): "polarsdf DataFrame has no .count()",
    # Batch 10 (metadata + set ops) — polarsdf gaps vs PySpark.
    # polarsdf DataFrame has no .isEmpty() method.
    ("TestIsEmpty", "polarsdf"): "polarsdf DataFrame has no .isEmpty()",
    # polarsdf DataFrame has no .printSchema() method.
    ("TestPrintSchema", "polarsdf"): "polarsdf DataFrame has no .printSchema()",
    # polarsdf DataFrame has no .toJSON() method (Spark returns an RDD).
    ("TestToJSON", "polarsdf"): "polarsdf DataFrame has no .toJSON()",
    # Batch 11 (groupBy + agg) — polarsdf gaps vs PySpark.
    # polarsdf functions module has no ``avg`` alias for ``mean``.
    ("test_avg_alias[polarsdf]", "polarsdf"): "polarsdf functions has no avg() alias",
    # polarsdf's GroupedData shortcuts (.sum/.mean/.min/.max) take no args; they
    # aggregate every numeric column. PySpark's API accepts a list of column names.
    ("test_groupby_sum[polarsdf]", "polarsdf"): "polarsdf GroupedData.sum() does not accept column args",
    ("test_groupby_mean[polarsdf]", "polarsdf"): "polarsdf GroupedData.mean() does not accept column args",
    ("test_groupby_min[polarsdf]", "polarsdf"): "polarsdf GroupedData.min() does not accept column args",
    ("test_groupby_max[polarsdf]", "polarsdf"): "polarsdf GroupedData.max() does not accept column args",
    # polarsdf sum returns 0 instead of None when every aggregated value is null
    # (Polars `pl.sum` over an all-null group returns 0; Spark returns NULL).
    ("test_agg_null_handling[polarsdf]", "polarsdf"): "polarsdf sum() returns 0 for all-null groups; Spark returns NULL",
    # Batch 14 (aggregate edge cases) — polarsdf gaps vs PySpark.
    # polarsdf's createDataFrame drops the schema when data is empty, so empty
    # DataFrames have zero columns and any agg referencing "a" raises
    # ColumnNotFoundError. Spark preserves the declared schema on empty input.
    ("test_empty_count_column_is_zero[polarsdf]", "polarsdf"): "polarsdf createDataFrame loses schema on empty data",
    ("test_empty_sum_is_null[polarsdf]", "polarsdf"): "polarsdf createDataFrame loses schema on empty data",
    ("test_empty_mean_is_null[polarsdf]", "polarsdf"): "polarsdf createDataFrame loses schema on empty data",
    ("test_empty_min_is_null[polarsdf]", "polarsdf"): "polarsdf createDataFrame loses schema on empty data",
    ("test_empty_max_is_null[polarsdf]", "polarsdf"): "polarsdf createDataFrame loses schema on empty data",
    ("test_empty_first_is_null[polarsdf]", "polarsdf"): "polarsdf createDataFrame loses schema on empty data",
    ("test_empty_collect_list_is_empty_array[polarsdf]", "polarsdf"): "polarsdf createDataFrame loses schema on empty data",
    ("test_empty_collect_set_is_empty_array[polarsdf]", "polarsdf"): "polarsdf createDataFrame loses schema on empty data",
    # Batch 15 (math functions) — polarsdf vs PySpark divergences.
    # polars Expr.round uses banker's rounding (HALF_EVEN) by default and also
    # rejects negative scales with OverflowError. Spark's round() uses HALF_UP.
    ("test_round_default_scale_zero[polarsdf]", "polarsdf"): "polars round uses HALF_EVEN; Spark uses HALF_UP",
    ("test_round_scale_two[polarsdf]", "polarsdf"): "polars round uses HALF_EVEN; Spark uses HALF_UP",
    ("test_round_negative_scale[polarsdf]", "polarsdf"): "polars round raises OverflowError on negative scale",
    # polarsdf.pow on int^int preserves Int64; Spark always returns Double.
    ("test_pow_col_col_ints[polarsdf]", "polarsdf"): "polarsdf pow(int,int) returns int; Spark returns Double",
    # Batch 12 (joins) — polarsdf full-outer aliases skip key coalescing.
    # polarsdf only coalesces full-outer join keys when ``how == "outer"``; the
    # aliases ``full`` / ``fullouter`` / ``full_outer`` slip past the special-
    # case and leave a stray ``id_right`` column behind.
    ("test_outer_alias[polarsdf-full]", "polarsdf"): "polarsdf only coalesces outer-join keys for how='outer' literal",
    ("test_outer_alias[polarsdf-fullouter]", "polarsdf"): "polarsdf only coalesces outer-join keys for how='outer' literal",
    ("test_outer_alias[polarsdf-full_outer]", "polarsdf"): "polarsdf only coalesces outer-join keys for how='outer' literal",
    # Batch 16 (string functions) — polarsdf gaps vs PySpark.
    # polarsdf.functions has no ``upper`` function (lower/initcap exist).
    ("TestUpper", "polarsdf"): "polarsdf functions has no upper()",
    # Batch 17 (date/time functions) — polarsdf gaps vs PySpark.
    # polarsdf's months_between does not round to 8 decimal places like Spark
    # (Spark's MonthsBetween rounds the fractional result; polarsdf returns the
    # raw float, producing extra digits on partial-month results).
    ("test_months_between_partial[polarsdf", "polarsdf"): "polarsdf months_between does not round to 8 dp",
    # polarsdf now()/current_timestamp() returns UTC wall-clock; Spark's
    # current_timestamp returns the JVM session-TZ datetime. On non-UTC hosts
    # the two diverge by the UTC offset (hours).
    ("test_current_timestamp_close_to_pyspark[polarsdf", "polarsdf"): "polarsdf current_timestamp uses UTC; Spark uses session TZ",
    ("test_now_close_to_pyspark[polarsdf", "polarsdf"): "polarsdf now() uses UTC; Spark current_timestamp uses session TZ",
    # Batch 18 (array functions) — polarsdf vs PySpark divergences.
    # Polars' ``list.sort(descending=True)`` keeps nulls at the front of the
    # array; Spark's ``sort_array(_, asc=False)`` puts nulls last.
    ("test_with_nulls_desc_puts_them_last[polarsdf]", "polarsdf"): "polarsdf sort_array desc keeps nulls first; Spark puts them last",
    # Batch 19 (struct / map functions) — polarsdf vs PySpark divergences.
    # polarsdf's size(map_keys(create_map(...))) collapses to an empty record
    # (the projection produces no output column at all). Likely a polarsdf
    # interaction between map_keys' map_elements UDF and size on the result.
    ("test_map_keys_non_empty[polarsdf]", "polarsdf"): "polarsdf size(map_keys(create_map(...))) drops the output column",
    ("test_map_from_entries_empty_array[polarsdf]", "polarsdf"): "polarsdf size(map_keys(map_from_entries(empty))) drops the output column",
    # Batch 20 (JSON functions) — polarsdf vs PySpark divergences.
    # polarsdf's str.json_path_match raises a SchemaError when the input column
    # is all-null (Polars cannot infer the String dtype). Spark returns NULL.
    ("test_null_input_returns_null[polarsdf]", "polarsdf"): "polarsdf json_path_match raises on all-null input",
    # polarsdf's to_json keeps null struct fields; Spark drops them by default
    # (ignoreNullFields=True).
    ("test_struct_with_null_field_drops_key_by_default[polarsdf]", "polarsdf"): "polarsdf to_json keeps null struct fields; Spark drops them by default",
    # Batch 22 (window functions) — polarsdf gaps vs PySpark.
    # polarsdf WindowSpec has no .rowsBetween (only .rangeBetween).
    ("test_rows_between_sets_frame[polarsdf]", "polarsdf"): "polarsdf WindowSpec has no .rowsBetween()",
    # polarsdf Column has no .over() — aggregates can't be windowed.
    ("test_sum_partition_only_whole_partition[polarsdf]", "polarsdf"): "polarsdf Column has no .over()",
    ("test_sum_running_with_order_by[polarsdf]", "polarsdf"): "polarsdf Column has no .over()",
    ("test_running_sum_with_rows_between_unbounded_preceding_to_current[polarsdf]", "polarsdf"): "polarsdf Column has no .over() and WindowSpec has no .rowsBetween()",
    ("test_sliding_three_row_window[polarsdf]", "polarsdf"): "polarsdf Column has no .over() and WindowSpec has no .rowsBetween()",
    ("test_count_over_partition[polarsdf]", "polarsdf"): "polarsdf Column has no .over()",
    ("test_min_max_mean_over_partition[polarsdf]", "polarsdf"): "polarsdf Column has no .over()",
    ("test_multi_partition_keys_sum[polarsdf]", "polarsdf"): "polarsdf Column has no .over()",
    # polarsdf rank/dense_rank/row_number require Column objects (with .asc()
    # metadata) in WindowSpec.orderBy; bare str keys raise AttributeError.
    ("test_row_number[polarsdf]", "polarsdf"): "polarsdf rank functions require Column (not str) in orderBy",
    ("test_rank_with_ties[polarsdf]", "polarsdf"): "polarsdf rank functions require Column (not str) in orderBy",
    ("test_dense_rank_with_ties[polarsdf]", "polarsdf"): "polarsdf rank functions require Column (not str) in orderBy",
    ("test_multi_order_keys_row_number[polarsdf]", "polarsdf"): "polarsdf rank functions require Column (not str) in orderBy",
    # Batch 24 (types system) — polarsdf gaps vs PySpark.
    # polarsdf does not expose a NullType class (Spark name "void").
    ("test_null_type_simple_string[polarsdf]", "polarsdf"): "polarsdf has no NullType class",
}

# Registry of (test node id substring, backend) -> reason for pythondf-specific gaps.
XFAIL_PYTHONDF: dict[tuple[str, str], str] = {}


def _maybe_xfail(request, backend: str) -> None:
    nodeid = request.node.nodeid
    registry = XFAIL_POLARSDF if backend == "polarsdf" else XFAIL_PYTHONDF
    for (substring, backend_name), reason in registry.items():
        if backend_name == backend and substring in nodeid:
            pytest.xfail(f"[{backend}] {reason}")


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    _maybe_xfail(request, request.param)
    return request.param


@pytest.fixture
def session(backend: str):
    module = importlib.import_module(f"sparkleframe.{backend}.session")
    return module.SparkSession()


@pytest.fixture
def F(backend: str):  # noqa: N802 — match PySpark conventional alias
    return importlib.import_module(f"sparkleframe.{backend}.functions")


@pytest.fixture
def DF(backend: str):  # noqa: N802
    return importlib.import_module(f"sparkleframe.{backend}.dataframe").DataFrame


@pytest.fixture
def Column(backend: str):  # noqa: N802
    return importlib.import_module(f"sparkleframe.{backend}.column").Column


@pytest.fixture
def T(backend: str):  # noqa: N802 — short alias for the backend's types module
    return importlib.import_module(f"sparkleframe.{backend}.types")


@pytest.fixture(scope="session")
def spark():
    from sparkleframe.tests.spark import spark as spark_session
    return spark_session


@pytest.fixture
def make_spark_df(spark):
    """Helper: build a PySpark DataFrame from row-oriented input matching what was used for the backend."""
    def _build(data: Iterable, schema=None):
        if schema is not None:
            return spark.createDataFrame(list(data), schema=schema)
        return spark.createDataFrame(list(data))
    return _build
