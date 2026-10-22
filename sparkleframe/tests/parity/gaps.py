"""Per-engine "not supported yet" gates for the shared parity suite.

A parity test is tagged with ``@pytest.mark.feature("<id>")``. If that id is in an
engine's gate set, the test is ``xfail`` for that engine (it may fail without
turning CI red). When an engine implements the behavior, **delete the id from its
set in the same change** — the matching parity tests then become required-to-pass.

This is the TDD ratchet: the Python set starts large (the engine is empty) and
drains to zero as the engine lands; deleting an entry is how a feature gets
"promoted" from optional to required. Each entry that the Python set still holds
but the Polars set does not is a polars gap the Python engine is expected to fix.
"""

from __future__ import annotations

# Polars engine — documented in docs/known_gaps.md. Type coercion is decided at
# expression-build time without a schema, so these cross-type cases raise instead
# of coercing the way Spark does.
POLARS_NOT_SUPPORTED = {
    "arithmetic.numeric_plus_string",
    "arithmetic.int_plus_date",
    # functions.pow does not cast its operands to double (Column ``**`` does) — see
    # docs/known_gaps.md "functions.pow operand types and Java NaN semantics".
    "functions.pow.integral_operands_return_double",
    "functions.pow.decimal_operands",
    "functions.pow.null_operands",
    "functions.pow.string_operand_casts_to_double",
    "functions.pow.java_nan_semantics",
    # Comparison / logical operators — see docs/known_gaps.md "Ordering on nested types" and
    # "String and non-boolean operands in comparisons and logical operators".
    "column.comparison.nested_nulls_and_nan",
    "column.comparison.string_implicit_cast_valid",
    "column.comparison.string_implicit_cast_malformed_raises",
    "column.logical.string_and_null_operands",
    "column.logical.malformed_string_raises",
    "column.logical.non_boolean_operand_raises",
    # Spark-only semantics the Python engine implements — see docs/known_gaps.md "Projection naming and
    # withColumns resolution", "Joins on same-named Column keys and cross joins", "explode on null, empty and
    # non-collection input", "months_between rounding", "String promotion in union", "Decimal scale in
    # createDataFrame" and "to_json decimal and timestamp output".
    "dataframe.select.expression_auto_name",
    "dataframe.with_columns.replaced_name_reads_input",
    "dataframe.join.with_duplicated_spark_keys",
    "dataframe.join.cross_keeps_duplicate_names",
    "functions.explode.drops_null_and_empty",
    "functions.explode.non_collection_raises",
    "functions.months_between",
    "dataframe.union.string_promotion",
    "dataframe.create.decimal_rescaled_to_field",
    "functions.to_json.decimal_scale_preserved",
    "functions.to_json.timestamp_format",
    # Found by the nba parity sweep — see docs/known_gaps.md "Polars deviations found by the nba parity sweep".
    "column.arithmetic.decimal_result_type",
    "column.cast.decimal_to_string_keeps_scale",
    "column.cast.timestamp_to_string",
    "column.cast.to_decimal_overflow_raises",
    "column.cast.to_decimal_rounds_half_up",
    "column.contains.literal_and_null",
    "column.divide.by_zero_raises",
    "column.divide.decimal_result_type",
    "column.divide.integral_returns_double",
    "column.eq_null_safe.mixed_types",
    "column.get_field.struct",
    "column.getitem.array_out_of_range_null",
    "column.getitem.map_missing_key_null",
    "column.isin.coerces_candidates",
    "column.isin.null_semantics",
    "column.rlike.nba_patterns",
    "column.sort.nulls_ordering",
    "column.try_cast.ddl_string_target",
    "column.try_cast.malformed_string_matrix",
    "column.try_cast.to_decimal_overflow_null",
    "dataframe.create.names_only_schema_infers_types",
    "dataframe.create.value_type_mismatch_raises",
    "dataframe.distinct.basic",
    "dataframe.drop_duplicates.all_columns",
    "dataframe.groupby.sum_decimal_result_type",
    "dataframe.is_empty",
    "dataframe.join.ambiguous_reference_raises",
    "dataframe.join.column_condition_qualified_refs",
    "dataframe.join.left_anti",
    "dataframe.to_pandas.dtypes",
    "dataframe.union.positional",
    "functions.create_map.empty",
    "functions.create_map.null_key_raises",
    "functions.first.ignorenulls",
    "functions.from_json.malformed_permissive",
    "functions.from_json.map_and_array_schemas",
    "functions.get_json_object.malformed_json_null",
    "functions.greatest.mixed_numeric_types",
    "functions.map_entries",
    "functions.round.half_up_double_and_decimal",
    "functions.to_json.options",
    "functions.try_divide.by_zero_null",
    "functions.try_to_date.nba_patterns",
    "window.dense_rank.ties",
    "window.order_by.desc_nulls_last",
    "window.row_number.without_order_raises",
    # Found by the Python engine gap work — see docs/known_gaps.md "Polars deviations found by the Python
    # engine gap work".
    "column.cast.ansi_datetime_mismatch_raises",
    "column.cast.non_finite_to_decimal_null",
    "column.cast.numeric_to_timestamp",
    "column.cast.time_only_string_to_timestamp",
    "column.cast.timestamp_to_numeric",
    "column.like",
    "dataframe.alias.qualified_references",
    "dataframe.create.ddl_schema",
    "dataframe.create.empty_inference_raises",
    "dataframe.create.from_arrow",
    "dataframe.create.from_arrow_stream",
    "dataframe.create.from_pandas",
    "dataframe.create.single_datatype_schema",
    "dataframe.create.string_field_python_str",
    "dataframe.filter.sql_string_expressions",
    "dataframe.to_arrow",
    "functions.date_format.pattern_letters",
    "functions.from_json.ddl_schema",
    "functions.get_json_object.number_rendering",
    "functions.get_json_object.wildcard",
    "functions.lit.list",
    "functions.to_timestamp.pattern_letters",
}

# Python engine — gated behaviors awaiting the build → analyze → evaluate engine
# (#102). Fully drained: every parity id passes on the Python engine, and the ratchet
# below keeps it that way — a new parity test must pass on the Python engine too.
PYTHON_NOT_IMPLEMENTED: set = set()

GATES = {
    "polars": POLARS_NOT_SUPPORTED,
    "python": PYTHON_NOT_IMPLEMENTED,
}

# Ratchet: the Python gate may only shrink. This is a committed high-water mark —
# lower it (never raise it) when you delete entries. The ratchet test enforces
# that the live set never exceeds this number, so a regression can't quietly
# re-gate a behavior instead of being fixed.
PYTHON_GATE_HIGH_WATER_MARK = 0
