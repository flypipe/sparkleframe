"""Unit tests for the ``functions`` surface: build-wired functions and unimplemented slots.

As functions are implemented they must move from ``_UNIMPLEMENTED`` (slot-raises) into a
positive build-shape test that asserts the AST node produced. Removing from one list without
adding the other leaves the surface untested.
"""

import pytest

import sparkleframe.python.functions as F
from sparkleframe.python.ast.expressions import AttributeReference, CaseWhen, FunctionCall, Literal
from sparkleframe.python.column import Column


class TestBuildWiredFunctions:
    def test_col_builds_attribute_reference(self):
        result = F.col("a")
        assert isinstance(result, Column)
        assert isinstance(result._expr, AttributeReference)
        assert result._expr.name == "a"

    @pytest.mark.parametrize("value", [5, "x", True, 1.5, None])
    def test_lit_builds_literal_for_each_supported_type(self, value):
        result = F.lit(value)
        assert isinstance(result._expr, Literal)
        assert result._expr.value == value

    @pytest.mark.parametrize("fn, name", [(F.abs, "abs"), (F.lower, "lower"), (F.count, "count")])
    def test_single_arg_functions_build_function_call(self, fn, name):
        result = fn("x")
        assert isinstance(result._expr, FunctionCall)
        assert result._expr.name == name
        assert isinstance(result._expr.args[0], AttributeReference)

    def test_coalesce_is_variadic(self):
        result = F.coalesce("a", "b", F.lit(0))
        assert isinstance(result._expr, FunctionCall)
        assert result._expr.name == "coalesce"
        assert len(result._expr.args) == 3

    def test_when_otherwise_builds_case_when_preserving_branch_order(self):
        result = F.when(F.col("a") == 1, F.lit("yes")).when(F.col("a") == 2, F.lit("maybe")).otherwise(F.lit("no"))
        assert isinstance(result, Column)
        assert isinstance(result._expr, CaseWhen)
        assert len(result._expr.branches) == 2

        # Lock branch order: first .when before second .when.
        first_value = result._expr.branches[0][1]
        second_value = result._expr.branches[1][1]
        assert isinstance(first_value, Literal) and first_value.value == "yes"
        assert isinstance(second_value, Literal) and second_value.value == "maybe"

        # otherwise is the trailing literal, not folded into a branch.
        assert isinstance(result._expr.otherwise, Literal)
        assert result._expr.otherwise.value == "no"


# Every remaining function is a declared slot that raises until implemented. Calling each one
# (with dummy arguments) both documents the surface and guards that the slot stays wired.
_UNIMPLEMENTED = [
    lambda: F.get_json_object("c", "$.a"),
    lambda: F.from_json("c", "int"),
    lambda: F.to_json("c"),
    lambda: F.least("a", "b"),
    lambda: F.greatest("a", "b"),
    lambda: F.sum("a"),
    lambda: F.mean("a"),
    lambda: F.min("a"),
    lambda: F.max("a"),
    lambda: F.first("a"),
    lambda: F.map_from_entries("a"),
    lambda: F.create_map("a", "b"),
    lambda: F.array("a", "b"),
    lambda: F.map_keys("a"),
    lambda: F.collect_list("a"),
    lambda: F.collect_set("a"),
    lambda: F.transform("a", lambda x: x),
    lambda: F.round("a", 1),
    lambda: F.to_timestamp("a"),
    lambda: F.date_format("a", "yyyy"),
    lambda: F.regexp_replace("a", "x", "y"),
    lambda: F.length("a"),
    lambda: F.asc("a"),
    lambda: F.asc_nulls_first("a"),
    lambda: F.asc_nulls_last("a"),
    lambda: F.desc("a"),
    lambda: F.desc_nulls_first("a"),
    lambda: F.desc_nulls_last("a"),
    lambda: F.rank(),
    lambda: F.dense_rank(),
    lambda: F.row_number(),
    lambda: F.floor("a"),
    lambda: F.pow("a", "b"),
    lambda: F.isnan("a"),
    lambda: F.try_divide("a", "b"),
    lambda: F.initcap("a"),
    lambda: F.md5("a"),
    lambda: F.trim("a"),
    lambda: F.nullif("a", "b"),
    lambda: F.split("a", ","),
    lambda: F.substring("a", 1, 2),
    lambda: F.now(),
    lambda: F.monotonically_increasing_id(),
    lambda: F.current_timestamp(),
    lambda: F.current_date(),
    lambda: F.date_sub("a", 1),
    lambda: F.datediff("a", "b"),
    lambda: F.months_between("a", "b"),
    lambda: F.rand(),
    lambda: F.broadcast(object()),
    lambda: F.sort_array("a"),
    lambda: F.array_contains("a", 1),
    lambda: F.size("a"),
    lambda: F.filter("a", lambda x: x),
    lambda: F.explode("a"),
    lambda: F.concat("a", "b"),
    lambda: F.struct("a", "b"),
    lambda: F.try_to_timestamp("a"),
    lambda: F.to_date("a"),
    lambda: F.try_to_date("a"),
    lambda: F.try_element_at("a", 1),
    lambda: F.element_at("a", 1),
    lambda: F.uuid(),
]


@pytest.mark.parametrize("call", _UNIMPLEMENTED)
def test_unimplemented_functions_raise(call):
    with pytest.raises(NotImplementedError):
        call()
