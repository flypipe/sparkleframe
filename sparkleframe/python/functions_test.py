"""Unit tests for the ``functions`` surface — every function is a slot that raises until the engine lands."""

import pytest

import sparkleframe.python.functions as F
import sparkleframe.python.functions_helpers as functions_helpers

# Every function is a declared slot that raises until implemented. Calling each one (with dummy
# arguments) both documents the surface and guards that the slot stays wired.
_UNIMPLEMENTED = [
    lambda: F.col("a"),
    lambda: F.lit(5),
    lambda: F.when(object(), object()),
    lambda: F.abs("a"),
    lambda: F.lower("a"),
    lambda: F.count("a"),
    lambda: F.coalesce("a", "b"),
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
def test_functions_raise(call):
    with pytest.raises(NotImplementedError):
        call()


def test_functions_helpers_module_importable():
    assert isinstance(functions_helpers.__name__, str)
