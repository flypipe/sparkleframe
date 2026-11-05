"""Unit tests for ``GroupedData`` — orchestration only (aggregate values are owned by parity)."""

import sparkleframe.python.functions as F
from sparkleframe.python.dataframe import DataFrame
from sparkleframe.python.group import GroupedData
from sparkleframe.python.types import DecimalType, IntegerType, LongType, StringType, StructField, StructType


def _df() -> DataFrame:
    schema = StructType(
        [StructField("k", StringType()), StructField("n", IntegerType()), StructField("d", DecimalType(5, 2))]
    )
    return DataFrame([("a", 1, None), ("b", 2, None), ("a", 3, None)], schema)


def test_constructor_stores_state():
    df = _df()
    grouped = GroupedData(df, ["k", "n"])
    assert grouped.df is df
    assert grouped.group_cols == ["k", "n"]


def test_agg_outputs_group_keys_then_aggregates_with_resolved_types():
    out = _df().groupBy("k").agg(F.sum("n").alias("s"), F.sum("d").alias("sd"))
    assert out.columns == ["k", "s", "sd"]
    assert out.schema["s"].dataType == LongType()
    assert out.schema["sd"].dataType == DecimalType(15, 2)


def test_count_and_dict_form_name_columns_like_spark():
    assert _df().groupBy("k").count().columns == ["k", "count"]
    assert _df().groupBy("k").agg({"n": "max"}).columns == ["k", "max(n)"]


def test_global_aggregate_on_empty_frame_yields_one_row():
    empty = DataFrame([], _df().schema)
    assert empty.groupBy().agg(F.count("*").alias("c")).to_records() == [{"c": 0}]
