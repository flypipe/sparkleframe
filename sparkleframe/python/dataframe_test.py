"""Unit tests for ``DataFrame``: the constructor stores state; every action is a slot that raises."""

import pytest

import sparkleframe.python.dataframe_helpers as dataframe_helpers
from sparkleframe.python.dataframe import DataFrame


class TestDataFrameConstruction:
    def test_stores_rows_and_schema(self):
        rows = [(1, 2), (3, 4)]
        df = DataFrame(rows, schema="schema-sentinel")
        assert df._rows == rows
        assert df._schema == "schema-sentinel"

    def test_none_data_becomes_empty(self):
        df = DataFrame(None)
        assert df._rows == []
        assert df._schema is None


class TestDataFrameActionsRaise:
    @pytest.mark.parametrize(
        "call",
        [
            lambda df: df.to_records(),
            lambda df: df.columns(),
            lambda df: df.schema(),
            lambda df: df.dtypes(),
            lambda df: df["a"],
            lambda df: df.alias("x"),
            lambda df: df.select("a"),
            lambda df: df.filter("a > 1"),
            lambda df: df.where("a > 1"),
            lambda df: df.withColumn("b", None),
            lambda df: df.withColumns({}),
            lambda df: df.withColumnRenamed("a", "b"),
            lambda df: df.drop("a"),
            lambda df: df.distinct(),
            lambda df: df.dropDuplicates(),
            lambda df: df.union(df),
            lambda df: df.unionByName(df),
            lambda df: df.join(df),
            lambda df: df.groupBy("a"),
            lambda df: df.groupby("a"),
            lambda df: df.sort("a"),
            lambda df: df.orderBy("a"),
            lambda df: df.fillna(0),
            lambda df: df.count(),
            lambda df: df.show(),
            lambda df: df.toPandas(),
            lambda df: df.to_arrow(),
        ],
    )
    def test_action_raises(self, call):
        df = DataFrame([(1,)], schema=None)
        with pytest.raises(NotImplementedError):
            call(df)


def test_dataframe_helpers_module_importable():
    assert isinstance(dataframe_helpers.__name__, str)
