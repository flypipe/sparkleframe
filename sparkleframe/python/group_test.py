"""Unit tests for ``GroupedData`` — the constructor stores state; aggregations are slots that raise."""

import pytest

from sparkleframe.python.dataframe import DataFrame
from sparkleframe.python.group import GroupedData


def _grouped() -> GroupedData:
    return GroupedData(DataFrame([(1,)], schema=None), ["a"])


def test_constructor_stores_state():
    df = DataFrame([(1,)], schema=None)
    grouped = GroupedData(df, ["a", "b"])
    assert grouped.df is df
    assert grouped.group_cols == ["a", "b"]


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.agg("a"),
        lambda g: g.count(),
        lambda g: g.sum(),
        lambda g: g.mean(),
        lambda g: g.max(),
        lambda g: g.min(),
    ],
)
def test_aggregations_raise(call):
    with pytest.raises(NotImplementedError):
        call(_grouped())
