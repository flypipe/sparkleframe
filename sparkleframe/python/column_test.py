"""Unit tests for ``Column`` — a declared husk whose every operator and transform raises."""

import pytest

import sparkleframe.python.column_helpers as column_helpers
from sparkleframe.python.column import Column


class TestColumnConstruction:
    def test_constructs(self):
        assert isinstance(Column(), Column)

    def test_repr(self):
        assert "Column" in repr(Column())

    def test_is_unhashable_like_pyspark(self):
        with pytest.raises(TypeError):
            hash(Column())


class TestColumnOpsRaise:
    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c + 1,
            lambda c: c - 1,
            lambda c: c * 2,
            lambda c: c / 2,
            lambda c: c**2,
            lambda c: 1 + c,
            lambda c: 1 - c,
            lambda c: 2 * c,
            lambda c: 2 / c,
            lambda c: 2**c,
            lambda c: -c,
            lambda c: +c,
            lambda c: ~c,
            lambda c: c == 1,
            lambda c: c != 1,
            lambda c: c < 1,
            lambda c: c <= 1,
            lambda c: c > 1,
            lambda c: c >= 1,
            lambda c: c & Column(),
            lambda c: c | Column(),
            lambda c: Column() & c,
            lambda c: Column() | c,
            lambda c: c.alias("b"),
            lambda c: c.cast("int"),
            lambda c: c.try_cast("int"),
            lambda c: c.isNull(),
            lambda c: c.isNotNull(),
            lambda c: c.isin(1, 2),
            lambda c: c.rlike("x"),
            lambda c: c.contains("x"),
            lambda c: c.getItem("k"),
            lambda c: c["k"],
            lambda c: c.asc(),
            lambda c: c.desc(),
            lambda c: c.asc_nulls_last(),
            lambda c: c.desc_nulls_last(),
        ],
    )
    def test_op_raises(self, call):
        with pytest.raises(NotImplementedError):
            call(Column())


def test_column_helpers_module_importable():
    # Placeholder module for shared column logic; assert it loads.
    assert isinstance(column_helpers.__name__, str)
