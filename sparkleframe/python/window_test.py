"""Unit tests for ``Window`` / ``WindowSpec`` — declared surface; builders are slots that raise."""

import pytest

from sparkleframe.python.window import Window, WindowSpec


def test_window_spec_constructs_empty():
    spec = WindowSpec()
    assert spec._partition_cols == []
    assert spec._order_cols == []


@pytest.mark.parametrize(
    "call",
    [
        lambda: WindowSpec().partitionBy("a"),
        lambda: WindowSpec().orderBy("a"),
        lambda: WindowSpec().rangeBetween(0, 1),
        lambda: Window.partitionBy("a"),
        lambda: Window.orderBy("a"),
    ],
)
def test_builders_raise(call):
    with pytest.raises(NotImplementedError):
        call()
