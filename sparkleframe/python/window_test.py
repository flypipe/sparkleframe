"""Unit tests for ``Window`` / ``WindowSpec`` — the specs record partition and order expressions."""

from sparkleframe.python.ast.expressions import AttributeReference, SortOrder, WindowExpression
import sparkleframe.python.functions as F
from sparkleframe.python.window import Window, WindowSpec


def test_window_spec_constructs_empty():
    spec = WindowSpec()
    assert spec._partition_cols == []
    assert spec._order_cols == []


def test_builders_return_new_specs_and_keep_previous_parts():
    base = Window.partitionBy("a")
    spec = base.orderBy("b")
    assert base._order_cols == []
    assert spec._partition_cols == ["a"] and spec._order_cols == ["b"]


def test_over_builds_window_expression_with_default_ascending_order():
    node = F.row_number().over(Window.partitionBy(["a"]).orderBy(F.col("b").desc(), "c"))._expr
    assert isinstance(node, WindowExpression)
    assert [p.name for p in node.partition_by] == ["a"]
    first, second = node.order_by
    assert isinstance(first, SortOrder) and (first.ascending, first.nulls_first) == (False, False)
    assert isinstance(second.child, AttributeReference) and (second.ascending, second.nulls_first) == (True, True)
