"""Unit tests for ``Column`` — the build-phase operators and the unimplemented slots.

These tests pin the *AST shape* produced by Column dunder methods. They must lock both the
operator and operand placement for non-commutative reflected ops (``__rsub__``,
``__rtruediv__``, ``__rpow__``) — otherwise a build bug that swaps left/right slips through.
"""

import pytest

from sparkleframe.python.ast.expressions import (
    FunctionCall,
    SortOrder,
    Alias,
    AttributeReference,
    BinaryExpression,
    Cast,
    Literal,
    UnaryExpression,
)
from sparkleframe.python.column import Column, _column_ref, _to_expression


def _col(name: str = "a") -> Column:
    return Column(AttributeReference(name))


class TestColumnConstruction:
    def test_rejects_non_expression(self):
        with pytest.raises(TypeError):
            Column("a")  # must wrap an Expression, not a raw name

    def test_repr_includes_inner_expression(self):
        # repr must delegate to the wrapped expression so debugging output is useful.
        assert "a" in repr(_col("a"))


class TestColumnBuildPhase:
    @pytest.mark.parametrize(
        "build, op",
        [
            (lambda c: c + 1, "+"),
            (lambda c: c - 1, "-"),
            (lambda c: c * 2, "*"),
            (lambda c: c / 2, "/"),
            (lambda c: c**2, "**"),
            (lambda c: 1 + c, "+"),
            (lambda c: 1 - c, "-"),
            (lambda c: 2 * c, "*"),
            (lambda c: 2 / c, "/"),
            (lambda c: 2**c, "**"),
            (lambda c: c == 1, "=="),
            (lambda c: c != 1, "!="),
            (lambda c: c < 1, "<"),
            (lambda c: c <= 1, "<="),
            (lambda c: c > 1, ">"),
            (lambda c: c >= 1, ">="),
            (lambda c: c & _col("b"), "and"),
            (lambda c: c | _col("b"), "or"),
            (lambda c: _col("b") & c, "and"),
            (lambda c: _col("b") | c, "or"),
        ],
    )
    def test_binary_ops_build_binary_expression(self, build, op):
        result = build(_col())
        assert isinstance(result, Column)
        assert isinstance(result._expr, BinaryExpression)
        assert result._expr.op == op

    @pytest.mark.parametrize(
        "build, op",
        [
            (lambda c: 1 - c, "-"),
            (lambda c: 2 / c, "/"),
            (lambda c: 2**c, "**"),
        ],
    )
    def test_reflected_non_commutative_ops_keep_literal_on_left(self, build, op):
        """Reflected dunders (``__rsub__`` etc.) must place the literal on the left.

        Asserting ``op == "-"`` alone passes even if ``__rsub__`` swaps operands. Lock
        operand placement so a swap regression actually fails.
        """
        result = build(_col("a"))
        assert result._expr.op == op
        assert isinstance(result._expr.left, Literal)
        assert isinstance(result._expr.right, AttributeReference)
        assert result._expr.right.name == "a"

    @pytest.mark.parametrize(
        "build, op",
        [
            (lambda c: _col("b") & c, "and"),
            (lambda c: _col("b") | c, "or"),
        ],
    )
    def test_reflected_boolean_ops_keep_other_column_on_left(self, build, op):
        result = build(_col("a"))
        assert result._expr.op == op
        assert isinstance(result._expr.left, AttributeReference) and result._expr.left.name == "b"
        assert isinstance(result._expr.right, AttributeReference) and result._expr.right.name == "a"

    @pytest.mark.parametrize(
        "build, op",
        [
            (lambda c: -c, "neg"),
            (lambda c: +c, "pos"),
            (lambda c: ~c, "not"),
            (lambda c: c.isNull(), "isnull"),
            (lambda c: c.isNotNull(), "isnotnull"),
        ],
    )
    def test_unary_ops_build_unary_expression(self, build, op):
        result = build(_col())
        assert isinstance(result._expr, UnaryExpression)
        assert result._expr.op == op

    def test_alias(self):
        result = _col().alias("b")
        assert isinstance(result._expr, Alias)
        assert result._expr.name == "b"

    def test_cast_is_strict(self):
        result = _col().cast("int")
        assert isinstance(result._expr, Cast)
        assert result._expr.strict is True

    def test_try_cast_is_lenient(self):
        result = _col().try_cast("int")
        assert isinstance(result._expr, Cast)
        assert result._expr.strict is False

    def test_column_is_unhashable_like_pyspark(self):
        with pytest.raises(TypeError):
            hash(_col())


class TestColumnTransformsBuildNodes:
    def test_isin_accepts_varargs_or_a_list(self):
        for col in (_col().isin(1, 2), _col().isin([1, 2])):
            assert isinstance(col._expr, FunctionCall) and col._expr.name == "in"
            assert [c.value for c in col._expr.children[1:]] == [1, 2]

    def test_pattern_functions_keep_the_pattern_as_an_option(self):
        for method, name in ((_col().rlike, "rlike"), (_col().like, "like")):
            node = method("^x%")._expr
            assert node.name == name and node.options == {"pattern": "^x%"}
            assert node.children[0].name == "a"

    def test_get_item_and_getitem_build_the_same_node(self):
        for col in (_col().getItem("k"), _col()["k"]):
            assert col._expr.name == "get_item" and col._expr.children[1].value == "k"

    @pytest.mark.parametrize(
        "method, ascending, nulls_first",
        [
            ("asc", True, True),
            ("asc_nulls_last", True, False),
            ("desc", False, False),
            ("desc_nulls_first", False, True),
        ],
    )
    def test_sort_orders_carry_direction_and_null_placement(self, method, ascending, nulls_first):
        node = getattr(_col(), method)()._expr
        assert isinstance(node, SortOrder)
        assert (node.ascending, node.nulls_first) == (ascending, nulls_first)

    def test_column_is_not_truthy_or_iterable(self):
        with pytest.raises(ValueError, match="CANNOT_CONVERT_COLUMN_INTO_BOOL"):
            bool(_col())
        with pytest.raises(TypeError, match="NOT_ITERABLE"):
            iter(_col())


class TestExpressionCoercion:
    def test_to_expression_passthrough_and_literal(self):
        col = _col()
        assert _to_expression(col) is col._expr
        node = Literal(1)
        assert _to_expression(node) is node
        assert isinstance(_to_expression(7), Literal)

    def test_column_ref_treats_str_as_attribute(self):
        assert isinstance(_column_ref("a"), AttributeReference)
        col = _col()
        assert _column_ref(col) is col._expr
        assert isinstance(_column_ref(7), Literal)
