"""Unit tests for ``Column`` — the build-phase operators and the unimplemented slots.

These tests pin the *AST shape* produced by Column dunder methods. They must lock both the
operator and operand placement for non-commutative reflected ops (``__rsub__``,
``__rtruediv__``, ``__rpow__``) — otherwise a build bug that swaps left/right slips through.
"""

import pytest

from sparkleframe.python.ast.expressions import (
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


class TestColumnSlotsRaise:
    @pytest.mark.parametrize(
        "call",
        [
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
    def test_unimplemented_transforms_raise(self, call):
        with pytest.raises(NotImplementedError):
            call(_col())


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
