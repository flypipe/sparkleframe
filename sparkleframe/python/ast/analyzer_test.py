"""Unit tests for the analyze phase: type resolution + coercion-cast insertion."""

import pytest
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from sparkleframe.python.ast.analyzer import analyze
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    Cast,
    Literal,
    UnaryExpression,
)


def _schema(*fields):
    return StructType([StructField(name, dtype) for name, dtype in fields])


class TestResolvesTypes:
    def test_attribute_reference_from_schema(self):
        resolved = analyze(AttributeReference("a"), _schema(("a", IntegerType())))
        assert isinstance(resolved, AttributeReference)
        assert isinstance(resolved.data_type, IntegerType)

    def test_literal_inferred(self):
        resolved = analyze(Literal(3.14), schema=None)
        assert isinstance(resolved.data_type, DoubleType)

    def test_alias_carries_name_and_child_type(self):
        expr = Alias(AttributeReference("a"), "renamed")
        resolved = analyze(expr, _schema(("a", IntegerType())))
        assert isinstance(resolved, Alias)
        assert resolved.name == "renamed"
        assert isinstance(resolved.data_type, IntegerType)

    def test_col_plus_col_same_type_no_cast(self):
        expr = BinaryExpression("+", AttributeReference("a"), AttributeReference("b"))
        resolved = analyze(expr, _schema(("a", IntegerType()), ("b", IntegerType())))
        assert isinstance(resolved.data_type, IntegerType)
        assert not isinstance(resolved.left, Cast)
        assert not isinstance(resolved.right, Cast)

    def test_col_plus_col_widening_inserts_cast(self):
        expr = BinaryExpression("+", AttributeReference("a"), AttributeReference("b"))
        resolved = analyze(expr, _schema(("a", IntegerType()), ("b", LongType())))
        assert isinstance(resolved.data_type, LongType)
        assert isinstance(resolved.left, Cast)
        assert isinstance(resolved.left.data_type, LongType)
        assert not isinstance(resolved.right, Cast)

    def test_double_plus_string_casts_string_side(self):
        expr = BinaryExpression("+", AttributeReference("d"), AttributeReference("s"))
        resolved = analyze(expr, _schema(("d", DoubleType()), ("s", StringType())))
        assert isinstance(resolved.data_type, DoubleType)
        assert not isinstance(resolved.left, Cast)
        assert isinstance(resolved.right, Cast)
        assert isinstance(resolved.right.data_type, DoubleType)

    def test_col_plus_lit(self):
        expr = BinaryExpression("+", AttributeReference("a"), Literal(1))
        resolved = analyze(expr, _schema(("a", IntegerType())))
        assert isinstance(resolved.data_type, IntegerType)

    def test_pow_resolves_to_double_and_casts_non_double_operands(self):
        expr = BinaryExpression("**", AttributeReference("i"), AttributeReference("d"))
        resolved = analyze(expr, _schema(("i", IntegerType()), ("d", DoubleType())))
        assert isinstance(resolved.data_type, DoubleType)
        assert resolved.op == "**"
        assert isinstance(resolved.left, Cast)
        assert isinstance(resolved.left.data_type, DoubleType)
        assert not isinstance(resolved.right, Cast)

    def test_comparison_resolves_to_boolean_and_casts_both_sides_to_common_type(self):
        # int vs string compares as bigint: both operands get a Cast.
        expr = BinaryExpression("<", AttributeReference("i"), AttributeReference("s"))
        resolved = analyze(expr, _schema(("i", IntegerType()), ("s", StringType())))
        assert isinstance(resolved.data_type, BooleanType)
        assert resolved.op == "<"
        assert isinstance(resolved.left, Cast) and isinstance(resolved.left.data_type, LongType)
        assert isinstance(resolved.right, Cast) and isinstance(resolved.right.data_type, LongType)

    def test_logical_casts_string_operand_to_boolean(self):
        expr = BinaryExpression("and", AttributeReference("b"), AttributeReference("s"))
        resolved = analyze(expr, _schema(("b", BooleanType()), ("s", StringType())))
        assert isinstance(resolved.data_type, BooleanType)
        assert not isinstance(resolved.left, Cast)
        assert isinstance(resolved.right, Cast) and isinstance(resolved.right.data_type, BooleanType)

    def test_not_resolves_to_boolean(self):
        resolved = analyze(UnaryExpression("not", AttributeReference("b")), _schema(("b", BooleanType())))
        assert isinstance(resolved, UnaryExpression) and resolved.op == "not"
        assert isinstance(resolved.data_type, BooleanType)
        assert not isinstance(resolved.child, Cast)

    def test_original_tree_untouched(self):
        expr = AttributeReference("a")
        analyze(expr, _schema(("a", IntegerType())))
        assert expr.data_type is None


class TestUnsupportedRaises:
    def test_unsupported_node_raises(self):
        with pytest.raises(NotImplementedError):
            analyze(UnaryExpression("neg", AttributeReference("a")), _schema(("a", IntegerType())))

    def test_unsupported_operator_raises(self):
        expr = BinaryExpression("/", AttributeReference("a"), AttributeReference("b"))
        with pytest.raises(NotImplementedError):
            analyze(expr, _schema(("a", IntegerType()), ("b", IntegerType())))
