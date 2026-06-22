"""Unit tests for the build-phase AST nodes.

These assert structure only — the build phase records sub-expressions and makes no type
decisions (``data_type`` stays ``None`` until the analyzer runs).
"""

from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    BinaryExpression,
    CaseWhen,
    Cast,
    Expression,
    FunctionCall,
    Literal,
    UnaryExpression,
)


class TestExpressions:
    def test_base_expression_holds_children_and_unresolved_type(self):
        child = Literal(1)
        expr = Expression([child])
        assert expr.children == [child]
        assert expr.data_type is None
        assert "Expression" in repr(expr)

    def test_literal(self):
        lit = Literal(5)
        assert lit.value == 5
        assert lit.children == []
        assert repr(lit) == "Literal(5)"

    def test_literal_with_explicit_type(self):
        lit = Literal(None, data_type="string")
        assert lit.data_type == "string"

    def test_attribute_reference(self):
        ref = AttributeReference("a")
        assert ref.name == "a"
        assert repr(ref) == "AttributeReference('a')"

    def test_binary_expression(self):
        node = BinaryExpression("+", Literal(1), AttributeReference("a"))
        assert node.op == "+"
        assert isinstance(node.left, Literal)
        assert isinstance(node.right, AttributeReference)
        assert repr(node) == "BinaryExpression('+', Literal(1), AttributeReference('a'))"

    def test_unary_expression(self):
        node = UnaryExpression("not", AttributeReference("a"))
        assert node.op == "not"
        assert isinstance(node.child, AttributeReference)
        assert "UnaryExpression('not'" in repr(node)

    def test_cast_keeps_strict_flag(self):
        strict = Cast(AttributeReference("a"), "int", strict=True)
        lenient = Cast(AttributeReference("a"), "int", strict=False)
        assert strict.strict is True
        assert lenient.strict is False
        assert strict.target_type == "int"
        assert isinstance(strict.child, AttributeReference)
        assert "strict=True" in repr(strict)

    def test_function_call(self):
        node = FunctionCall("abs", [AttributeReference("x")])
        assert node.name == "abs"
        assert node.args == node.children
        assert "FunctionCall('abs'" in repr(node)

    def test_case_when(self):
        branches = [(BinaryExpression("==", AttributeReference("a"), Literal(1)), Literal("yes"))]
        node = CaseWhen(branches, otherwise=Literal("no"))
        assert node.branches == branches
        assert isinstance(node.otherwise, Literal)
        # children = condition + value + otherwise
        assert len(node.children) == 3
        assert "CaseWhen(" in repr(node)

    def test_case_when_without_otherwise(self):
        branches = [(AttributeReference("a"), Literal(1))]
        node = CaseWhen(branches)
        assert node.otherwise is None
        assert len(node.children) == 2

    def test_alias(self):
        node = Alias(AttributeReference("a"), "b")
        assert node.name == "b"
        assert isinstance(node.child, AttributeReference)
        assert repr(node) == "Alias(AttributeReference('a'), 'b')"
