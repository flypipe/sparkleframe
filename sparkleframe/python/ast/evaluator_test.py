"""Unit tests for the evaluate phase: run resolved trees over sample rows."""

import pytest
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

from sparkleframe.python.ast.analyzer import analyze
from sparkleframe.python.ast.evaluator import evaluate
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


class TestEvaluate:
    def test_attribute_reference_reads_field(self):
        rows = [{"a": 1}, {"a": 2}]
        assert evaluate(AttributeReference("a"), rows) == [1, 2]

    def test_literal_broadcasts(self):
        rows = [{"a": 1}, {"a": 2}]
        assert evaluate(Literal(7), rows) == [7, 7]

    def test_col_plus_col(self):
        rows = [{"a": 1, "b": 10}, {"a": 2, "b": 20}]
        resolved = analyze(
            BinaryExpression("+", AttributeReference("a"), AttributeReference("b")),
            _schema(("a", IntegerType()), ("b", IntegerType())),
        )
        assert evaluate(resolved, rows) == [11, 22]

    def test_alias_passthrough(self):
        rows = [{"a": 5}]
        assert evaluate(Alias(AttributeReference("a"), "x"), rows) == [5]

    def test_cast_string_to_double(self):
        rows = [{"s": "3.14"}, {"s": "0.5"}]
        assert evaluate(Cast(AttributeReference("s"), DoubleType(), strict=True), rows) == [3.14, 0.5]

    def test_cast_numeric_widening(self):
        rows = [{"a": 1}, {"a": 2}]
        assert evaluate(Cast(AttributeReference("a"), LongType(), strict=True), rows) == [1, 2]

    def test_double_plus_string_resolved(self):
        rows = [{"d": 1.0, "s": "3.14"}, {"d": 2.5, "s": "0.5"}]
        resolved = analyze(
            BinaryExpression("+", AttributeReference("d"), AttributeReference("s")),
            _schema(("d", DoubleType()), ("s", StringType())),
        )
        assert evaluate(resolved, rows) == [pytest.approx(4.14), pytest.approx(3.0)]

    def test_null_operand_yields_null(self):
        rows = [{"a": None, "b": 1}]
        resolved = analyze(
            BinaryExpression("+", AttributeReference("a"), AttributeReference("b")),
            _schema(("a", IntegerType()), ("b", IntegerType())),
        )
        assert evaluate(resolved, rows) == [None]

    def test_unsupported_node_raises(self):
        with pytest.raises(NotImplementedError):
            evaluate(UnaryExpression("neg", AttributeReference("a")), [{"a": 1}])
