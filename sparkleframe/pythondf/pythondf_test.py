from sparkleframe.pythondf import functions as F
from sparkleframe.pythondf.dataframe import DataFrame
from sparkleframe.pythondf.session import SparkSession


class TestSession:
    def test_builder_returns_session(self):
        session = SparkSession.builder.appName("t").master("local").getOrCreate()
        assert isinstance(session, SparkSession)

    def test_create_from_dict_of_lists(self):
        session = SparkSession()
        df = session.createDataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        assert df.columns == ["x", "y"]
        assert df.collect() == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, {"x": 3, "y": "c"}]

    def test_create_from_list_of_dicts(self):
        session = SparkSession()
        df = session.createDataFrame([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}])
        assert df.columns == ["x", "y"]
        assert len(df) == 2

    def test_create_from_row_tuples_with_schema(self):
        session = SparkSession()
        df = session.createDataFrame([(1, "a"), (2, "b")], schema=["x", "y"])
        assert df.columns == ["x", "y"]
        assert df.collect() == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]


class TestWithColumn:
    def test_arithmetic(self):
        df = DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        out = df.withColumn("c", F.col("a") + F.col("b"))
        assert out.collect() == [
            {"a": 1, "b": 10, "c": 11},
            {"a": 2, "b": 20, "c": 22},
            {"a": 3, "b": 30, "c": 33},
        ]

    def test_lit_and_mul(self):
        df = DataFrame({"a": [1, 2, 3]})
        out = df.withColumn("c", F.col("a") * F.lit(10))
        assert [r["c"] for r in out.collect()] == [10, 20, 30]

    def test_replace_existing_column_keeps_order(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        out = df.withColumn("a", F.col("a") + F.lit(100))
        assert out.columns == ["a", "b"]
        assert [r["a"] for r in out.collect()] == [101, 102]

    def test_subtract_and_divide(self):
        df = DataFrame({"a": [10, 20], "b": [2, 4]})
        out = df.withColumn("diff", F.col("a") - F.col("b")).withColumn("ratio", F.col("a") / F.col("b"))
        rows = out.collect()
        assert [r["diff"] for r in rows] == [8, 16]
        assert [r["ratio"] for r in rows] == [5.0, 5.0]


class TestSelect:
    def test_by_name(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        out = df.select("a", "c")
        assert out.columns == ["a", "c"]
        assert out.collect() == [{"a": 1, "c": 5}, {"a": 2, "c": 6}]

    def test_with_expression_and_alias(self):
        df = DataFrame({"a": [1, 2]})
        out = df.select((F.col("a") + F.lit(1)).alias("a_plus_1"))
        assert out.columns == ["a_plus_1"]
        assert out.collect() == [{"a_plus_1": 2}, {"a_plus_1": 3}]


class TestFilter:
    def test_equality(self):
        df = DataFrame({"a": [1, 2, 3, 2], "b": ["x", "y", "z", "w"]})
        out = df.filter(F.col("a") == F.lit(2))
        assert out.collect() == [{"a": 2, "b": "y"}, {"a": 2, "b": "w"}]

    def test_greater_than(self):
        df = DataFrame({"a": [1, 2, 3, 4]})
        out = df.filter(F.col("a") > F.lit(2))
        assert [r["a"] for r in out.collect()] == [3, 4]

    def test_where_is_alias_of_filter(self):
        df = DataFrame({"a": [1, 2, 3]})
        out = df.where(F.col("a") <= F.lit(2))
        assert [r["a"] for r in out.collect()] == [1, 2]


class TestShow:
    def test_show_renders_without_error(self, capsys):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df.show()
        captured = capsys.readouterr().out
        assert "a" in captured and "b" in captured
        assert "1" in captured and "x" in captured
