"""Unit tests for ``SparkSession`` — the builder and ``createDataFrame`` entry point are wired."""

from sparkleframe.python.dataframe import DataFrame
from sparkleframe.python.session import SparkSession


class TestSparkSession:
    def test_builder_chain_returns_session(self):
        session = SparkSession.builder.appName("app").master("local").config("k", "v").getOrCreate()
        assert isinstance(session, SparkSession)

    def test_create_dataframe_returns_dataframe(self):
        session = SparkSession()
        df = session.createDataFrame([(1, 2)], schema=None)
        assert isinstance(df, DataFrame)
        assert df._rows == [(1, 2)]

    def test_spark_context_set_log_level_is_noop(self):
        assert SparkSession.sparkContext.setLogLevel("INFO") is None
