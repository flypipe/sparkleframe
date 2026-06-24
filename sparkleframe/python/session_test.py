"""Unit tests for ``SparkSession`` — the builder and ``createDataFrame`` entry point are wired."""

from pyspark.sql.types import IntegerType, StructField, StructType

from sparkleframe.python.dataframe import DataFrame
from sparkleframe.python.session import SparkSession


class TestSparkSession:
    def test_builder_chain_returns_session(self):
        session = SparkSession.builder.appName("app").master("local").config("k", "v").getOrCreate()
        assert isinstance(session, SparkSession)

    def test_create_dataframe_passes_rows_and_schema_through(self):
        schema = StructType([StructField("a", IntegerType()), StructField("b", IntegerType())])
        session = SparkSession()
        df = session.createDataFrame([(1, 2), (3, 4)], schema=schema)

        assert isinstance(df, DataFrame)
        assert df._rows == [(1, 2), (3, 4)]
        # Schema must be passed through (same instance), not copied or normalized away.
        assert df._schema is schema

    def test_spark_context_set_log_level_is_documented_noop(self):
        # Spark API surface compatibility: ``setLogLevel`` exists and returns ``None`` so
        # caller code written against PySpark continues to type-check and run.
        assert SparkSession.sparkContext.setLogLevel("INFO") is None
