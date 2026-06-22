"""Example shared parity tests for column arithmetic.

Each test is written once and runs against every engine via the ``engine``
fixture. The same assertion is checked against real Spark, so the expected
result is defined by Spark — not by any engine.

The gating in ``gaps.py`` is what makes this safe while engines are incomplete:

- ``same_type`` (int + int) is supported by both Polars and Python (runs green).
- ``numeric_plus_string`` is a documented Polars gap (still ``xfail`` for Polars)
  but is **implemented for Python**: its analyze phase resolves the operand types
  against the schema and casts the string to double, so the test is now
  required-to-pass for Python — proving Python fixed a gap Polars still has.
- ``int_plus_date`` is still gated for **both** engines (not built yet).
"""

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from sparkleframe.tests.parity.oracle import assert_matches_spark


@pytest.mark.feature("arithmetic.same_type")
def test_int_plus_int(engine, spark):
    # Row tuples + one canonical PySpark schema, shared by the engine adapter and Spark itself.
    rows = [(1, 10), (2, 20), (3, 30)]
    schema = StructType([StructField("a", IntegerType()), StructField("b", IntegerType())])
    F = engine.functions

    actual = engine.build_df(rows, schema).select((F.col("a") + F.col("b")).alias("r"))
    expected = spark.createDataFrame(rows, schema).select((SF.col("a") + SF.col("b")).alias("r"))

    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("arithmetic.numeric_plus_string")
def test_double_plus_string(engine, spark):
    # Spark implicitly casts the string operand to a number (e.g. "3.14" -> 3.14).
    rows = [(1.0, "3.14"), (2.5, "0.5")]
    schema = StructType([StructField("d", DoubleType()), StructField("s", StringType())])
    F = engine.functions

    actual = engine.build_df(rows, schema).select((F.col("d") + F.col("s")).alias("r"))
    expected = spark.createDataFrame(rows, schema).select((SF.col("d") + SF.col("s")).alias("r"))

    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("arithmetic.numeric_plus_string")
def test_int_plus_string(engine, spark):
    # With an int operand, Spark casts the string to BIGINT (e.g. 1 + "3" -> 4),
    # unlike the double case which casts to double. Decimal strings like "0.5"
    # would be a malformed BIGINT cast and raise under ANSI.
    rows = [(1, "3"), (2, "40")]
    schema = StructType([StructField("a", IntegerType()), StructField("s", StringType())])
    F = engine.functions

    actual = engine.build_df(rows, schema).select((F.col("a") + F.col("s")).alias("r"))
    expected = spark.createDataFrame(rows, schema).select((SF.col("a") + SF.col("s")).alias("r"))

    assert_matches_spark(actual, expected, engine)
