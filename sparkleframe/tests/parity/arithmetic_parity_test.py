"""Example shared parity tests for column arithmetic.

Each test is written once and runs against every engine via the ``engine``
fixture. The same assertion is checked against real Spark, so the expected
result is defined by Spark — not by any engine.

The gating in ``gaps.py`` is what makes this safe while engines are incomplete:

- ``same_type`` (int + int) is supported by Polars (runs green) and gated for
  Python (xfail until the engine lands).
- ``numeric_plus_string`` and ``int_plus_date`` are gated for **both** Polars
  (documented gaps) and Python (not built). When Python's analyzer lands and the
  id is deleted from ``PYTHON_NOT_IMPLEMENTED``, this test becomes required-to-pass
  for Python — proving Python fixed a gap Polars still has.
"""

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from sparkleframe.tests.parity.oracle import assert_matches_spark
from sparkleframe.tests.utils import spark_rows_from_dict


@pytest.mark.feature("arithmetic.same_type")
def test_int_plus_int(engine, spark):
    data = {"a": [1, 2, 3], "b": [10, 20, 30]}
    # One canonical PySpark schema, shared by the engine adapter and by Spark itself.
    schema = StructType([StructField("a", IntegerType()), StructField("b", IntegerType())])
    F = engine.functions

    actual = engine.build_df(data, schema).select((F.col("a") + F.col("b")).alias("r"))
    expected = spark.createDataFrame(spark_rows_from_dict(data), schema).select((SF.col("a") + SF.col("b")).alias("r"))

    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("arithmetic.numeric_plus_string")
def test_double_plus_string(engine, spark):
    # Spark implicitly casts the string operand to a number (e.g. "3.14" -> 3.14).
    data = {"d": [1.0, 2.5], "s": ["3.14", "0.5"]}
    schema = StructType([StructField("d", DoubleType()), StructField("s", StringType())])
    F = engine.functions

    actual = engine.build_df(data, schema).select((F.col("d") + F.col("s")).alias("r"))
    expected = spark.createDataFrame(spark_rows_from_dict(data), schema).select((SF.col("d") + SF.col("s")).alias("r"))

    assert_matches_spark(actual, expected, engine)
