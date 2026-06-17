"""Canonical sample frame shared across the parity suite and polars unit tests.

Row tuples + a canonical PySpark ``StructType`` — the same shape
``spark.createDataFrame(rows, schema)`` takes. The engine adapter translates the
schema into the engine's own ``StructType``. This is the single source of truth:
both ``tests/parity/conftest.py`` (the ``sample`` fixture) and
``polarsdf/dataframe_test.py`` import these names rather than redeclaring them.
"""

from __future__ import annotations

from datetime import date, datetime

from pyspark.sql.types import DateType as SparkDateType
from pyspark.sql.types import LongType as SparkLongType
from pyspark.sql.types import StringType as SparkStringType
from pyspark.sql.types import StructField as SparkStructField
from pyspark.sql.types import StructType as SparkStructType
from pyspark.sql.types import TimestampType as SparkTimestampType

sample_data = [
    ("Alice", 25, 70000, date.fromisoformat("1990-01-01"), datetime.fromisoformat("2024-01-01T08:00:00")),
    ("Bob", 30, 80000, date.fromisoformat("1985-05-15"), datetime.fromisoformat("2024-01-02T09:30:00")),
    ("Charlie", 35, 90000, date.fromisoformat("1970-12-30"), datetime.fromisoformat("2024-01-03T11:45:00")),
]

sample_schema = SparkStructType(
    [
        SparkStructField("name", SparkStringType(), True),
        SparkStructField("age", SparkLongType(), True),
        SparkStructField("salary", SparkLongType(), True),
        SparkStructField("birth_date", SparkDateType(), True),
        SparkStructField("login_time", SparkTimestampType(), True),
    ]
)
