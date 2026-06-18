"""The single, engine-agnostic Spark-parity comparison.

There must be exactly one implementation of "does this match real Spark?" —
duplicating it per engine is the dangerous duplication (two subtly different
normalizations give false confidence). We reuse the normalization primitives
already proven in ``sparkleframe.tests.utils`` so the harness compares values
identically across engines.

(When the Polars engine is eventually retired, move those normalization helpers
into this package so the oracle no longer transitively imports ``polarsdf``.)
"""

from __future__ import annotations

import json
from typing import Any

from pyspark.sql.dataframe import DataFrame as SparkDataFrame

from sparkleframe.tests.parity.engines import EngineAdapter
from sparkleframe.tests.utils import _normalize_compare_value, _records_from_spark


def _normalized(raw_records: list) -> list:
    return [_normalize_compare_value(row) for row in raw_records]


def assert_matches_spark(
    actual: Any, expected_spark: SparkDataFrame, engine: EngineAdapter, check_row_order: bool = False
) -> bool:
    """Assert that ``actual`` (a frame from ``engine``) equals real Spark's ``expected_spark``.

    Records are extracted via the engine adapter (never via engine internals) and
    normalized through the shared comparison logic. Rows are compared as a multiset
    by default (a Spark DataFrame is unordered); pass ``check_row_order=True`` only
    when the ordering itself is under test.
    """
    assert isinstance(expected_spark, SparkDataFrame), "expected_spark must be a real PySpark DataFrame"

    actual_rows = [json.dumps(r, sort_keys=True) for r in _normalized(engine.to_records(actual))]
    spark_rows = [json.dumps(r, sort_keys=True) for r in _records_from_spark(expected_spark)]
    if not check_row_order:
        actual_rows = sorted(actual_rows)
        spark_rows = sorted(spark_rows)

    assert actual_rows == spark_rows, f"\n[{engine.name}] {actual_rows}\nvs\n[spark]  {spark_rows}"
    return True
