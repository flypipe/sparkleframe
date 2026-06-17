"""Engine adapters for the shared parity harness.

Each adapter is the *only* place the harness touches a concrete engine. A parity
test never imports ``polarsdf`` or ``python`` directly — it asks the adapter to
build a frame, exposes the engine's ``functions`` module for building
expressions, and hands results to the oracle via ``to_records``.

Naming: the runtime uses ``sparkleframe.engine.Engine`` (the enum) for the engine
identifier; this module's ``EngineAdapter`` is the *test-side* adapter
implementation for one such engine. The ``ENGINES`` dict is keyed by the runtime
``Engine`` enum so the harness coverage is tied 1:1 to the declared engines.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pyspark.sql.types import StructType as SparkStructType

from sparkleframe.engine import Engine


@dataclass(frozen=True)
class EngineAdapter:
    """Everything a parity test needs to drive one engine, and nothing engine-specific leaks out."""

    name: str
    functions: Any  # the engine's ``functions`` module (col, lit, ...)
    build_df: Callable[[dict, SparkStructType], Any]  # (column_data, spark schema) -> engine DataFrame
    to_records: Callable[[Any], list]  # engine DataFrame -> list[dict] (pre-normalization)


def _column_dict_to_rows(data: dict) -> list:
    """Column-oriented ``{name: [values]}`` -> row tuples, preserving column order."""
    if not data:
        return []
    names = list(data.keys())
    return list(zip(*[data[name] for name in names]))


def _polars_engine() -> EngineAdapter:
    import sparkleframe.polarsdf.functions as functions
    from sparkleframe.polarsdf.dataframe import DataFrame
    from sparkleframe.polarsdf.types import StructField, StructType, spark_name_to_datatype

    def _to_sparkle_struct(spark_schema: SparkStructType):
        # Translate the canonical PySpark schema into the polars engine's own StructType,
        # reusing the engine's existing Spark-name -> DataType registry (_SCALAR_TYPE_REGISTRY).
        # ``simpleString()`` yields the exact registry keys ("int", "decimal(10,2)", ...).
        return StructType(
            [StructField(f.name, spark_name_to_datatype(f.dataType.simpleString())) for f in spark_schema.fields]
        )

    def build_df(data: dict, schema: SparkStructType):
        # Boss directive: always construct with an explicit schema.
        return DataFrame(_column_dict_to_rows(data), schema=_to_sparkle_struct(schema))

    def to_records(df):
        return df.to_native_df().to_dicts()

    return EngineAdapter("polars", functions, build_df, to_records)


def _python_engine() -> Optional[EngineAdapter]:
    # The pure-Python engine does not exist yet (tracked under #102). Until the
    # package is scaffolded, this adapter is None so the parametrization skips.
    if importlib.util.find_spec("sparkleframe.python") is None:
        return None

    import sparkleframe.python.functions as functions  # type: ignore[import-not-found]
    from sparkleframe.python.dataframe import DataFrame  # type: ignore[import-not-found]

    def build_df(data: dict, schema: SparkStructType):
        # The python engine will translate the PySpark schema via its OWN spark-name -> type
        # registry (the equivalent of polarsdf's spark_name_to_datatype), keeping engines split.
        return DataFrame(_column_dict_to_rows(data), schema=schema)

    def to_records(df):
        # Contract for the Python engine: expose normalized-ready records.
        return df.to_records()

    return EngineAdapter("python", functions, build_df, to_records)


# Single source of truth: the engine adapters, built eagerly at import time and
# keyed by the runtime ``Engine`` enum. Engines that don't exist yet (e.g. the
# pure-Python engine before #102 lands) are ``None`` here; callers handle that —
# the parametrized fixture skips, and co-located polars tests always use
# ``ENGINES[Engine.POLARS]`` directly.
#
#     from sparkleframe.engine import Engine
#     from sparkleframe.tests.parity.engines import ENGINES
#     from sparkleframe.tests.parity.oracle import assert_matches_spark
#     assert_matches_spark(sparkle_result, spark_expected, ENGINES[Engine.POLARS])
ENGINES: dict[Engine, Optional[EngineAdapter]] = {
    Engine.POLARS: _polars_engine(),
    Engine.PYTHON: _python_engine(),
}
