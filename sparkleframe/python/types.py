"""Type handling for the pure-Python engine.

First-pass decision (see the plan and ``docs/design/python-engine-ast.md``): the engine
consumes PySpark ``DataType`` / ``StructType`` objects **directly**. The parity adapter passes
the Spark schema straight through, so no per-engine type registry exists yet (unlike
``polarsdf/types.py``); add one only when the engine needs its own type identities.

The common PySpark type names are re-exported here for API compatibility. The import is guarded
because under ``activate()`` the real ``pyspark`` package is replaced by a mock, and the engine
must still import cleanly.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only with the real pyspark installed
    from pyspark.sql.types import (  # noqa: F401
        ArrayType,
        BinaryType,
        BooleanType,
        ByteType,
        DataType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        MapType,
        ShortType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )
except Exception:  # pragma: no cover - mock pyspark (under activate) has no real types
    pass
