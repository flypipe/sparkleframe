"""
Sparkleframe
"""

# ruff: noqa: F401
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

# Helps users to use `from sparkleframe import Engine` instead of `from sparkleframe.engine import Engine`
from .engine import Engine


NAME_TO_FILE_OVERRIDE = {
    # "DataFrameNaFunctions": "dataframe",
    # "DataFrameStatFunctions": "dataframe",
    # "DataFrameReader": "readwriter",
    # "DataFrameWriter": "readwriter",
    # "GroupedData": "group",
    "SparkSession": "session",
    "WindowSpec": "window",
    # "UDFRegistration": "udf",
}

ACTIVATE_CONFIG = {}


def activate() -> None:

    pyspark_mock = MagicMock()
    pyspark_mock.__file__ = "pyspark"
    sys.modules["pyspark"] = pyspark_mock
    # pyspark_mock.testing = testing
    # sys.modules["pyspark.testing"] = testing

    engine = "polarsdf"
    prefix = "Polars"
    engine_module = importlib.import_module(f"sparkleframe.{engine}")

    sys.modules["pyspark.sql"] = engine_module
    pyspark_mock.sql = engine_module
    types = engine_module.__dict__.copy()
    resolved_files = set()
    for name, obj in types.items():
        if name.startswith(prefix) or name in [
            "Column",
            "Window",
            "WindowSpec",
            "functions",
            "types",
            "Column",
            "DataFrame",
            "SparkSession",
        ]:

            name_without_prefix = name.replace(prefix, "")
            if name_without_prefix == "Session":
                name_without_prefix = "SparkSession"

            setattr(engine_module, name_without_prefix, obj)

            file = NAME_TO_FILE_OVERRIDE.get(name_without_prefix, name_without_prefix).lower()

            engine_file = importlib.import_module(f"sparkleframe.{engine}.{file}")

            if engine_file not in resolved_files:
                sys.modules[f"pyspark.sql.{file}"] = engine_file
                resolved_files.add(engine_file)

            setattr(engine_file, name_without_prefix, obj)


def deactivate() -> None:
    pyspark_imports = [k for k in sys.modules if k.startswith("pyspark")]

    for k, v in sys.modules.copy().items():
        if k in pyspark_imports:
            del sys.modules[k]
    # Try importing the pyspark imports again and see if pyspark is installed and therefore available
    # if not then nothing will change
    for k in pyspark_imports:
        try:
            sys.modules[k] = importlib.import_module(k)
        except ImportError:
            pass
    ACTIVATE_CONFIG.clear()


@contextmanager
def activate_context():
    activate()
    yield
