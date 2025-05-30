import importlib
import sys
import types
from contextlib import contextmanager
from typing import Callable

from sparkleframe import Engine

SPARKLE_MODULES = [
    "functions",
    "types",
    "session",
    "dataframe",
]


@contextmanager
def mock_pyspark_modules(mock_map):
    # Save original pyspark modules
    original_modules = {k: sys.modules[k] for k in list(sys.modules) if k.startswith("pyspark")}

    # Remove existing pyspark.* modules
    for key in original_modules:
        del sys.modules[key]

    # Make sure root pyspark module exists
    if "pyspark" not in sys.modules:
        sys.modules["pyspark"] = types.ModuleType("pyspark")

    # Add parent modules for each mock
    for full_name in mock_map:
        parts = full_name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                sys.modules[parent] = types.ModuleType(parent)

    # Inject mocks
    sys.modules.update(mock_map)

    try:
        yield
    finally:
        # Clean up mocked modules
        for key in mock_map:
            pyspark_module_name = key
            if pyspark_module_name in sys.modules:
                del sys.modules[pyspark_module_name]

            sparkle_module_name = mock_map[key].__name__
            if sparkle_module_name in sys.modules:
                del sys.modules[sparkle_module_name]

        # Restore and reload original modules
        for name, module in original_modules.items():
            sys.modules[name] = module
            importlib.import_module(name)


def get_mock_map(engine: Engine) -> dict:

    sparkleframe_module = f"sparkleframe.{engine.module}"
    engine_module = importlib.import_module(sparkleframe_module)

    mock_map = {"pyspark.sql": engine_module}
    types = engine_module.__dict__.copy()
    for name, obj in types.items():
        name = engine.clean_class_name(name)

        if name in SPARKLE_MODULES:
            module_name = obj.__name__.split(".")[-1]
            pyspark_module = f"pyspark.sql.{module_name}"
            sparkle_module = f"{sparkleframe_module}.{module_name}"
            mock_map[pyspark_module] = importlib.import_module(sparkle_module)

    return mock_map


def run_with_context(func: Callable, engine: Engine = None):
    engine = engine or Engine.POLARS

    with mock_pyspark_modules(get_mock_map(engine)):
        for name in SPARKLE_MODULES:
            importlib.import_module(f"sparkleframe.{engine.module}.{name}")

        return func()
