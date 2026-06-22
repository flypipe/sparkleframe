"""``SparkSession`` for the pure-Python engine.

Mirrors :mod:`sparkleframe.polarsdf.session`: the builder and ``createDataFrame`` entry point
are wired (they just construct a :class:`~sparkleframe.python.dataframe.DataFrame`); the frame's
operations are where the unimplemented work lives.
"""

from __future__ import annotations

from typing import Any, Optional

from sparkleframe.python.dataframe import DataFrame


class SparkSession:
    def __init__(self) -> None:
        self.appName_str = ""
        self.master_str = ""

    def createDataFrame(self, data: Any, schema: Optional[Any] = None) -> DataFrame:
        return DataFrame(data, schema=schema)

    class Builder:
        def appName(self, name: str) -> "SparkSession.Builder":
            self.appName_str = name
            return self

        def master(self, master_str: str) -> "SparkSession.Builder":
            self.master_str = master_str
            return self

        def getOrCreate(self) -> "SparkSession":
            return SparkSession()

        def config(self, key: Any, value: Any) -> "SparkSession.Builder":
            return self

    builder = Builder()

    class SparkContext:
        def setLogLevel(self, level: Any) -> None:
            pass

    sparkContext = SparkContext()
