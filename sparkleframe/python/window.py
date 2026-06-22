"""``Window`` / ``WindowSpec`` for the pure-Python engine.

Declared so the public surface matches PySpark; the partition/order/frame builders are slots
that raise until window functions are implemented in the analyze/evaluate phases.
"""

from __future__ import annotations

from typing import List, Union

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.column import Column


class WindowSpec:
    def __init__(self) -> None:
        self._partition_cols: List[Union[str, Column]] = []
        self._order_cols: List[Union[str, Column]] = []

    def partitionBy(self, *cols: Union[str, Column]) -> "WindowSpec":
        not_implemented_yet("WindowSpec.partitionBy")

    def orderBy(self, *cols: Union[str, Column]) -> "WindowSpec":
        not_implemented_yet("WindowSpec.orderBy")

    def rangeBetween(self, start: int, end: int) -> "WindowSpec":
        not_implemented_yet("WindowSpec.rangeBetween")


class Window:
    @staticmethod
    def partitionBy(*cols: Union[str, Column]) -> WindowSpec:
        not_implemented_yet("Window.partitionBy")

    @staticmethod
    def orderBy(*cols: Union[str, Column]) -> WindowSpec:
        not_implemented_yet("Window.orderBy")
