from sparkleframe.polarsdf import Column, WindowSpec


class _RankWrapper(Column):
    """
    A wrapper for deferred window function binding, enabling rank().over(...).
    """

    def __init__(self, fn):
        self._fn = fn

    def over(self, window_spec: WindowSpec) -> Column:
        return self._fn(window_spec)
