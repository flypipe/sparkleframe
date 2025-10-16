import pytest
import polars as pl

from sparkleframe.polarsdf.column import Column
from sparkleframe.polarsdf.functions_utils import _RankWrapper


class DummyWindowSpec:
    """Minimal stand-in object for WindowSpec (type is not enforced at runtime)."""

    def __init__(self, name="win"):
        self.name = name


class Test_RankWrapper:
    def test_over_calls_fn_with_window_spec_and_returns_column(self, monkeypatch):
        called = {"flag": False, "arg": None}

        def deferred(window_spec):
            called["flag"] = True
            called["arg"] = window_spec
            # Return a simple Column (polars expr on a literal) to keep it independent of a real DF
            return Column(pl.lit(1).alias("rank"))

        wrapper = _RankWrapper(deferred)
        ws = DummyWindowSpec("p1")

        out = wrapper.over(ws)

        assert called["flag"] is True
        assert called["arg"] is ws  # exact object is passed through
        assert isinstance(out, Column)  # return type is Column

    def test_over_propagates_exceptions(self):
        class Boom(Exception):
            pass

        def deferred(_ws):
            raise Boom("kaboom")

        wrapper = _RankWrapper(deferred)
        with pytest.raises(Boom, match="kaboom"):
            wrapper.over(DummyWindowSpec())

    def test_over_returned_column_is_usable_with_polars_df(self):
        # Deferred function builds a Column derived from the data; window spec ignored for this unit test
        def deferred(_ws):
            # Example: pretend it's rank; we’ll just do x * 2 to verify the Column integrates
            return Column(pl.col("x") * 2).alias("r2")

        wrapper = _RankWrapper(deferred)
        col_expr = wrapper.over(DummyWindowSpec())
        assert isinstance(col_expr, Column)

        # Use the Column's native polars expression against a small DF
        df = pl.DataFrame({"x": [1, 3, 5]})
        out = df.with_columns(col_expr.to_native())  # col_expr already has alias "r2"

        assert out.columns == ["x", "r2"]
        assert out["r2"].to_list() == [2, 6, 10]

    @pytest.mark.parametrize("ws_name", ["wA", "wB", "wC"])
    def test_over_passes_through_different_window_specs(self, ws_name):
        seen = []

        def deferred(ws):
            seen.append(ws.name)
            return Column(pl.lit(0).alias("z"))

        wrapper = _RankWrapper(deferred)
        out = wrapper.over(DummyWindowSpec(ws_name))
        assert isinstance(out, Column)
        assert seen == [ws_name]
