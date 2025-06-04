import pytest

from sparkleframe.polarsdf.window import Window, WindowSpec


class TestWindowSpec:
    @pytest.mark.parametrize(
        "partition_cols",
        [
            (["a"]),
            (["x", "y"]),
            ([]),
        ],
    )
    def test_partition_by(self, partition_cols):
        spec = Window.partitionBy(*partition_cols)
        assert isinstance(spec, WindowSpec)
        assert spec.partition_cols == partition_cols

    @pytest.mark.parametrize(
        "order_cols",
        [
            (["a"]),
            (["x", "y"]),
        ],
    )
    def test_order_by(self, order_cols):
        spec = Window.orderBy(*order_cols)
        assert isinstance(spec, WindowSpec)
        assert spec.order_cols == order_cols

    def test_partition_and_order_by(self):
        spec = Window.partitionBy("group").orderBy("value")
        assert spec.partition_cols == ["group"]
        assert spec.order_cols == ["value"]

    @pytest.mark.parametrize(
        "start, end",
        [
            (-1, 1),
            (0, 0),
            (-5, 0),
            (0, 10),
        ],
    )
    def test_range_between_valid(self, start, end):
        spec = Window.orderBy("timestamp").rangeBetween(start, end)
        assert spec.frame_start == start
        assert spec.frame_end == end

    @pytest.mark.parametrize(
        "start, end",
        [
            ("-1", 1),
            (0, "5"),
            ("a", "b"),
        ],
    )
    def test_range_between_invalid_raises(self, start, end):
        with pytest.raises(TypeError):
            Window.orderBy("timestamp").rangeBetween(start, end)
