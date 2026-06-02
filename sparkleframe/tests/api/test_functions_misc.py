"""Batch 23 parity tests: misc functions in ``F``.

Covered:
    - ``F.uuid()`` — random UUIDv4 string per row; values cannot match PySpark,
      so we assert shape (count, uniqueness, regex match, version 4).
    - ``F.rand(seed=None)`` — uniform float in ``[0.0, 1.0)`` per row. Backend
      uses a Python PRNG that does not match Spark's JVM PRNG bit-for-bit, so
      we assert per-row shape (float in unit interval) plus determinism with
      a seed within a single process.
    - ``F.monotonically_increasing_id()`` — contiguous 0..N-1 Int64 IDs in
      current row order (Spark's is partition-aware; for single-partition
      in-memory data the IDs are 0,1,2,..., which matches PySpark on a
      ``parallelism=1`` session).
    - ``F.broadcast(df)`` — no-op pass-through; broadcast is a meaningless
      planner hint for in-process backends.
"""
from __future__ import annotations

import re
import uuid as _std_uuid

import pyspark.sql.functions as SF

from sparkleframe.tests.utils import _records_from_sparkle, assert_sparkle_spark_frame_are_equal


_RE_UUID_V4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _backend_values(df, name: str) -> list:
    records = _records_from_sparkle(df)
    return [r[name] if isinstance(r, dict) else r[0] for r in records]


# ---------------------------------------------------------------------------
# F.uuid — shape-only assertions (values can't match Spark).
# ---------------------------------------------------------------------------


class TestUuid:
    def test_uuid_yields_one_v4_string_per_row(self, session, F):
        rows = [{"x": i} for i in range(20)]
        sdf = session.createDataFrame(rows).select(F.uuid().alias("u"))
        vals = _backend_values(sdf, "u")
        assert len(vals) == 20
        assert len(set(vals)) == 20  # vanishingly small collision probability
        for v in vals:
            assert isinstance(v, str)
            assert _RE_UUID_V4.match(v) is not None
            assert _std_uuid.UUID(v).version == 4

    def test_uuid_single_row(self, session, F):
        rows = [{"x": 1}]
        sdf = session.createDataFrame(rows).select(F.uuid().alias("u"))
        vals = _backend_values(sdf, "u")
        assert len(vals) == 1
        assert _RE_UUID_V4.match(vals[0]) is not None


# ---------------------------------------------------------------------------
# F.rand — uniform[0,1) per row.
# ---------------------------------------------------------------------------


class TestRand:
    def test_rand_unseeded_in_unit_interval(self, session, F):
        rows = [{"x": i} for i in range(50)]
        sdf = session.createDataFrame(rows).select(F.rand().alias("r"))
        vals = _backend_values(sdf, "r")
        assert len(vals) == 50
        for v in vals:
            assert isinstance(v, float)
            assert 0.0 <= v < 1.0

    def test_rand_seeded_is_deterministic_within_process(self, session, F):
        rows = [{"x": i} for i in range(5)]
        a = _backend_values(session.createDataFrame(rows).select(F.rand(42).alias("r")), "r")
        b = _backend_values(session.createDataFrame(rows).select(F.rand(42).alias("r")), "r")
        assert a == b
        for v in a:
            assert isinstance(v, float)
            assert 0.0 <= v < 1.0

    def test_rand_seeded_shape_matches_spark(self, session, F, spark):
        rows = [{"x": i} for i in range(5)]
        sdf = session.createDataFrame(rows).select(F.rand(42).alias("r"))
        pdf = spark.createDataFrame(rows).select(SF.rand(42).alias("r"))
        sf_vals = _backend_values(sdf, "r")
        sp_vals = [row[0] for row in pdf.collect()]
        assert len(sf_vals) == len(sp_vals)
        for v in sf_vals + sp_vals:
            assert isinstance(v, float)
            assert 0.0 <= v < 1.0


# ---------------------------------------------------------------------------
# F.monotonically_increasing_id — exact parity against single-partition Spark.
# ---------------------------------------------------------------------------


class TestMonotonicallyIncreasingId:
    def test_three_rows_contiguous_from_zero(self, session, F):
        rows = [{"k": "a"}, {"k": "b"}, {"k": "c"}]
        sdf = session.createDataFrame(rows).select(F.monotonically_increasing_id().alias("id"))
        vals = _backend_values(sdf, "id")
        assert vals == [0, 1, 2]

    def test_single_row_is_zero(self, session, F):
        rows = [{"k": "only"}]
        sdf = session.createDataFrame(rows).select(F.monotonically_increasing_id().alias("id"))
        vals = _backend_values(sdf, "id")
        assert vals == [0]

    def test_matches_spark_single_partition(self, session, F, spark):
        # Coalesce PySpark to a single partition so the bit-packed partition id is 0,
        # which makes Spark's monotonically_increasing_id() return 0, 1, 2, ...
        rows = [{"k": "a"}, {"k": "b"}, {"k": "c"}, {"k": "d"}]
        sdf = session.createDataFrame(rows).select(F.monotonically_increasing_id().alias("id"))
        pdf = (
            spark.createDataFrame(rows)
            .coalesce(1)
            .select(SF.monotonically_increasing_id().alias("id"))
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.broadcast — pass-through.
# ---------------------------------------------------------------------------


class TestBroadcast:
    def test_broadcast_returns_same_dataframe(self, session, F):
        df = session.createDataFrame([{"x": 1}, {"x": 2}])
        assert F.broadcast(df) is df

    def test_broadcast_then_join_preserves_results(self, session, F, spark):
        left_rows = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}, {"id": 3, "v": "c"}]
        right_rows = [{"id": 1, "w": "x"}, {"id": 2, "w": "y"}]
        left = session.createDataFrame(left_rows)
        right = session.createDataFrame(right_rows)
        sdf = left.join(F.broadcast(right), on="id", how="inner")

        pleft = spark.createDataFrame(left_rows)
        pright = spark.createDataFrame(right_rows)
        pdf = pleft.join(SF.broadcast(pright), on="id", how="inner")

        assert_sparkle_spark_frame_are_equal(sdf, pdf)
