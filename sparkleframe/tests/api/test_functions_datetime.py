"""Batch 17 parity tests: date/time functions in ``F``.

Covered:
    - ``F.current_date()`` / ``F.current_timestamp()`` / ``F.now()`` — clock-skew-tolerant
      parity (timestamp values within a few seconds of PySpark; date exact).
    - ``F.to_date(col, fmt=None)`` / ``F.try_to_date`` — ISO + Spark-pattern parse;
      lenient variant returns NULL on malformed input.
    - ``F.to_timestamp(col, fmt=None)`` / ``F.try_to_timestamp`` — same for datetimes.
    - ``F.date_format(col, fmt)`` — Spark pattern → string (subset of tokens).
    - ``F.date_sub(col, days)`` — subtract calendar days.
    - ``F.datediff(end, start)`` — whole-day delta (int).
    - ``F.months_between(end, start)`` — Spark's day/31 approximation (float).

Java-pattern subset supported: ``yyyy``/``yy``/``MM``/``dd``/``HH``/``mm``/``ss``
plus ``SSS``/``SSSSSS`` for sub-second precision. Anything else is left as-is in
the format string (best-effort pass-through to ``strftime``).

All tests use the ``(session, F, spark)`` fixtures and assert parity with
native PySpark via ``assert_sparkle_spark_frame_are_equal`` unless a tolerance
window is required (clock-skew tests below).
"""
from __future__ import annotations

import datetime as _dt

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.types import (
    DateType as SparkDateType,
    StringType as SparkStringType,
    StructField as SparkStructField,
    StructType as SparkStructType,
    TimestampType as SparkTimestampType,
)

from sparkleframe.tests.utils import _records_from_sparkle, assert_sparkle_spark_frame_are_equal


_STR_SCHEMA = SparkStructType([SparkStructField("d", SparkStringType(), nullable=True)])
_DATE_SCHEMA = SparkStructType([SparkStructField("d", SparkDateType(), nullable=True)])
_TS_SCHEMA = SparkStructType([SparkStructField("t", SparkTimestampType(), nullable=True)])


# ---------------------------------------------------------------------------
# F.current_date / F.current_timestamp / F.now — clock-skew-tolerant parity.
#
# These compare backend output to PySpark within a tolerance because the two
# system clocks are read milliseconds apart in the same test.
# ---------------------------------------------------------------------------


def _backend_records(df) -> list[dict]:
    """Return rows as list[dict] for any backend (polarsdf or pythondf)."""
    return _records_from_sparkle(df)


def _backend_value(df) -> object:
    """Extract the (single) value from a sparkleframe DataFrame produced via select."""
    records = _backend_records(df)
    assert len(records) >= 1
    first = records[0]
    if isinstance(first, dict):
        return next(iter(first.values()))
    return first[0]


def _spark_value(df) -> object:
    rows = df.collect()
    assert len(rows) >= 1
    return rows[0][0]


def _parse_value(v) -> object:
    """Parse a string ISO date/datetime back into an object. Pass through otherwise."""
    if isinstance(v, _dt.date) and not isinstance(v, _dt.datetime):
        return v
    if isinstance(v, _dt.datetime):
        return v
    if isinstance(v, str):
        s = v.strip()
        # date-only?
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            try:
                return _dt.date.fromisoformat(s)
            except ValueError:
                pass
        try:
            return _dt.datetime.fromisoformat(s.replace(" ", "T", 1))
        except ValueError:
            return v
    return v


class TestCurrentDate:
    def test_current_date_matches_pyspark(self, session, F, spark):
        rows = [{"x": 1}, {"x": 2}, {"x": 3}]
        sdf = session.createDataFrame(rows).select(F.current_date().alias("d"))
        pdf = spark.createDataFrame(rows).select(SF.current_date().alias("d"))
        backend_date = _parse_value(_backend_value(sdf))
        spark_date = _spark_value(pdf)
        assert isinstance(backend_date, _dt.date)
        assert backend_date == spark_date

    def test_current_date_broadcast_to_all_rows(self, session, F):
        rows = [{"x": 1}, {"x": 2}, {"x": 3}]
        sdf = session.createDataFrame(rows).select(F.current_date().alias("d"))
        records = _backend_records(sdf)
        vals = [r["d"] if isinstance(r, dict) else r[0] for r in records]
        assert len(set(vals)) == 1
        # Each backend may normalize date to ISO str via _normalize_compare_value.
        parsed = _parse_value(vals[0])
        assert isinstance(parsed, _dt.date)


class TestCurrentTimestamp:
    def test_current_timestamp_close_to_pyspark(self, session, F, spark):
        rows = [{"x": 1}, {"x": 2}]
        sdf = session.createDataFrame(rows).select(F.current_timestamp().alias("t"))
        pdf = spark.createDataFrame(rows).select(SF.current_timestamp().alias("t"))
        backend_ts = _parse_value(_backend_value(sdf))
        spark_ts = _spark_value(pdf)
        assert isinstance(backend_ts, _dt.datetime)
        # Compare wall-clock components; both are naive local time.
        delta = abs((backend_ts - spark_ts).total_seconds())
        assert delta < 60.0, f"current_timestamp drift: {backend_ts} vs {spark_ts}"

    def test_current_timestamp_broadcast_to_all_rows(self, session, F):
        rows = [{"x": 1}, {"x": 2}, {"x": 3}]
        sdf = session.createDataFrame(rows).select(F.current_timestamp().alias("t"))
        records = _backend_records(sdf)
        vals = [r["t"] if isinstance(r, dict) else r[0] for r in records]
        assert len(set(vals)) == 1


class TestNow:
    def test_now_close_to_pyspark(self, session, F, spark):
        rows = [{"x": 1}, {"x": 2}]
        sdf = session.createDataFrame(rows).select(F.now().alias("t"))
        pdf = spark.createDataFrame(rows).select(SF.current_timestamp().alias("t"))
        backend_ts = _parse_value(_backend_value(sdf))
        spark_ts = _spark_value(pdf)
        assert isinstance(backend_ts, _dt.datetime)
        delta = abs((backend_ts - spark_ts).total_seconds())
        assert delta < 60.0, f"now() drift: {backend_ts} vs {spark_ts}"


# ---------------------------------------------------------------------------
# F.to_date / F.try_to_date
# ---------------------------------------------------------------------------


class TestToDate:
    def test_to_date_iso_default(self, session, F, spark):
        data = [("2024-01-15",), ("1997-02-28",)]
        rows = [{"d": t[0]} for t in data]
        sdf = session.createDataFrame(rows).select(F.to_date("d").alias("r"))
        pdf = spark.createDataFrame(data, schema=_STR_SCHEMA).select(SF.to_date("d").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_to_date_with_format(self, session, F, spark):
        data = [("01/15/2024",), ("12/31/2024",)]
        rows = [{"d": t[0]} for t in data]
        sdf = session.createDataFrame(rows).select(F.to_date("d", "MM/dd/yyyy").alias("r"))
        pdf = spark.createDataFrame(data, schema=_STR_SCHEMA).select(
            SF.to_date("d", "MM/dd/yyyy").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


class TestTryToDate:
    def test_try_to_date_malformed_returns_null(self, session, F, spark):
        data = [("2024-01-15",), ("not-a-date",), (None,)]
        rows = [{"d": t[0]} for t in data]
        sdf = session.createDataFrame(rows).select(F.try_to_date("d").alias("r"))
        pdf = spark.createDataFrame(data, schema=_STR_SCHEMA).select(SF.try_to_date("d").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_try_to_date_with_format_malformed_returns_null(self, session, F, spark):
        data = [("01/15/2024",), ("not-a-date",), (None,)]
        rows = [{"d": t[0]} for t in data]
        sdf = session.createDataFrame(rows).select(
            F.try_to_date("d", "MM/dd/yyyy").alias("r")
        )
        pdf = spark.createDataFrame(data, schema=_STR_SCHEMA).select(
            SF.try_to_date("d", "MM/dd/yyyy").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.to_timestamp / F.try_to_timestamp
# ---------------------------------------------------------------------------


class TestToTimestamp:
    """to_timestamp parity is checked by round-tripping through ``date_format``.

    Spark stores TIMESTAMP values in TIMESTAMP_LTZ and the JVM driver shifts
    them by the session/local TZ on ``.collect()``. Comparing the raw datetime
    objects therefore depends on the test runner's TZ; instead we format both
    sides back to a string so the wall-clock components match regardless of TZ.
    """

    def test_to_timestamp_default_format(self, session, F, spark):
        data = [("2024-01-15 10:30:45",), ("2023-06-15 14:05:09",)]
        rows = [{"d": t[0]} for t in data]
        fmt = "yyyy-MM-dd HH:mm:ss"
        sdf = session.createDataFrame(rows).select(
            F.date_format(F.to_timestamp("d"), fmt).alias("r")
        )
        pdf = spark.createDataFrame(data, schema=_STR_SCHEMA).select(
            SF.date_format(SF.to_timestamp("d"), fmt).alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_to_timestamp_with_format(self, session, F, spark):
        data = [("2024-01-15 10:30:45",), ("2023-06-15 14:05:09",)]
        rows = [{"d": t[0]} for t in data]
        fmt = "yyyy-MM-dd HH:mm:ss"
        sdf = session.createDataFrame(rows).select(
            F.date_format(F.to_timestamp("d", fmt), fmt).alias("r")
        )
        pdf = spark.createDataFrame(data, schema=_STR_SCHEMA).select(
            SF.date_format(SF.to_timestamp("d", fmt), fmt).alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


class TestTryToTimestamp:
    def test_try_to_timestamp_malformed_returns_null(self, session, F, spark):
        # Round-trip through date_format so TZ-shift doesn't affect the wall-clock
        # comparison; NULLs survive the date_format pass.
        data = [("2024-01-15 10:30:45",), ("not-a-date",), (None,)]
        rows = [{"d": t[0]} for t in data]
        fmt = "yyyy-MM-dd HH:mm:ss"
        sdf = session.createDataFrame(rows).select(
            F.date_format(F.try_to_timestamp("d"), fmt).alias("r")
        )
        pdf = spark.createDataFrame(data, schema=_STR_SCHEMA).select(
            SF.date_format(SF.try_to_timestamp("d"), fmt).alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.date_format
# ---------------------------------------------------------------------------


class TestDateFormat:
    def test_date_format_date_part(self, session, F, spark):
        data = [("2023-01-15 10:30:45",), ("2024-12-25 00:00:00",)]
        rows = [{"t": t[0]} for t in data]
        sdf = session.createDataFrame(rows).select(
            F.date_format(F.to_timestamp("t"), "yyyy-MM-dd").alias("r")
        )
        pdf = spark.createDataFrame(
            data, schema=SparkStructType([SparkStructField("t", SparkStringType(), True)])
        ).select(SF.date_format(SF.to_timestamp("t"), "yyyy-MM-dd").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_date_format_time_part(self, session, F, spark):
        data = [("2023-06-15 14:05:09",)]
        rows = [{"t": t[0]} for t in data]
        sdf = session.createDataFrame(rows).select(
            F.date_format(F.to_timestamp("t"), "HH:mm:ss").alias("r")
        )
        pdf = spark.createDataFrame(
            data, schema=SparkStructType([SparkStructField("t", SparkStringType(), True)])
        ).select(SF.date_format(SF.to_timestamp("t"), "HH:mm:ss").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_date_format_full(self, session, F, spark):
        data = [("2023-01-15 10:30:45",)]
        rows = [{"t": t[0]} for t in data]
        sdf = session.createDataFrame(rows).select(
            F.date_format(F.to_timestamp("t"), "yyyy-MM-dd HH:mm:ss").alias("r")
        )
        pdf = spark.createDataFrame(
            data, schema=SparkStructType([SparkStructField("t", SparkStringType(), True)])
        ).select(SF.date_format(SF.to_timestamp("t"), "yyyy-MM-dd HH:mm:ss").alias("r"))
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.date_sub
# ---------------------------------------------------------------------------


class TestDateSub:
    def test_date_sub_seven_days(self, session, F, spark):
        data = [(_dt.date(2024, 1, 15),), (_dt.date(2023, 12, 1),), (None,)]
        rows = [{"d": t[0]} for t in data]
        sdf = session.createDataFrame(rows).select(F.date_sub("d", 7).alias("r"))
        pdf = spark.createDataFrame(data, schema=_DATE_SCHEMA).select(
            SF.date_sub("d", 7).alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_date_sub_zero(self, session, F, spark):
        data = [(_dt.date(2024, 1, 15),)]
        rows = [{"d": t[0]} for t in data]
        sdf = session.createDataFrame(rows).select(F.date_sub("d", 0).alias("r"))
        pdf = spark.createDataFrame(data, schema=_DATE_SCHEMA).select(
            SF.date_sub("d", 0).alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.datediff
# ---------------------------------------------------------------------------


_DATE_PAIR_SCHEMA = SparkStructType(
    [
        SparkStructField("e", SparkDateType(), True),
        SparkStructField("s", SparkDateType(), True),
    ]
)


class TestDatediff:
    @pytest.mark.parametrize(
        "end_d, start_d",
        [
            (_dt.date(2024, 1, 15), _dt.date(2024, 1, 1)),   # positive: 14
            (_dt.date(2024, 1, 1), _dt.date(2024, 1, 15)),   # negative: -14
            (_dt.date(2024, 6, 1), _dt.date(2024, 6, 1)),    # zero
            (_dt.date(2025, 1, 1), _dt.date(2024, 1, 1)),    # year diff: 366 (leap)
        ],
    )
    def test_datediff_against_pyspark(self, session, F, spark, end_d, start_d):
        data = [(end_d, start_d)]
        rows = [{"e": t[0], "s": t[1]} for t in data]
        sdf = session.createDataFrame(rows).select(F.datediff("e", "s").alias("r"))
        pdf = spark.createDataFrame(data, schema=_DATE_PAIR_SCHEMA).select(
            SF.datediff("e", "s").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_datediff_null_in_either_side(self, session, F, spark):
        data = [(_dt.date(2024, 1, 15), None), (None, _dt.date(2024, 1, 1))]
        rows = [{"e": t[0], "s": t[1]} for t in data]
        sdf = session.createDataFrame(rows).select(F.datediff("e", "s").alias("r"))
        pdf = spark.createDataFrame(data, schema=_DATE_PAIR_SCHEMA).select(
            SF.datediff("e", "s").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)


# ---------------------------------------------------------------------------
# F.months_between
# ---------------------------------------------------------------------------


class TestMonthsBetween:
    def test_months_between_same_month(self, session, F, spark):
        # Same day-of-month → exact zero (whole + 0/31).
        data = [(_dt.date(2024, 3, 10), _dt.date(2024, 3, 10))]
        rows = [{"e": t[0], "s": t[1]} for t in data]
        sdf = session.createDataFrame(rows).select(F.months_between("e", "s").alias("r"))
        pdf = spark.createDataFrame(data, schema=_DATE_PAIR_SCHEMA).select(
            SF.months_between("e", "s").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_months_between_one_month_exact(self, session, F, spark):
        # Same day-of-month, 1 month apart → exact 1.0.
        data = [(_dt.date(2024, 4, 15), _dt.date(2024, 3, 15))]
        rows = [{"e": t[0], "s": t[1]} for t in data]
        sdf = session.createDataFrame(rows).select(F.months_between("e", "s").alias("r"))
        pdf = spark.createDataFrame(data, schema=_DATE_PAIR_SCHEMA).select(
            SF.months_between("e", "s").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_months_between_partial(self, session, F, spark):
        # Different day-of-month → fractional via day/31 rule.
        data = [(_dt.date(2024, 5, 20), _dt.date(2024, 3, 10))]
        rows = [{"e": t[0], "s": t[1]} for t in data]
        sdf = session.createDataFrame(rows).select(F.months_between("e", "s").alias("r"))
        pdf = spark.createDataFrame(data, schema=_DATE_PAIR_SCHEMA).select(
            SF.months_between("e", "s").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)

    def test_months_between_null_returns_null(self, session, F, spark):
        data = [(_dt.date(2024, 5, 20), None), (None, _dt.date(2024, 3, 10))]
        rows = [{"e": t[0], "s": t[1]} for t in data]
        sdf = session.createDataFrame(rows).select(F.months_between("e", "s").alias("r"))
        pdf = spark.createDataFrame(data, schema=_DATE_PAIR_SCHEMA).select(
            SF.months_between("e", "s").alias("r")
        )
        assert_sparkle_spark_frame_are_equal(sdf, pdf)
