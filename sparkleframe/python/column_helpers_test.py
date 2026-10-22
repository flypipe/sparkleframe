"""Unit tests for ``column_helpers``: the strict/lenient cast contract and its stated boundaries.

Parity owns the cast *values* (exercised through comparison coercion against real Spark). These
tests own what parity can't reach yet: the lenient (``try_cast``) side of the shared ``strict``
flag, and the inputs deliberately left unimplemented.
"""

import time
from datetime import date, datetime
from decimal import Decimal

import pytest
from sparkleframe.python.types import BooleanType, DateType, DecimalType, LongType, StringType, TimestampType

from sparkleframe.python.column_helpers import cast_value


@pytest.mark.parametrize(
    "value, target_type",
    [("1.5", LongType()), ("maybe", BooleanType()), ("2024-02-30", DateType()), ("2024-01-15 24:00", TimestampType())],
)
class TestStrictFlag:
    def test_strict_raises_cast_invalid_input(self, value, target_type):
        with pytest.raises(ValueError, match="CAST_INVALID_INPUT"):
            cast_value(value, target_type, strict=True)

    def test_lenient_returns_null(self, value, target_type):
        assert cast_value(value, target_type, strict=False) is None


def test_null_stays_null_regardless_of_flag():
    assert cast_value(None, DateType(), strict=True) is None


@pytest.fixture
def session_zone(monkeypatch):
    """Pin the process-local zone (the engine's session time zone) for the duration of a test."""

    def pin(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield pin
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize(
    "zone, value, expected",
    [
        ("UTC", "2024-01-15 10:30:45Z", datetime(2024, 1, 15, 10, 30, 45)),
        ("UTC", "2024-01-15 10:30:45+02:00", datetime(2024, 1, 15, 8, 30, 45)),
        ("UTC", "2024-01-15T10:30:45 UTC", datetime(2024, 1, 15, 10, 30, 45)),
        ("UTC", "2024-01-15T10:30:45GMT-0130", datetime(2024, 1, 15, 12, 0, 45)),
        ("UTC", "2024-01-15 10:30 Europe/Paris", datetime(2024, 1, 15, 9, 30)),
        ("America/Sao_Paulo", "2024-01-15T10:30:45Z", datetime(2024, 1, 15, 7, 30, 45)),
    ],
)
def test_timestamp_with_zone_converts_to_session_zone(session_zone, zone, value, expected):
    session_zone(zone)
    assert cast_value(value, TimestampType(), strict=True) == expected


@pytest.mark.parametrize("value", ["2024-01-15 10:30:45+25:00", "2024-01-15 10:30:45 Nowhere/City"])
def test_timestamp_with_malformed_zone_is_invalid_input(value):
    with pytest.raises(ValueError, match="CAST_INVALID_INPUT"):
        cast_value(value, TimestampType(), strict=True)
    assert cast_value(value, TimestampType(), strict=False) is None


@pytest.mark.parametrize("value", ["10:30:45", "T10:30:45"])
def test_time_only_timestamp_takes_todays_date(value):
    # Spark fills in the current date in the session time zone.
    result = cast_value(value, TimestampType(), strict=True)
    assert result.date() == date.today() and (result.hour, result.minute, result.second) == (10, 30, 45)


def test_date_to_string_is_iso():
    assert cast_value(date(2024, 1, 1), StringType(), strict=True) == "2024-01-01"


def test_string_to_decimal_rounds_half_up_to_scale():
    assert cast_value("1.555", DecimalType(10, 2), strict=True) == Decimal("1.56")


def test_decimal_overflow_is_strict_error_or_lenient_null():
    with pytest.raises(ArithmeticError, match="NUMERIC_VALUE_OUT_OF_RANGE"):
        cast_value("1234.5", DecimalType(4, 2), strict=True)
    assert cast_value("1234.5", DecimalType(4, 2), strict=False) is None
