"""Unit tests for ``column_helpers``: the strict/lenient cast contract and its stated boundaries.

Parity owns the cast *values* (exercised through comparison coercion against real Spark). These
tests own what parity can't reach yet: the lenient (``try_cast``) side of the shared ``strict``
flag, and the inputs deliberately left unimplemented.
"""

from datetime import date

import pytest
from pyspark.sql.types import BooleanType, DateType, DecimalType, LongType, StringType, TimestampType

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


@pytest.mark.parametrize(
    "value",
    ["2024-01-15 10:30:45Z", "2024-01-15 10:30:45+02:00", "2024-01-15T10:30:45 UTC", "2024-01-15 10:30 Europe/Paris"],
)
def test_timestamp_with_zone_not_implemented(value):
    # Spark converts these into the session time zone, which the engine does not model yet.
    with pytest.raises(NotImplementedError):
        cast_value(value, TimestampType(), strict=True)


@pytest.mark.parametrize("value", ["10:30:45", "T10:30:45"])
def test_time_only_timestamp_not_implemented(value):
    # Spark fills in the current date in the session time zone.
    with pytest.raises(NotImplementedError):
        cast_value(value, TimestampType(), strict=True)


def test_unsupported_target_not_implemented():
    with pytest.raises(NotImplementedError):
        cast_value(date(2024, 1, 1), StringType(), strict=True)


def test_unsupported_string_target_not_implemented():
    with pytest.raises(NotImplementedError):
        cast_value("1.5", DecimalType(10, 2), strict=True)
