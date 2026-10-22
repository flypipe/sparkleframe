"""Shared, private helpers for ``functions.py`` (pure-Python engine).

Per the repo convention (CLAUDE.md), shared/private logic for the public ``functions`` surface
lives here rather than in ``functions.py``. These are the per-value routines the evaluate phase
applies: Spark's string semantics (``substring``, ``initcap``, Java regex replacements), its
datetime patterns (parse and format), ``months_between``, and JSON reading for
``from_json`` / ``get_json_object``.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta, timezone, tzinfo
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.column_helpers import java_float_string


def spark_pow(base: Optional[float], exponent: Optional[float]) -> Optional[float]:
    """``base ** exponent`` with Spark's (Java ``StrictMath.pow``) semantics; null in → null out.

    Python's ``math.pow`` follows C99 and *raises* where Java returns a special value, and it
    also differs on two NaN cases, so each divergence is mapped explicitly:

    - ``x ** ±0`` is ``1.0`` for every ``x``, NaN included (same as C99).
    - A NaN operand otherwise yields NaN — including ``1 ** NaN`` (C99 says ``1.0``).
    - ``(±1) ** ±inf`` is NaN (C99 says ``1.0``).
    - Overflow is ``±inf`` (Python raises ``OverflowError``).
    - ``±0 ** negative`` is ``±inf`` (Python raises ``ValueError``).
    - A negative finite base to a non-integer power is NaN (Python raises ``ValueError``).
    """
    if base is None or exponent is None:
        return None
    if exponent == 0.0:
        return 1.0
    if math.isnan(base) or math.isnan(exponent):
        return math.nan
    if math.isinf(exponent) and abs(base) == 1.0:
        return math.nan
    try:
        return math.pow(base, exponent)
    except OverflowError:
        return -math.inf if base < 0 and _is_odd_integer(exponent) else math.inf
    except ValueError:
        if base == 0.0:
            # Only -0.0 raised to a negative odd integer keeps its sign.
            return -math.inf if math.copysign(1.0, base) < 0 and _is_odd_integer(exponent) else math.inf
        return math.nan


def _is_odd_integer(value: float) -> bool:
    """Whether ``value`` is a finite, odd integral double (decides the sign of ``pow``'s infinities)."""
    return math.isfinite(value) and value.is_integer() and value % 2 == 1


# --------------------------------------------------------------------------- #
# Strings
# --------------------------------------------------------------------------- #


def spark_substring(value: str, pos: int, length: int) -> str:
    """``substring(str, pos, len)`` with Spark's ``UTF8String.substringSQL`` index rules (1-based, negative from end)."""
    size = len(value)
    start = pos - 1 if pos > 0 else (size + pos if pos < 0 else 0)
    end = start + length
    start = max(start, 0)
    if start >= end:
        return ""
    return value[start : min(end, size)]


def spark_initcap(value: str) -> str:
    """Lower-case everything, then upper-case the first letter of each space-separated word."""
    lowered = value.lower()
    return "".join(ch.upper() if i == 0 or lowered[i - 1] == " " else ch for i, ch in enumerate(lowered))


def spark_trim(value: str) -> str:
    """Spark's ``trim`` removes only the space character (not tabs or newlines)."""
    return value.strip(" ")


def java_replacement(match: "re.Match[str]", replacement: str) -> str:
    """Expand a Java ``Matcher.replaceAll`` replacement: ``$n`` / ``${name}`` group refs, ``\\x`` literal ``x``."""
    out = []
    i = 0
    while i < len(replacement):
        ch = replacement[i]
        if ch == "\\" and i + 1 < len(replacement):
            out.append(replacement[i + 1])
            i += 2
        elif ch == "$" and i + 1 < len(replacement) and replacement[i + 1] == "{":
            close = replacement.index("}", i)
            out.append(match.group(replacement[i + 2 : close]) or "")
            i = close + 1
        elif ch == "$" and i + 1 < len(replacement) and replacement[i + 1].isdigit():
            j = i + 1
            # Java keeps consuming digits while the group number stays valid.
            while (
                j + 1 < len(replacement)
                and replacement[j + 1].isdigit()
                and int(replacement[i + 1 : j + 2]) <= (match.re.groups)
            ):
                j += 1
            group = int(replacement[i + 1 : j + 1])
            if group > match.re.groups:
                raise IndexError(f"No group {group}")
            out.append(match.group(group) or "")
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def compile_java_regex(pattern: str) -> "re.Pattern[str]":
    """Compile a Java regex; malformed patterns raise Spark's ``INVALID_PARAMETER_VALUE.PATTERN``.

    ``re.ASCII`` because Java's ``\\d``, ``\\w``, ``\\s``, ``\\b`` and ``(?i)`` are ASCII-only by default.
    """
    try:
        _check_java_repetitions(pattern)
        return re.compile(pattern, re.ASCII)
    except re.error as exc:
        raise ValueError(
            f"[INVALID_PARAMETER_VALUE.PATTERN] The value of parameter(s) `regexp` is invalid: {pattern!r} ({exc})."
        ) from None


def compile_like(pattern: str, escape: str = "\\") -> "re.Pattern[str]":
    """Spark's ``escapeLikeRegex``: ``_`` is one character, ``%`` any run; the escape may precede only those or itself.

    The result is matched against the whole string (``fullmatch``), across line breaks.
    """
    out, chars = [], iter(pattern)
    for char in chars:
        if char == escape:
            following = next(chars, None)
            if following is None:
                raise ValueError(
                    f"[INVALID_FORMAT.ESC_AT_THE_END] The format is invalid: '{pattern}'. The escape character is "
                    f"not allowed to end with."
                )
            if following not in ("_", "%", escape):
                raise ValueError(
                    f"[INVALID_FORMAT.ESC_IN_THE_MIDDLE] The format is invalid: '{pattern}'. The escape character "
                    f"is not allowed to precede '{following}'."
                )
            out.append(re.escape(following))
        elif char == "_":
            out.append(".")
        elif char == "%":
            out.append(".*")
        else:
            out.append(re.escape(char))
    return re.compile("".join(out), re.DOTALL)


_JAVA_QUANTIFIER_RE = re.compile(r"\{\d+(?:,\d*)?\}")


def _check_java_repetitions(pattern: str) -> None:
    """Java rejects a ``{`` that does not open a ``{n}`` / ``{n,}`` / ``{n,m}`` quantifier; Python reads it literally."""
    pos, in_class = 0, False
    while pos < len(pattern):
        ch = pattern[pos]
        if ch == "\\":
            if pattern[pos + 1 : pos + 3] in ("p{", "P{"):  # \p{Alpha}: a property name, not a quantifier
                pos = pattern.find("}", pos) + 1 or len(pattern)
            else:
                pos += 2
            continue
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "{" and not in_class and not _JAVA_QUANTIFIER_RE.match(pattern, pos):
            raise re.error("Illegal repetition", pattern, pos)
        pos += 1


# --------------------------------------------------------------------------- #
# Datetime patterns (Spark's DateTimeFormatter subset)
# --------------------------------------------------------------------------- #

_PATTERN_LETTERS = set("GyuYDMLdQqwWEeFaBhKkHmsSnAzVvOXxZpc")
_MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTHS_LONG = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_DAYS_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DAYS_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_QUARTERS_LONG = ["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]

# The widths Spark 4's formatter accepts per letter (``None``: any width). Anything else fails the way
# DateTimeFormatterHelper does: unknown widths and the week-based ``Y w W u`` are
# DATETIME_PATTERN_RECOGNITION errors, ``e c`` DATETIME_WEEK_BASED_PATTERN, and the checks below name the rest.
_FORMAT_WIDTHS: Dict[str, Optional[range]] = {
    "y": None,
    "M": range(1, 5),
    "L": range(1, 5),
    "d": range(1, 3),
    "D": range(1, 4),
    "Q": range(1, 5),
    "q": range(1, 5),
    "E": range(1, 5),
    "F": range(1, 2),
    "G": range(1, 4),
    "a": range(1, 2),
    "h": range(1, 3),
    "K": range(1, 3),
    "k": range(1, 3),
    "H": range(1, 3),
    "m": range(1, 3),
    "s": range(1, 3),
    "S": range(1, 10),
    "z": range(1, 5),
    "VV": range(2, 3),
    "O": range(1, 5),
    "X": range(1, 6),
    "x": range(1, 6),
    "Z": range(1, 6),
}
_ILLEGAL_LETTERS = set("ABnNp")
_PARSE_ILLEGAL_LETTERS = set("Qq")  # quarters only format
_PARSE_UNSUPPORTED_LETTERS = set("EF")  # day-of-week text and aligned week days only format
_LENGTH_LIMITS = {"G": 5, "M": 5, "L": 5, "E": 5, "Q": 5, "q": 5, "y": 7}


def _tokenize_pattern(pattern: str, parsing: bool) -> List[Tuple[str, str]]:
    """Split a Spark datetime pattern into ``("field", "yyyy")`` / ``("literal", "-")`` tokens.

    Raises Spark's pattern errors (``INVALID_DATETIME_PATTERN.*``,
    ``INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_PATTERN_RECOGNITION``) for letters and widths its
    formatter rejects; these are not data errors, so ``try_*`` functions raise them too.
    """
    tokens: List[Tuple[str, str]] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "'":
            close = pattern.find("'", i + 1)
            if close == -1:
                raise ValueError(f"[INVALID_DATETIME_PATTERN] Unterminated quote in pattern {pattern!r}.")
            tokens.append(("literal", pattern[i + 1 : close] or "'"))
            i = close + 1
        elif ch in _PATTERN_LETTERS:
            j = i
            while j < len(pattern) and pattern[j] == ch:
                j += 1
            _check_field(pattern, pattern[i:j], parsing)
            tokens.append(("field", pattern[i:j]))
            i = j
        elif ch.isalpha():
            raise _illegal_character(pattern, ch)
        else:
            tokens.append(("literal", ch))
            i += 1
    return tokens


def _check_field(pattern: str, token: str, parsing: bool) -> None:
    letter, width = token[0], len(token)
    if letter in "ec":
        raise ValueError(
            f"[INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_WEEK_BASED_PATTERN] You may get a different result due to "
            f"the upgrading to Spark >= 3.0: All week-based patterns are unsupported since Spark 3.0, detected "
            f"week-based character: {letter}."
        )
    if letter in _ILLEGAL_LETTERS or (parsing and letter in _PARSE_ILLEGAL_LETTERS):
        raise _illegal_character(pattern, letter)
    if width >= _LENGTH_LIMITS.get(letter, width + 1):
        raise ValueError(
            f"[INVALID_DATETIME_PATTERN.LENGTH] Unrecognized datetime pattern: {token}. Too many letters in datetime "
            f"pattern: {token}. Please reduce pattern length."
        )
    if (letter == "V" and width != 2) or (letter == "O" and width in (2, 3)):
        raise ValueError(
            f"[INVALID_DATETIME_PATTERN.WITH_SUGGESTION] Unrecognized datetime pattern: '{token}'. You can form a "
            f"valid datetime pattern with the guide from 'https://spark.apache.org/docs/latest/sql-ref-datetime-"
            f"pattern.html'."
        )
    widths = _FORMAT_WIDTHS.get("VV" if letter == "V" else letter, range(0))
    if (widths is not None and width not in widths) or (parsing and letter in _PARSE_UNSUPPORTED_LETTERS):
        raise ValueError(
            f"[INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_PATTERN_RECOGNITION] You may get a different result due "
            f"to the upgrading to Spark >= 3.0: Fail to recognize '{pattern}' pattern in the DateTimeFormatter."
        )


def _illegal_character(pattern: str, letter: str) -> ValueError:
    return ValueError(
        f"[INVALID_DATETIME_PATTERN.ILLEGAL_CHARACTER] Unrecognized datetime pattern: {pattern}. Illegal pattern "
        f"character found in datetime pattern: {letter}. Please provide legal character."
    )


def parse_datetime(text: str, pattern: str) -> Optional[datetime]:
    """Parse ``text`` with a Spark datetime pattern; ``None`` when it does not match (caller decides strictness).

    Field widths follow ``DateTimeFormatter``: a single letter (``M``, ``d``, ``H``) reads one or
    more digits, a doubled letter (``MM``) exactly two, ``yyyy`` four or more, and ``yy`` is a
    two-digit year in 2000–2099. The whole string must be consumed and the date must exist; a
    day-of-year (``D``) must fall in the year and agree with any month or day also given.
    """
    fields: Dict[str, Any] = {}
    pos = 0
    tokens = _tokenize_pattern(pattern, parsing=True)
    for index, (kind, token) in enumerate(tokens):
        if kind == "literal":
            if not text.startswith(token, pos):
                return None
            pos += len(token)
            continue
        letter, width = token[0], len(token)
        next_is_digit_field = index + 1 < len(tokens) and tokens[index + 1][0] == "field"
        if letter == "y":
            if width == 2:
                digits, pos = _read_digits(text, pos, 2, 2)
                if digits is None:
                    return None
                fields["year"] = 2000 + digits
            else:
                digits, pos = _read_digits(text, pos, max(width, 1), 4 if next_is_digit_field else 10)
                if digits is None:
                    return None
                fields["year"] = digits
        elif letter in "ML" and width >= 3:
            names = _MONTHS_LONG if width >= 4 else _MONTHS_SHORT
            for month, name in enumerate(names, start=1):
                if text[pos : pos + len(name)].lower() == name.lower():
                    fields["month"], pos = month, pos + len(name)
                    break
            else:
                return None
        elif letter == "G":
            if text[pos : pos + 2].upper() != "AD":
                return None
            pos += 2
        elif letter == "a":
            marker = text[pos : pos + 2].upper()
            if marker not in ("AM", "PM"):
                return None
            fields["ampm"], pos = (0 if marker == "AM" else 12), pos + 2
        elif letter == "S":
            digits_text = text[pos : pos + width]
            if len(digits_text) != width or not digits_text.isdigit():
                return None
            fields["micro"] = int(digits_text[:6].ljust(6, "0"))
            pos += width
        elif letter in "XxZ":
            offset, pos = _read_offset(text, pos)
            if offset is None:
                return None
            fields["zone"] = timezone(offset)
        elif letter in "zVO":
            zone, pos = _read_zone(text, pos, letter)
            if zone is None:
                return None
            fields["zone"] = zone
        else:
            key = {
                "M": "month",
                "L": "month",
                "d": "day",
                "H": "hour",
                "h": "hour12",
                "K": "hour_of_ampm",
                "k": "clock_hour",
                "m": "minute",
                "s": "second",
                "D": "day_of_year",
            }[letter]
            max_width = 2 if width == 1 and next_is_digit_field else (width if width > 1 else 9)
            digits, pos = _read_digits(text, pos, width, max_width)
            if digits is None:
                return None
            fields[key] = digits
    if pos != len(text):
        return None
    hour = _parsed_hour(fields)
    if hour is None:
        return None
    try:
        day = _parsed_date(fields)
        if day is None:
            return None
        result = datetime.combine(
            day, datetime.min.time().replace(hour=hour, minute=fields.get("minute", 0), second=fields.get("second", 0))
        ).replace(microsecond=fields.get("micro", 0))
    except ValueError:
        return None
    if "zone" in fields:
        result = result.replace(tzinfo=fields["zone"]).astimezone().replace(tzinfo=None)
    return result


def _parsed_hour(fields: Dict[str, Any]) -> Optional[int]:
    """The hour of day from ``H``, ``h``/``K`` (with ``a``) or ``k`` (1-24, where 24 is midnight)."""
    if "hour12" in fields:
        return fields["hour12"] % 12 + fields.get("ampm", 0) if 1 <= fields["hour12"] <= 12 else None
    if "hour_of_ampm" in fields:
        return fields["hour_of_ampm"] + fields.get("ampm", 0) if fields["hour_of_ampm"] <= 11 else None
    if "clock_hour" in fields:
        return fields["clock_hour"] % 24 if 1 <= fields["clock_hour"] <= 24 else None
    hour = fields.get("hour", 0)
    return hour if hour <= 23 else None


def _parsed_date(fields: Dict[str, Any]) -> Optional[date]:
    year = fields.get("year", 1970)
    if "day_of_year" not in fields:
        return date(year, fields.get("month", 1), fields.get("day", 1))
    day = date(year, 1, 1) + timedelta(days=fields["day_of_year"] - 1)
    if not 1 <= fields["day_of_year"] or day.year != year:
        return None
    if fields.get("month", day.month) != day.month or fields.get("day", day.day) != day.day:
        return None  # DateTimeFormatter's "Conflict found" between the day-of-year and the month/day
    return day


def _read_digits(text: str, pos: int, min_width: int, max_width: int) -> Tuple[Optional[int], int]:
    end = pos
    while end < len(text) and end - pos < max_width and text[end].isdigit():
        end += 1
    if end - pos < min_width:
        return None, pos
    return int(text[pos:end]), end


_OFFSET_RE = re.compile(r"Z|[+-]\d{2}(?::?\d{2})?")
_GMT_OFFSET_RE = re.compile(r"GMT(?:([+-])(\d{1,2})(?::(\d{2}))?)?")
_ZONE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_/+\-]*")
# Java's zone-text parser maps the common North American abbreviations to their regions (so DST applies).
_ZONE_ABBREVIATIONS = {
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
}


def _read_offset(text: str, pos: int) -> Tuple[Optional[timedelta], int]:
    match = _OFFSET_RE.match(text, pos)
    if match is None:
        return None, pos
    token = match.group(0)
    if token == "Z":
        return timedelta(0), match.end()
    sign = -1 if token[0] == "-" else 1
    digits = token[1:].replace(":", "")
    return sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:4] or 0)), match.end()


def _read_zone(text: str, pos: int, letter: str) -> Tuple[Optional[tzinfo], int]:
    """A zone name (``z``: ``UTC``, ``PST``), region id (``VV``) or localized offset (``O``: ``GMT+2``)."""
    if letter == "O":
        match = _GMT_OFFSET_RE.match(text, pos)
        if match is None:
            return None, pos
        if match.group(1) is None:
            return timezone.utc, match.end()
        offset = timedelta(hours=int(match.group(2)), minutes=int(match.group(3) or 0))
        return timezone(-offset if match.group(1) == "-" else offset), match.end()
    match = _ZONE_NAME_RE.match(text, pos)
    if match is None:
        return None, pos
    name = match.group(0)
    if name in ("UTC", "GMT", "UT", "Z"):
        return timezone.utc, match.end()
    try:
        return ZoneInfo(_ZONE_ABBREVIATIONS.get(name, name) if letter == "z" else name), match.end()
    except (ValueError, ZoneInfoNotFoundError):
        return None, pos


def format_datetime(value: datetime, pattern: str) -> str:
    """Format a (session-local, naive) datetime with a Spark datetime pattern."""
    out = []
    for kind, token in _tokenize_pattern(pattern, parsing=False):
        if kind == "literal":
            out.append(token)
            continue
        letter, width = token[0], len(token)
        if letter == "y":
            out.append(f"{value.year % 100:02d}" if width == 2 else f"{value.year:0{width}d}")
        elif letter in "ML":
            if width >= 4:
                out.append(_MONTHS_LONG[value.month - 1])
            elif width == 3:
                out.append(_MONTHS_SHORT[value.month - 1])
            else:
                out.append(f"{value.month:0{width}d}")
        elif letter in "Qq":
            quarter = (value.month - 1) // 3 + 1
            out.append({3: f"Q{quarter}", 4: _QUARTERS_LONG[quarter - 1]}.get(width, f"{quarter:0{width}d}"))
        elif letter == "d":
            out.append(f"{value.day:0{width}d}")
        elif letter == "D":
            out.append(f"{value.timetuple().tm_yday:0{width}d}")
        elif letter == "F":
            out.append(str((value.day - 1) % 7 + 1))  # aligned day-of-week in month
        elif letter == "G":
            out.append("AD")
        elif letter == "H":
            out.append(f"{value.hour:0{width}d}")
        elif letter == "k":
            out.append(f"{value.hour or 24:0{width}d}")
        elif letter == "K":
            out.append(f"{value.hour % 12:0{width}d}")
        elif letter == "h":
            out.append(f"{(value.hour % 12) or 12:0{width}d}")
        elif letter == "m":
            out.append(f"{value.minute:0{width}d}")
        elif letter == "s":
            out.append(f"{value.second:0{width}d}")
        elif letter == "S":
            out.append(f"{value.microsecond:06d}"[:width].ljust(width, "0"))
        elif letter == "a":
            out.append("AM" if value.hour < 12 else "PM")
        elif letter == "E":
            out.append((_DAYS_LONG if width >= 4 else _DAYS_SHORT)[value.weekday()])
        elif letter in "XxZO":
            out.append(_format_offset(value, letter, width))
        elif letter == "z":
            out.append(_format_zone_name(value, width))
        else:
            not_implemented_yet(f"datetime pattern letter {token!r}")
    return "".join(out)


def _format_zone_name(value: datetime, width: int) -> str:
    """``z``: the zone's abbreviation (``UTC``, ``EST``); ``zzzz`` its full name, known here only for UTC."""
    name = value.astimezone().tzname() or ""
    if width < 4:
        return name
    if name == "UTC":
        return "Coordinated Universal Time"
    return not_implemented_yet(f"the full time-zone name 'zzzz' for {name}")


def _format_offset(value: datetime, letter: str, width: int) -> str:
    offset = value.astimezone().utcoffset() or timedelta(0)
    total = int(offset.total_seconds() // 60)
    sign = "-" if total < 0 else "+"
    hours, minutes = divmod(abs(total), 60)
    if letter == "O" or (letter == "Z" and width == 4):
        # Localized offset: ``GMT``, ``GMT+2`` / ``GMT+5:30`` (``O``), ``GMT+02:00`` (``OOOO`` / ``ZZZZ``).
        if total == 0:
            return "GMT"
        if width == 1:
            return f"GMT{sign}{hours}" + (f":{minutes:02d}" if minutes else "")
        return f"GMT{sign}{hours:02d}:{minutes:02d}"
    if letter == "Z":
        return (
            f"{sign}{hours:02d}{minutes:02d}"
            if width < 4
            else ("Z" if total == 0 else f"{sign}{hours:02d}:{minutes:02d}")
        )
    if total == 0 and letter == "X":
        return "Z"
    if width == 1:
        return f"{sign}{hours:02d}" + (f"{minutes:02d}" if minutes else "")
    if width in (2, 4):
        return f"{sign}{hours:02d}{minutes:02d}"
    return f"{sign}{hours:02d}:{minutes:02d}"


def months_between(end: datetime, start: datetime, round_off: bool) -> float:
    """Spark's ``months_between``: whole months when both days match (or both are month-ends), else /31 days."""
    month_diff = (end.year * 12 + end.month) - (start.year * 12 + start.month)
    if end.day == start.day or (_is_last_day(end) and _is_last_day(start)):
        return float(month_diff)
    seconds_in_day = 24 * 3600
    end_seconds = end.hour * 3600 + end.minute * 60 + end.second + end.microsecond / 1e6
    start_seconds = start.hour * 3600 + start.minute * 60 + start.second + start.microsecond / 1e6
    seconds_diff = (end.day - start.day) * seconds_in_day + end_seconds - start_seconds
    result = month_diff + seconds_diff / (31 * seconds_in_day)
    if round_off:
        return float(Decimal(repr(result)).quantize(Decimal("1E-8"), rounding=ROUND_HALF_UP))
    return result


def _is_last_day(value: date) -> bool:
    return (value + timedelta(days=1)).month != value.month


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


class JsonNumber(Decimal):
    """A JSON number that remembers its source text (Spark copies numbers verbatim into string results)."""

    text: str

    def __new__(cls, text: str) -> "JsonNumber":
        number = Decimal.__new__(cls, text)
        number.text = text
        return number


def _first_key_wins(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        result.setdefault(key, value)
    return result


def load_json(text: str) -> Any:
    """Parse JSON keeping number source text; raises ``ValueError`` on malformed input."""
    return json.loads(text, parse_float=JsonNumber, parse_int=JsonNumber, object_pairs_hook=_first_key_wins)


def json_text(value: Any) -> str:
    """Compact JSON text of a parsed value, numbers written as in the source."""
    if isinstance(value, JsonNumber):
        return value.text
    if isinstance(value, dict):
        return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{json_text(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(json_text(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def _jackson_text(value: Any) -> str:
    """JSON text the way Jackson's ``copyCurrentStructure`` writes it for ``get_json_object``.

    Integers keep their digits; any other number is read as a double and written with Java's
    ``Double.toString`` (``1.50`` is ``1.5``, ``1e2`` is ``100.0``), a non-finite one as a string.
    """
    if isinstance(value, JsonNumber):
        if not re.search(r"[.eE]", value.text):
            return value.text
        number = float(value.text)
        return java_float_string(number) if math.isfinite(number) else f'"{java_float_string(number)}"'
    if isinstance(value, dict):
        return (
            "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{_jackson_text(v)}" for k, v in value.items()) + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(_jackson_text(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def _double_quoted(text: str) -> str:
    """Rewrite single-quoted JSON strings as double-quoted ones (``get_json_object`` accepts ``{'a': 1}``)."""
    if "'" not in text:
        return text
    out: List[str] = []
    pos = 0
    while pos < len(text):
        ch = text[pos]
        if ch not in "\"'":
            out.append(ch)
            pos += 1
            continue
        end = pos + 1
        while end < len(text) and text[end] != ch:
            end += 2 if text[end] == "\\" else 1
        if end >= len(text):
            raise ValueError("unterminated string")
        body = text[pos + 1 : end]
        if ch == "'":
            # A literal `"` needs escaping once double-quoted, and `\'` is just a quote.
            body = re.sub(r'(?<!\\)"', r"\"", body.replace("\\'", "'"))
        out.append(f'"{body}"')
        pos = end + 1
    return "".join(out)


_JSON_PATH_STEP_RE = re.compile(r"\.([^.\[\]]+)|\['([^']*)'\]|\[(\d+)\]|\[\*\]")


def get_json_object(text: str, path: str) -> Optional[str]:
    """Spark's ``get_json_object``: ``$``-rooted dotted / bracketed path; strings come back unquoted."""
    if not path.startswith("$"):
        return None
    try:
        node = load_json(_double_quoted(text))
    except ValueError:
        return None
    steps: List[Tuple[str, Any]] = []
    pos = 1
    while pos < len(path):
        match = _JSON_PATH_STEP_RE.match(path, pos)
        if match is None:
            return None
        name = match.group(1) if match.group(1) is not None else match.group(2)
        if name is not None:
            steps.append(("key", name))
        elif match.group(3) is not None:
            steps.append(("index", int(match.group(3))))
        else:
            steps.append(("wildcard", None))
        pos = match.end()
    if any(kind == "wildcard" for kind, _ in steps):
        out: List[str] = []
        return "".join(out) if _evaluate_json_path(node, steps, _RAW, out) else None
    for kind, step in steps:
        if kind == "key":
            if not isinstance(node, dict) or step not in node:
                return None
            node = node[step]
        else:
            if not isinstance(node, list) or step >= len(node):
                return None
            node = node[step]
    if node is None:
        return None
    if isinstance(node, str):
        return node
    return _jackson_text(node)


_RAW, _QUOTED, _FLATTEN = "raw", "quoted", "flatten"


def _evaluate_json_path(node: Any, steps: List[Tuple[str, Any]], style: str, out: List[str]) -> bool:
    """Spark's ``GetJsonObject.evaluatePath`` for paths with ``[*]``: append the JSON written, return whether it matched.

    A top-level string is written unquoted (raw); under a wildcard values are quoted JSON, and a
    wildcard that matched exactly one element drops its outer array. ``[*][*]`` flattens nested
    arrays, and a field holding JSON null does not match.
    """
    if not steps:
        if isinstance(node, str) and style == _RAW:
            out.append(node)
            return True
        if isinstance(node, list) and style == _FLATTEN:
            matched = False
            for item in node:
                matched |= _evaluate_json_path(item, [], style, out)
            return matched
        out.append(_jackson_text(node))
        return True
    (kind, step), rest = steps[0], steps[1:]
    if kind == "key":
        if not isinstance(node, dict) or node.get(step) is None:
            return False
        return _evaluate_json_path(node[step], rest, style, out)
    if not isinstance(node, list):
        return False
    if kind == "index":
        if step >= len(node):
            return False
        next_style = _QUOTED if rest and rest[0][0] == "wildcard" else style
        return _evaluate_json_path(node[step], rest, next_style, out)
    inner: List[str] = []
    if rest and rest[0][0] == "wildcard" and style != _QUOTED:  # [*][*]: Hive's flattening double wildcard
        matched = False
        for item in node:
            matched |= _evaluate_json_path(item, rest[1:], _FLATTEN, inner)
        out.append("[" + ",".join(inner) + "]")
        return matched
    if style == _QUOTED:
        matched = False
        for item in node:
            matched |= _evaluate_json_path(item, rest, _QUOTED, inner)
        out.append("[" + ",".join(inner) + "]")
        return matched
    count = sum(_evaluate_json_path(item, rest, _QUOTED if style == _RAW else _FLATTEN, inner) for item in node)
    if count:
        out.append(",".join(inner) if count == 1 else "[" + ",".join(inner) + "]")
    return count > 0
