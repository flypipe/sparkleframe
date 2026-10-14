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
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple

from sparkleframe.python._errors import not_implemented_yet


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
    """Compile a Java regex; malformed patterns raise Spark's ``INVALID_PARAMETER_VALUE.PATTERN``."""
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(
            f"[INVALID_PARAMETER_VALUE.PATTERN] The value of parameter(s) `regexp` is invalid: {pattern!r} ({exc})."
        ) from None


# --------------------------------------------------------------------------- #
# Datetime patterns (Spark's DateTimeFormatter subset)
# --------------------------------------------------------------------------- #

_PATTERN_LETTERS = set("GyuYDMLdQqwWEeFaBhKkHmsSnAzVvOXxZ")
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


def _tokenize_pattern(pattern: str) -> List[Tuple[str, str]]:
    """Split a Spark datetime pattern into ``("field", "yyyy")`` / ``("literal", "-")`` tokens."""
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
            tokens.append(("field", pattern[i:j]))
            i = j
        elif ch.isalpha():
            raise ValueError(f"[INVALID_DATETIME_PATTERN] Illegal pattern character '{ch}' in {pattern!r}.")
        else:
            tokens.append(("literal", ch))
            i += 1
    return tokens


def parse_datetime(text: str, pattern: str) -> Optional[datetime]:
    """Parse ``text`` with a Spark datetime pattern; ``None`` when it does not match (caller decides strictness).

    Field widths follow ``DateTimeFormatter``: a single letter (``M``, ``d``, ``H``) reads one or
    more digits, a doubled letter (``MM``) exactly two, ``yyyy`` four or more, and ``yy`` is a
    two-digit year in 2000–2099. The whole string must be consumed and the date must exist.
    """
    fields: Dict[str, int] = {}
    pos = 0
    tokens = _tokenize_pattern(pattern)
    for index, (kind, token) in enumerate(tokens):
        if kind == "literal":
            if not text.startswith(token, pos):
                return None
            pos += len(token)
            continue
        letter, width = token[0], len(token)
        next_is_digit_field = index + 1 < len(tokens) and tokens[index + 1][0] == "field"
        if letter in "yu":
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
        elif letter in "E":
            names = _DAYS_LONG if width >= 4 else _DAYS_SHORT
            for name in names:
                if text[pos : pos + len(name)].lower() == name.lower():
                    pos += len(name)
                    break
            else:
                return None
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
            fields["offset"] = offset
        else:
            key = {
                "M": "month",
                "L": "month",
                "d": "day",
                "H": "hour",
                "h": "hour12",
                "k": "hour",
                "m": "minute",
                "s": "second",
                "D": "day_of_year",
            }.get(letter)
            if key is None:
                not_implemented_yet(f"datetime pattern letter {letter!r}")
            max_width = 2 if width == 1 and next_is_digit_field else (width if width > 1 else 9)
            digits, pos = _read_digits(text, pos, width, max_width)
            if digits is None:
                return None
            fields[key] = digits
    if pos != len(text):
        return None
    hour = fields.get("hour", 0)
    if "hour12" in fields:
        if not 1 <= fields["hour12"] <= 12:
            return None
        hour = fields["hour12"] % 12 + fields.get("ampm", 0)
    try:
        result = datetime(
            fields.get("year", 1970),
            fields.get("month", 1),
            fields.get("day", 1),
            hour,
            fields.get("minute", 0),
            fields.get("second", 0),
            fields.get("micro", 0),
        )
    except ValueError:
        return None
    if "offset" in fields:
        result = (result - fields["offset"]).replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    return result


def _read_digits(text: str, pos: int, min_width: int, max_width: int) -> Tuple[Optional[int], int]:
    end = pos
    while end < len(text) and end - pos < max_width and text[end].isdigit():
        end += 1
    if end - pos < min_width:
        return None, pos
    return int(text[pos:end]), end


_OFFSET_RE = re.compile(r"Z|[+-]\d{2}(?::?\d{2})?")


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


def format_datetime(value: datetime, pattern: str) -> str:
    """Format a (session-local, naive) datetime with a Spark datetime pattern."""
    out = []
    for kind, token in _tokenize_pattern(pattern):
        if kind == "literal":
            out.append(token)
            continue
        letter, width = token[0], len(token)
        if letter in "yu":
            out.append(f"{value.year % 100:02d}" if width == 2 else f"{value.year:0{width}d}")
        elif letter in "ML":
            if width >= 4:
                out.append(_MONTHS_LONG[value.month - 1])
            elif width == 3:
                out.append(_MONTHS_SHORT[value.month - 1])
            else:
                out.append(f"{value.month:0{width}d}")
        elif letter == "d":
            out.append(f"{value.day:0{width}d}")
        elif letter == "D":
            out.append(f"{value.timetuple().tm_yday:0{width}d}")
        elif letter in "Hk":
            out.append(f"{value.hour:0{width}d}")
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
        elif letter in "XxZ":
            out.append(_format_offset(value, letter, width))
        else:
            not_implemented_yet(f"datetime pattern letter {letter!r}")
    return "".join(out)


def _format_offset(value: datetime, letter: str, width: int) -> str:
    offset = value.astimezone().utcoffset() or timedelta(0)
    if offset == timedelta(0) and letter == "X":
        return "Z"
    total = int(offset.total_seconds() // 60)
    sign = "-" if total < 0 else "+"
    hours, minutes = divmod(abs(total), 60)
    if width == 1:
        return f"{sign}{hours:02d}" + (f"{minutes:02d}" if minutes else "")
    if width == 2 or letter == "Z":
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


_JSON_PATH_STEP_RE = re.compile(r"\.([^.\[\]]+)|\['([^']*)'\]|\[(\d+)\]|\[\*\]")


def get_json_object(text: str, path: str) -> Optional[str]:
    """Spark's ``get_json_object``: ``$``-rooted dotted / bracketed path; strings come back unquoted."""
    if not path.startswith("$"):
        return None
    try:
        node = load_json(text)
    except ValueError:
        return None
    pos = 1
    while pos < len(path):
        match = _JSON_PATH_STEP_RE.match(path, pos)
        if match is None:
            return None
        name = match.group(1) if match.group(1) is not None else match.group(2)
        if name is not None:
            if not isinstance(node, dict) or name not in node:
                return None
            node = node[name]
        elif match.group(3) is not None:
            index = int(match.group(3))
            if not isinstance(node, list) or index >= len(node):
                return None
            node = node[index]
        else:
            not_implemented_yet("get_json_object wildcard paths")
        pos = match.end()
    if node is None:
        return None
    if isinstance(node, str):
        return node
    return json_text(node)
