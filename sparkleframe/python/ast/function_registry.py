"""Per-function analyze and evaluate rules for :class:`~sparkleframe.python.ast.expressions.FunctionCall`.

Every ``pyspark.sql.functions`` entry the engine supports is a :class:`FunctionSpec` keyed by the
``FunctionCall.name`` the build phase emits:

- ``resolve(node, rctx)`` runs in the **analyze** phase on a node whose arguments are already
  resolved. It inserts Spark's implicit argument casts (by replacing ``node.children[i]``) and
  returns the result ``DataType``. Analysis errors use Spark's error classes.
- ``evaluate(values, node, ctx)`` runs in the **evaluate** phase once per row on the argument
  values. With ``null_propagating`` (the default) a null argument short-cuts to null, the way
  Spark's ``NullIntolerant`` expressions behave. ``lazy`` specs instead receive ``(node, row, ctx)``
  and evaluate their own children (``coalesce`` short-circuits; lambdas run per element).
- ``kind`` marks the non-scalar functions the DataFrame layer drives: ``"aggregate"`` (with an
  ``aggregate(rows_of_arg_values, node)`` routine), ``"window"`` (ranking functions) and
  ``"generator"`` (``explode``).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import uuid as _uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import cmp_to_key
from typing import Any, Callable, Dict, List, Optional

from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast import coercion
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    Cast,
    Expression,
    FunctionCall,
    LambdaFunction,
    LambdaVariable,
    Literal,
    Star,
)
from sparkleframe.python.column_helpers import (
    _INTEGRAL_RANGE,
    arithmetic,
    cast_value,
    compare_values,
    java_float_string,
    spark_order,
    to_decimal,
    to_spark_string,
    _Overflow,
)
from sparkleframe.python.functions_helpers import (
    JsonNumber,
    compile_java_regex,
    format_datetime,
    get_json_object,
    java_replacement,
    json_text,
    load_json,
    months_between,
    parse_datetime,
    spark_initcap,
    spark_substring,
    spark_trim,
)
from sparkleframe.python.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    Row,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    parse_ddl_schema,
)

_INTEGRAL = (ByteType, ShortType, IntegerType, LongType)
_NUMERIC = (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType)
_ATOMIC = (*_NUMERIC, StringType, BooleanType, DateType, TimestampType, BinaryType, NullType)


@dataclass(frozen=True)
class FunctionSpec:
    resolve: Callable[[FunctionCall, Any], Any]
    evaluate: Optional[Callable[..., Any]] = None
    null_propagating: bool = True
    lazy: bool = False
    kind: str = "scalar"
    aggregate: Optional[Callable[[List[tuple], FunctionCall], Any]] = None


# --------------------------------------------------------------------------- #
# Resolve helpers
# --------------------------------------------------------------------------- #


def cast_child(node: Expression, index: int, target: Any, strict: bool = True) -> None:
    """Replace ``node.children[index]`` with a resolved cast to ``target`` (no-op if already that type)."""
    child = node.children[index]
    if child.data_type == target:
        return
    cast = Cast(child, target, strict=strict)
    cast.data_type = target
    node.children[index] = cast


def _ordinal(index: int) -> str:
    return ["first", "second", "third", "fourth", "fifth"][index] if index < 5 else f"{index + 1}th"


def mismatch(name: str, index: int, required: str, actual: Any) -> TypeError:
    return TypeError(
        f'[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve "{name}" due to data type mismatch: '
        f"The {_ordinal(index)} parameter requires the {required} type, however it has the type "
        f'"{actual.simpleString().upper()}".'
    )


def _arg_type(node: FunctionCall, index: int) -> Any:
    return node.children[index].data_type


def _require_string(node: FunctionCall, index: int) -> None:
    """Spark implicitly casts any atomic argument to string for string functions."""
    dtype = _arg_type(node, index)
    if isinstance(dtype, StringType):
        return
    if isinstance(dtype, _ATOMIC) and not isinstance(dtype, BinaryType):
        cast_child(node, index, StringType())
        return
    raise mismatch(node.name, index, '"STRING"', dtype)


def _require_numeric(node: FunctionCall, index: int, string_to: Any = None) -> None:
    dtype = _arg_type(node, index)
    if isinstance(dtype, _NUMERIC):
        return
    if isinstance(dtype, (StringType, NullType)):
        cast_child(node, index, string_to or DoubleType())
        return
    raise mismatch(node.name, index, '"NUMERIC"', dtype)


def _require_integral(node: FunctionCall, index: int) -> None:
    dtype = _arg_type(node, index)
    if isinstance(dtype, (ByteType, ShortType, IntegerType)):
        cast_child(node, index, IntegerType())
        return
    if isinstance(dtype, (StringType, NullType, LongType)):
        cast_child(node, index, IntegerType())
        return
    raise mismatch(node.name, index, '"INT"', dtype)


def _require_date(node: FunctionCall, index: int) -> None:
    dtype = _arg_type(node, index)
    if isinstance(dtype, DateType):
        return
    if isinstance(dtype, (StringType, TimestampType, NullType)):
        cast_child(node, index, DateType())
        return
    raise mismatch(node.name, index, '"DATE"', dtype)


def _require_timestamp(node: FunctionCall, index: int) -> None:
    dtype = _arg_type(node, index)
    if isinstance(dtype, TimestampType):
        return
    if isinstance(dtype, (StringType, DateType, NullType)):
        cast_child(node, index, TimestampType())
        return
    raise mismatch(node.name, index, '"TIMESTAMP"', dtype)


def _coerce_all_to_wider(node: FunctionCall, name: str) -> Any:
    """Cast every argument to their wider common type (``coalesce``, ``greatest``, ``array``, ...)."""
    types = [c.data_type for c in node.children]
    common = coercion.find_wider_common_type(types)
    if common is None:
        raise TypeError(
            f'[DATATYPE_MISMATCH.DATA_DIFF_TYPES] Cannot resolve "{name}" due to data type mismatch: Input to '
            f"`{name}` should all be the same type, but it's "
            f"[{', '.join(t.simpleString().upper() for t in types)}]."
        )
    for index in range(len(node.children)):
        cast_child(node, index, common)
    return common


def _fixed(dtype: Any) -> Callable[[FunctionCall, Any], Any]:
    return lambda node, rctx: dtype


def _string_unary(node: FunctionCall, rctx: Any) -> Any:
    _require_string(node, 0)
    return StringType()


def _same_as_first(node: FunctionCall, rctx: Any) -> Any:
    return _arg_type(node, 0)


# --------------------------------------------------------------------------- #
# Scalar functions
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, FunctionSpec] = {}


def register(name: str, spec: FunctionSpec) -> None:
    _REGISTRY[name] = spec


def lookup(name: str) -> FunctionSpec:
    spec = _REGISTRY.get(name)
    if spec is None:
        not_implemented_yet(f"functions.{name}")
    return spec


# --- strings ---------------------------------------------------------------


def _resolve_abs(node: FunctionCall, rctx: Any) -> Any:
    _require_numeric(node, 0)
    return _arg_type(node, 0)


def _eval_abs(values: list, node: FunctionCall, ctx: Any) -> Any:
    result = abs(values[0])
    dtype = node.data_type
    if isinstance(dtype, _INTEGRAL) and result > _INTEGRAL_RANGE[type(dtype)][1]:
        raise ArithmeticError(f"[ARITHMETIC_OVERFLOW] {dtype.simpleString()} overflow.")
    return result


register("abs", FunctionSpec(_resolve_abs, _eval_abs))
register("lower", FunctionSpec(_string_unary, lambda v, n, c: v[0].lower()))
register("upper", FunctionSpec(_string_unary, lambda v, n, c: v[0].upper()))
register("trim", FunctionSpec(_string_unary, lambda v, n, c: spark_trim(v[0])))
register("initcap", FunctionSpec(_string_unary, lambda v, n, c: spark_initcap(v[0])))


def _resolve_md5(node: FunctionCall, rctx: Any) -> Any:
    if not isinstance(_arg_type(node, 0), BinaryType):
        if isinstance(_arg_type(node, 0), (StringType, NullType)):
            cast_child(node, 0, BinaryType())
        else:
            raise mismatch("md5", 0, '"BINARY"', _arg_type(node, 0))
    return StringType()


register("md5", FunctionSpec(_resolve_md5, lambda v, n, c: hashlib.md5(bytes(v[0])).hexdigest()))


def _resolve_length(node: FunctionCall, rctx: Any) -> Any:
    if not isinstance(_arg_type(node, 0), BinaryType):
        _require_string(node, 0)
    return IntegerType()


register("length", FunctionSpec(_resolve_length, lambda v, n, c: len(v[0])))


def _resolve_concat(node: FunctionCall, rctx: Any) -> Any:
    types = [c.data_type for c in node.children]
    if (
        types
        and all(isinstance(t, (ArrayType, NullType)) for t in types)
        and any(isinstance(t, ArrayType) for t in types)
    ):
        return _coerce_all_to_wider(node, "concat")
    if types and all(isinstance(t, BinaryType) for t in types):
        return BinaryType()
    for index in range(len(node.children)):
        _require_string(node, index)
    return StringType()


def _eval_concat(values: list, node: FunctionCall, ctx: Any) -> Any:
    if isinstance(node.data_type, ArrayType):
        return [item for value in values for item in value]
    if isinstance(node.data_type, BinaryType):
        return b"".join(bytes(v) for v in values)
    return "".join(values)


register("concat", FunctionSpec(_resolve_concat, _eval_concat))


def _resolve_substring(node: FunctionCall, rctx: Any) -> Any:
    if not isinstance(_arg_type(node, 0), BinaryType):
        _require_string(node, 0)
    _require_integral(node, 1)
    _require_integral(node, 2)
    return _arg_type(node, 0)


register("substring", FunctionSpec(_resolve_substring, lambda v, n, c: spark_substring(v[0], v[1], v[2])))


def _resolve_split(node: FunctionCall, rctx: Any) -> Any:
    _require_string(node, 0)
    compile_java_regex(node.options["pattern"])
    return ArrayType(StringType(), False)


def _eval_split(values: list, node: FunctionCall, ctx: Any) -> Any:
    regex = compile_java_regex(node.options["pattern"])
    limit = node.options.get("limit", -1)
    text = values[0]
    if text == "" or limit == 1:
        return [text]
    # limit <= 0 splits as often as possible and, unlike Java's String.split, keeps trailing empty strings.
    return regex.split(text, maxsplit=limit - 1 if limit > 1 else 0)


register("split", FunctionSpec(_resolve_split, _eval_split))


def _resolve_regexp_replace(node: FunctionCall, rctx: Any) -> Any:
    _require_string(node, 0)
    compile_java_regex(node.options["pattern"])
    return StringType()


def _eval_regexp_replace(values: list, node: FunctionCall, ctx: Any) -> Any:
    regex = compile_java_regex(node.options["pattern"])
    replacement = node.options["replacement"]
    return regex.sub(lambda m: java_replacement(m, replacement), values[0])


register("regexp_replace", FunctionSpec(_resolve_regexp_replace, _eval_regexp_replace))


def _resolve_regexp_extract(node: FunctionCall, rctx: Any) -> Any:
    _require_string(node, 0)
    regex = compile_java_regex(node.options["pattern"])
    if node.options["idx"] > regex.groups:
        raise ValueError(
            f"[INVALID_PARAMETER_VALUE.REGEX_GROUP_INDEX] The value of parameter(s) `idx` in `regexp_extract` is "
            f"invalid: Expects group index between 0 and {regex.groups}, but got {node.options['idx']}."
        )
    return StringType()


def _eval_regexp_extract(values: list, node: FunctionCall, ctx: Any) -> Any:
    match = compile_java_regex(node.options["pattern"]).search(values[0])
    if match is None:
        return ""
    return match.group(node.options["idx"]) or ""


register("regexp_extract", FunctionSpec(_resolve_regexp_extract, _eval_regexp_extract))


def _resolve_rlike(node: FunctionCall, rctx: Any) -> Any:
    _require_string(node, 0)
    compile_java_regex(node.options["pattern"])
    return BooleanType()


register(
    "rlike",
    FunctionSpec(_resolve_rlike, lambda v, n, c: compile_java_regex(n.options["pattern"]).search(v[0]) is not None),
)


def _resolve_string_predicate(node: FunctionCall, rctx: Any) -> Any:
    for index in range(2):
        dtype = _arg_type(node, index)
        if isinstance(dtype, (StringType, NullType)):
            cast_child(node, index, StringType())
        elif isinstance(dtype, BinaryType):
            continue
        else:
            raise mismatch(node.name, index, '"STRING" or "BINARY"', dtype)
    return BooleanType()


register("contains", FunctionSpec(_resolve_string_predicate, lambda v, n, c: v[1] in v[0]))
register("startswith", FunctionSpec(_resolve_string_predicate, lambda v, n, c: v[0].startswith(v[1])))
register("endswith", FunctionSpec(_resolve_string_predicate, lambda v, n, c: v[0].endswith(v[1])))


def _resolve_concat_ws(node: FunctionCall, rctx: Any) -> Any:
    for index in range(len(node.children)):
        if not isinstance(_arg_type(node, index), ArrayType):
            _require_string(node, index)
    return StringType()


def _eval_concat_ws(node: FunctionCall, row: Any, ctx: Any) -> Any:
    separator = ctx.eval(node.children[0], row)
    if separator is None:
        return None
    parts: List[str] = []
    for child in node.children[1:]:
        value = ctx.eval(child, row)
        if isinstance(value, list):
            parts.extend(v for v in value if v is not None)
        elif value is not None:
            parts.append(value)
    return separator.join(parts)


register("concat_ws", FunctionSpec(_resolve_concat_ws, _eval_concat_ws, lazy=True))


# --- conditional / null handling --------------------------------------------


def _resolve_coalesce(node: FunctionCall, rctx: Any) -> Any:
    if not node.children:
        raise TypeError("[WRONG_NUM_ARGS.WITHOUT_SUGGESTION] The `coalesce` requires > 0 parameters but got 0.")
    return _coerce_all_to_wider(node, "coalesce")


def _eval_coalesce(node: FunctionCall, row: Any, ctx: Any) -> Any:
    for child in node.children:
        value = ctx.eval(child, row)
        if value is not None:
            return value
    return None


register("coalesce", FunctionSpec(_resolve_coalesce, _eval_coalesce, lazy=True))


def _resolve_greatest_least(node: FunctionCall, rctx: Any) -> Any:
    if len(node.children) < 2:
        raise TypeError(
            f"[WRONG_NUM_ARGS.WITHOUT_SUGGESTION] The `{node.name}` requires > 1 parameters but got "
            f"{len(node.children)}."
        )
    return _coerce_all_to_wider(node, node.name)


def _eval_greatest_least(values: list, node: FunctionCall, ctx: Any) -> Any:
    present = [v for v in values if v is not None]
    if not present:
        return None
    sign = 1 if node.name == "greatest" else -1
    best = present[0]
    for value in present[1:]:
        if sign * spark_order(value, best, node.data_type) > 0:
            best = value
    return best


register("greatest", FunctionSpec(_resolve_greatest_least, _eval_greatest_least, null_propagating=False))
register("least", FunctionSpec(_resolve_greatest_least, _eval_greatest_least, null_propagating=False))


def _resolve_nullif(node: FunctionCall, rctx: Any) -> Any:
    """``nullif(a, b)`` is ``a`` unless ``a = b``; the comparison coerces like ``==`` does."""
    _, left_cast, right_cast = coercion.coerce_comparison(_arg_type(node, 0), _arg_type(node, 1))
    node.options["comparison"] = (left_cast or _arg_type(node, 0), left_cast, right_cast)
    return _arg_type(node, 0)


def _eval_nullif(node: FunctionCall, row: Any, ctx: Any) -> Any:
    first = ctx.eval(node.children[0], row)
    second = ctx.eval(node.children[1], row)
    if first is None or second is None:
        return first
    common, left_cast, right_cast = node.options["comparison"]
    left = cast_value(first, left_cast, True, node.children[0].data_type) if left_cast else first
    right = cast_value(second, right_cast, True, node.children[1].data_type) if right_cast else second
    return None if compare_values("==", left, right, common) else first


register("nullif", FunctionSpec(_resolve_nullif, _eval_nullif, lazy=True))


def _resolve_isnan(node: FunctionCall, rctx: Any) -> Any:
    dtype = _arg_type(node, 0)
    if not isinstance(dtype, (FloatType, DoubleType)):
        if isinstance(dtype, (_NUMERIC + (StringType, NullType))):
            cast_child(node, 0, DoubleType())
        else:
            raise mismatch("isnan", 0, '"DOUBLE" or "FLOAT"', dtype)
    return BooleanType()


register(
    "isnan",
    FunctionSpec(_resolve_isnan, lambda v, n, c: v[0] is not None and math.isnan(v[0]), null_propagating=False),
)


# --- math --------------------------------------------------------------------


def _resolve_floor(node: FunctionCall, rctx: Any) -> Any:
    _require_numeric(node, 0)
    dtype = _arg_type(node, 0)
    if isinstance(dtype, DecimalType):
        return DecimalType(min(dtype.precision - dtype.scale + 1, 38), 0)
    if isinstance(dtype, (FloatType, DoubleType)):
        cast_child(node, 0, DoubleType())
    return LongType()


def _eval_floor(values: list, node: FunctionCall, ctx: Any) -> Any:
    value = values[0]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        # Spark: floor(NaN) / floor(±inf) overflow to the long range ends (NaN -> 0).
        return (
            0 if math.isnan(value) else (_INTEGRAL_RANGE[LongType][1] if value > 0 else _INTEGRAL_RANGE[LongType][0])
        )
    result = math.floor(value)
    return Decimal(result) if isinstance(node.data_type, DecimalType) else int(result)


register("floor", FunctionSpec(_resolve_floor, _eval_floor))


def _resolve_round(node: FunctionCall, rctx: Any) -> Any:
    _require_numeric(node, 0)
    dtype = _arg_type(node, 0)
    scale = node.options["scale"]
    if isinstance(dtype, DecimalType):
        integral_digits = dtype.precision - dtype.scale + 1
        if scale < 0:
            return DecimalType(min(max(integral_digits, -scale + 1), 38), 0)
        new_scale = min(dtype.scale, scale)
        return DecimalType(min(integral_digits + new_scale, 38), new_scale)
    return dtype


def _eval_round(values: list, node: FunctionCall, ctx: Any) -> Any:
    value, scale = values[0], node.options["scale"]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value
    exact = Decimal(repr(value)) if isinstance(value, float) else Decimal(value)
    rounded = exact.quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP)
    if isinstance(node.data_type, DecimalType):
        return to_decimal(rounded, node.data_type)
    if isinstance(value, float):
        return float(rounded) + 0.0  # Spark rounds through BigDecimal, which has no negative zero
    return int(rounded)


register("round", FunctionSpec(_resolve_round, _eval_round))


def _resolve_try_divide(node: FunctionCall, rctx: Any) -> Any:
    result, left_cast, right_cast = coercion.coerce_arithmetic("/", _arg_type(node, 0), _arg_type(node, 1))
    if left_cast is not None:
        cast_child(node, 0, left_cast)
    if right_cast is not None:
        cast_child(node, 1, right_cast)
    return result


register(
    "try_divide",
    FunctionSpec(_resolve_try_divide, lambda v, n, c: arithmetic("/", v[0], v[1], n.data_type, strict=False)),
)


# --- dates -------------------------------------------------------------------


register("current_date", FunctionSpec(_fixed(DateType()), lambda v, n, c: c.now.date(), null_propagating=False))
register("current_timestamp", FunctionSpec(_fixed(TimestampType()), lambda v, n, c: c.now, null_propagating=False))


def _resolve_date_add_sub(node: FunctionCall, rctx: Any) -> Any:
    _require_date(node, 0)
    _require_integral(node, 1)
    return DateType()


register("date_sub", FunctionSpec(_resolve_date_add_sub, lambda v, n, c: v[0] - timedelta(days=v[1])))
register("date_add", FunctionSpec(_resolve_date_add_sub, lambda v, n, c: v[0] + timedelta(days=v[1])))


def _resolve_datediff(node: FunctionCall, rctx: Any) -> Any:
    _require_date(node, 0)
    _require_date(node, 1)
    return IntegerType()


register("datediff", FunctionSpec(_resolve_datediff, lambda v, n, c: (v[0] - v[1]).days))


def _add_months(day: date, months: int) -> date:
    total = day.year * 12 + day.month - 1 + months
    year, month = divmod(total, 12)
    month += 1
    last_day = (date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(day.day, last_day))


register("add_months", FunctionSpec(_resolve_date_add_sub, lambda v, n, c: _add_months(v[0], v[1])))


def _resolve_months_between(node: FunctionCall, rctx: Any) -> Any:
    _require_timestamp(node, 0)
    _require_timestamp(node, 1)
    return DoubleType()


register(
    "months_between",
    FunctionSpec(_resolve_months_between, lambda v, n, c: months_between(v[0], v[1], n.options.get("roundOff", True))),
)


def _resolve_date_format(node: FunctionCall, rctx: Any) -> Any:
    _require_timestamp(node, 0)
    format_datetime(datetime(2000, 1, 1), node.options["format"])  # validate the pattern at analysis
    return StringType()


register("date_format", FunctionSpec(_resolve_date_format, lambda v, n, c: format_datetime(v[0], n.options["format"])))


def _resolve_trunc(node: FunctionCall, rctx: Any) -> Any:
    _require_date(node, 0)
    return DateType()


def _eval_trunc(values: list, node: FunctionCall, ctx: Any) -> Any:
    day, unit = values[0], node.options["format"].lower()
    if unit in ("year", "yyyy", "yy"):
        return date(day.year, 1, 1)
    if unit in ("quarter",):
        return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
    if unit in ("month", "mm", "mon"):
        return date(day.year, day.month, 1)
    if unit == "week":
        return day - timedelta(days=day.weekday())
    return None


register("trunc", FunctionSpec(_resolve_trunc, _eval_trunc))


def _make_parse(target: Any, strict: bool) -> FunctionSpec:
    """``to_date`` / ``try_to_date`` / ``to_timestamp`` / ``try_to_timestamp`` — one helper, a ``strict`` flag."""

    def resolve(node: FunctionCall, rctx: Any) -> Any:
        dtype = _arg_type(node, 0)
        if node.options.get("format") is None or not isinstance(dtype, StringType):
            if isinstance(dtype, (StringType, DateType, TimestampType, NullType)):
                return target
            raise mismatch(node.name, 0, '"STRING" or "DATE" or "TIMESTAMP"', dtype)
        return target

    def evaluate(values: list, node: FunctionCall, ctx: Any) -> Any:
        value, pattern = values[0], node.options.get("format")
        if not isinstance(value, str) or pattern is None:
            return cast_value(value, target, strict, node.children[0].data_type)
        parsed = parse_datetime(value, pattern)
        if parsed is None:
            if not strict:
                return None
            raise ValueError(
                f"[CANNOT_PARSE_TIMESTAMP] Text '{value}' could not be parsed with pattern '{pattern}'. "
                f"Use `try_to_timestamp` to tolerate invalid input string and return NULL instead."
            )
        return parsed.date() if isinstance(target, DateType) else parsed

    return FunctionSpec(resolve, evaluate)


register("to_date", _make_parse(DateType(), strict=True))
register("try_to_date", _make_parse(DateType(), strict=False))
register("to_timestamp", _make_parse(TimestampType(), strict=True))
register("try_to_timestamp", _make_parse(TimestampType(), strict=False))


# --- complex types -----------------------------------------------------------


def field_name(expr: Expression, index: int) -> str:
    """The struct field name Spark derives for a ``struct(...)`` argument."""
    if isinstance(expr, Alias):
        return expr.name
    if isinstance(expr, AttributeReference):
        return expr.name.split(".")[-1]
    if isinstance(expr, FunctionCall) and expr.name == "get_field":
        return expr.options["name"]
    return f"col{index + 1}"


def _resolve_struct(node: FunctionCall, rctx: Any) -> Any:
    return StructType([StructField(field_name(c, i), c.data_type) for i, c in enumerate(node.children)])


register(
    "struct",
    FunctionSpec(
        _resolve_struct,
        lambda v, n, c: Row.from_fields(n.data_type.names, list(v)),
        null_propagating=False,
    ),
)


def _resolve_array(node: FunctionCall, rctx: Any) -> Any:
    if not node.children:
        return ArrayType(StringType(), False)
    element = _coerce_all_to_wider(node, "array")
    return ArrayType(element, True)


register("array", FunctionSpec(_resolve_array, lambda v, n, c: list(v), null_propagating=False))


def _resolve_create_map(node: FunctionCall, rctx: Any) -> Any:
    if len(node.children) % 2:
        raise TypeError(
            f"[WRONG_NUM_ARGS.WITHOUT_SUGGESTION] The `map` requires 2n (n > 0) parameters but the actual "
            f"number is {len(node.children)}."
        )
    if not node.children:
        return MapType(StringType(), StringType(), False)
    keys = coercion.find_wider_common_type([c.data_type for c in node.children[0::2]])
    values = coercion.find_wider_common_type([c.data_type for c in node.children[1::2]])
    if keys is None or values is None:
        raise TypeError(
            "[DATATYPE_MISMATCH.CREATE_MAP_KEY_DIFF_TYPES] The given keys of function `map` should all be the same type."
        )
    for index in range(0, len(node.children), 2):
        cast_child(node, index, keys)
        cast_child(node, index + 1, values)
    return MapType(keys, values, True)


def build_map(pairs: List[tuple]) -> dict:
    """A Spark map from ``(key, value)`` pairs: null keys and duplicate keys raise (Spark 4 defaults)."""
    result: dict = {}
    for key, value in pairs:
        if key is None:
            raise ValueError("[NULL_MAP_KEY] Cannot use null as map key.")
        hashable = _hashable(key)
        if hashable in result:
            raise ValueError(
                f"[DUPLICATED_MAP_KEY] Duplicate map key {key} was found, please check the input data. If you want "
                f'to remove the duplicated keys, you can set "spark.sql.mapKeyDedupPolicy" to "LAST_WIN".'
            )
        result[hashable] = value
    return result


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _hashable(v)) for k, v in value.items())
    return value


register(
    "create_map",
    FunctionSpec(
        _resolve_create_map,
        lambda v, n, c: build_map(list(zip(v[0::2], v[1::2]))),
        null_propagating=False,
    ),
)


def _resolve_map_from_entries(node: FunctionCall, rctx: Any) -> Any:
    dtype = _arg_type(node, 0)
    if not (
        isinstance(dtype, ArrayType)
        and isinstance(dtype.elementType, StructType)
        and len(dtype.elementType.fields) == 2
    ):
        raise mismatch("map_from_entries", 0, '"ARRAY" of pair "STRUCT"', dtype)
    key_field, value_field = dtype.elementType.fields
    return MapType(key_field.dataType, value_field.dataType, True)


def _eval_map_from_entries(values: list, node: FunctionCall, ctx: Any) -> Any:
    if any(entry is None for entry in values[0]):
        return None
    return build_map([(entry[0], entry[1]) for entry in values[0]])


register("map_from_entries", FunctionSpec(_resolve_map_from_entries, _eval_map_from_entries))


def _require_map(node: FunctionCall) -> MapType:
    dtype = _arg_type(node, 0)
    if not isinstance(dtype, MapType):
        raise mismatch(node.name, 0, '"MAP"', dtype)
    return dtype


register(
    "map_keys",
    FunctionSpec(lambda n, r: ArrayType(_require_map(n).keyType, False), lambda v, n, c: list(v[0].keys())),
)
register(
    "map_values",
    FunctionSpec(
        lambda n, r: ArrayType(_require_map(n).valueType, _require_map(n).valueContainsNull),
        lambda v, n, c: list(v[0].values()),
    ),
)


def _resolve_map_entries(node: FunctionCall, rctx: Any) -> Any:
    dtype = _require_map(node)
    return ArrayType(
        StructType([StructField("key", dtype.keyType, False), StructField("value", dtype.valueType, True)]), False
    )


register(
    "map_entries",
    FunctionSpec(
        _resolve_map_entries, lambda v, n, c: [Row.from_fields(["key", "value"], [k, val]) for k, val in v[0].items()]
    ),
)


def _resolve_get_field(node: FunctionCall, rctx: Any) -> Any:
    dtype = _arg_type(node, 0)
    name = node.options["name"]
    if not isinstance(dtype, StructType):
        raise TypeError(
            f'[INVALID_EXTRACT_BASE_FIELD_TYPE] Can\'t extract a value from "{node.children[0]!r}". Need a complex '
            f'type [STRUCT, ARRAY, MAP] but got "{dtype.simpleString().upper()}".'
        )
    index = _struct_field_index(dtype, name)
    node.options["index"] = index
    return dtype.fields[index].dataType


def _struct_field_index(dtype: StructType, name: str) -> int:
    for index, field in enumerate(dtype.fields):
        if field.name == name:
            return index
    matches = [i for i, f in enumerate(dtype.fields) if f.name.lower() == name.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"[FIELD_NOT_FOUND] No such struct field `{name}` in {', '.join(f'`{f.name}`' for f in dtype.fields)}."
    )


register("get_field", FunctionSpec(_resolve_get_field, lambda v, n, c: v[0][n.options["index"]]))


def _resolve_get_item(node: FunctionCall, rctx: Any) -> Any:
    """``col[key]`` / ``getItem``: struct field by name, map value by key, or array element by 0-based index."""
    dtype = _arg_type(node, 0)
    if isinstance(dtype, StructType):
        key = node.children[1]
        if not isinstance(key, Literal) or not isinstance(key.value, str):
            raise TypeError("[INVALID_EXTRACT_FIELD_TYPE] Field name should be a non-null string literal.")
        index = _struct_field_index(dtype, key.value)
        node.options["struct_index"] = index
        return dtype.fields[index].dataType
    if isinstance(dtype, MapType):
        cast_child(node, 1, dtype.keyType)
        return dtype.valueType
    if isinstance(dtype, ArrayType):
        if not isinstance(_arg_type(node, 1), _INTEGRAL + (NullType,)):
            raise mismatch("getItem", 1, '"INT"', _arg_type(node, 1))
        cast_child(node, 1, IntegerType())
        return dtype.elementType
    if isinstance(dtype, NullType):
        return NullType()
    raise TypeError(
        f'[INVALID_EXTRACT_BASE_FIELD_TYPE] Can\'t extract a value from "{node.children[0]!r}". Need a complex '
        f'type [STRUCT, ARRAY, MAP] but got "{dtype.simpleString().upper()}".'
    )


def _eval_get_item(values: list, node: FunctionCall, ctx: Any) -> Any:
    container, key = values
    if "struct_index" in node.options:
        return container[node.options["struct_index"]]
    if isinstance(container, dict):
        return container.get(_hashable(key))
    if not 0 <= key < len(container):
        raise IndexError(
            f"[INVALID_ARRAY_INDEX] The index {key} is out of bounds. The array has {len(container)} elements. "
            f"Use the SQL function `get()` to tolerate accessing element at invalid index and return NULL instead."
        )
    return container[key]


register("get_item", FunctionSpec(_resolve_get_item, _eval_get_item))


def _make_element_at(strict: bool) -> FunctionSpec:
    def resolve(node: FunctionCall, rctx: Any) -> Any:
        dtype = _arg_type(node, 0)
        if isinstance(dtype, MapType):
            cast_child(node, 1, dtype.keyType)
            return dtype.valueType
        if isinstance(dtype, ArrayType):
            if not isinstance(_arg_type(node, 1), _INTEGRAL + (NullType,)):
                raise mismatch(node.name, 1, '"INT"', _arg_type(node, 1))
            cast_child(node, 1, IntegerType())
            return dtype.elementType
        raise mismatch(node.name, 0, '"ARRAY" or "MAP"', dtype)

    def evaluate(values: list, node: FunctionCall, ctx: Any) -> Any:
        container, key = values
        if isinstance(container, dict):
            return container.get(_hashable(key))
        if key == 0:
            raise IndexError("[INVALID_INDEX_OF_ZERO] The index 0 is invalid. An index shall be either < 0 or > 0.")
        if abs(key) > len(container):
            if not strict:
                return None
            raise IndexError(
                f"[INVALID_ARRAY_INDEX_IN_ELEMENT_AT] The index {key} is out of bounds. The array has "
                f"{len(container)} elements. Use `try_element_at` to tolerate accessing element at invalid index "
                f"and return NULL instead."
            )
        return container[key - 1 if key > 0 else key]

    return FunctionSpec(resolve, evaluate)


register("element_at", _make_element_at(strict=True))
register("try_element_at", _make_element_at(strict=False))


def _require_array(node: FunctionCall, index: int = 0) -> ArrayType:
    dtype = _arg_type(node, index)
    if not isinstance(dtype, ArrayType):
        raise mismatch(node.name, index, '"ARRAY"', dtype)
    return dtype


def _resolve_array_contains(node: FunctionCall, rctx: Any) -> Any:
    dtype = _require_array(node)
    common = coercion.find_wider_type(dtype.elementType, _arg_type(node, 1))
    if common is None:
        raise TypeError(
            f'[DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES] Cannot resolve "array_contains" due to data type '
            f'mismatch: Input to `array_contains` should have been "ARRAY" followed by a value with same element '
            f'type, but it\'s ["{dtype.simpleString().upper()}", "{_arg_type(node, 1).simpleString().upper()}"].'
        )
    cast_child(node, 0, ArrayType(common, dtype.containsNull))
    cast_child(node, 1, common)
    return BooleanType()


def _eval_array_contains(values: list, node: FunctionCall, ctx: Any) -> Any:
    array, needle = values
    element_type = node.children[0].data_type.elementType
    saw_null = False
    for item in array:
        if item is None:
            saw_null = True
        elif compare_values("==", item, needle, element_type):
            return True
    return None if saw_null else False


register("array_contains", FunctionSpec(_resolve_array_contains, _eval_array_contains))


def _resolve_size(node: FunctionCall, rctx: Any) -> Any:
    if not isinstance(_arg_type(node, 0), (ArrayType, MapType)):
        raise mismatch("size", 0, '"ARRAY" or "MAP"', _arg_type(node, 0))
    return IntegerType()


register("size", FunctionSpec(_resolve_size, lambda v, n, c: len(v[0])))


def _sort_key(element_type: Any) -> Callable[[Any], Any]:
    """Ascending Spark order with nulls first (``sort_array`` reverses it for descending, nulls last)."""

    def compare(left: Any, right: Any) -> int:
        if left is None or right is None:
            return (left is not None) - (right is not None)
        return spark_order(left, right, element_type)

    return cmp_to_key(compare)


def _eval_sort_array(values: list, node: FunctionCall, ctx: Any) -> Any:
    element_type = node.data_type.elementType
    ascending = node.options.get("asc", True)
    ordered = sorted(values[0], key=_sort_key(element_type))
    return ordered if ascending else ordered[::-1]


register("sort_array", FunctionSpec(lambda n, r: _require_array(n), _eval_sort_array))


def _eval_array_max(values: list, node: FunctionCall, ctx: Any) -> Any:
    present = [v for v in values[0] if v is not None]
    if not present:
        return None
    best = present[0]
    for value in present[1:]:
        if spark_order(value, best, node.data_type) > 0:
            best = value
    return best


register("array_max", FunctionSpec(lambda n, r: _require_array(n).elementType, _eval_array_max))


def _resolve_explode(node: FunctionCall, rctx: Any) -> Any:
    dtype = _arg_type(node, 0)
    if isinstance(dtype, ArrayType):
        return StructType([StructField("col", dtype.elementType, dtype.containsNull)])
    if isinstance(dtype, MapType):
        return StructType([StructField("key", dtype.keyType, False), StructField("value", dtype.valueType, True)])
    raise mismatch("explode", 0, '"ARRAY" or "MAP"', dtype)


def explode_rows(value: Any) -> List[tuple]:
    """The rows ``explode`` generates for one input value (none for null / empty)."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [(k, v) for k, v in value.items()]
    return [(v,) for v in value]


register("explode", FunctionSpec(_resolve_explode, kind="generator"))


# --- higher-order functions ---------------------------------------------------


def _resolve_higher_order(node: FunctionCall, rctx: Any) -> Any:
    array_type = _require_array(node)
    func = node.children[1]
    assert isinstance(func, LambdaFunction)
    arity = func.func.__code__.co_argcount if hasattr(func.func, "__code__") else 1
    params = [LambdaVariable("x"), LambdaVariable("i")][:arity]
    params[0].data_type = array_type.elementType
    if arity > 1:
        params[1].data_type = IntegerType()
    body = rctx.analyze_lambda(func, params)
    node.children[1] = body
    node.options["slots"] = [p.slot for p in params]
    if node.name == "filter":
        if not isinstance(body.data_type, (BooleanType, NullType)):
            raise mismatch("filter", 1, '"BOOLEAN"', body.data_type)
        return array_type
    return ArrayType(body.data_type, True)


def _eval_higher_order(node: FunctionCall, row: Any, ctx: Any) -> Any:
    array = ctx.eval(node.children[0], row)
    if array is None:
        return None
    body, slots = node.children[1], node.options["slots"]
    out = []
    for index, element in enumerate(array):
        ctx.lambda_values[slots[0]] = element
        if len(slots) > 1:
            ctx.lambda_values[slots[1]] = index
        result = ctx.eval(body, row)
        if node.name == "filter":
            if result is True:
                out.append(element)
        else:
            out.append(result)
    return out


register("transform", FunctionSpec(_resolve_higher_order, _eval_higher_order, lazy=True))
register("filter", FunctionSpec(_resolve_higher_order, _eval_higher_order, lazy=True))


# --- membership --------------------------------------------------------------


def _resolve_in(node: FunctionCall, rctx: Any) -> Any:
    types = [c.data_type for c in node.children]
    common = coercion.find_wider_common_type(types)
    if common is None:
        raise TypeError(
            f'[DATATYPE_MISMATCH.DATA_DIFF_TYPES] Cannot resolve "in" due to data type mismatch: Input to `in` '
            f"should all be the same type, but it's [{', '.join(t.simpleString().upper() for t in types)}]."
        )
    for index in range(len(node.children)):
        cast_child(node, index, common)
    return BooleanType()


def _eval_in(node: FunctionCall, row: Any, ctx: Any) -> Any:
    value = ctx.eval(node.children[0], row)
    if len(node.children) == 1:
        return False  # an empty IN list is false whatever the probe
    if value is None:
        return None
    common = node.children[0].data_type
    saw_null = False
    for child in node.children[1:]:
        candidate = ctx.eval(child, row)
        if candidate is None:
            saw_null = True
        elif compare_values("==", value, candidate, common):
            return True
    return None if saw_null else False


register("in", FunctionSpec(_resolve_in, _eval_in, lazy=True))


# --- JSON --------------------------------------------------------------------


def _resolve_get_json_object(node: FunctionCall, rctx: Any) -> Any:
    _require_string(node, 0)
    return StringType()


register(
    "get_json_object",
    FunctionSpec(_resolve_get_json_object, lambda v, n, c: get_json_object(v[0], n.options["path"])),
)


def _json_schema(node: FunctionCall) -> Any:
    schema = node.options["schema"]
    return parse_ddl_schema(schema) if isinstance(schema, str) else schema  # Spark's ``DataType.fromDDL``


def _resolve_from_json(node: FunctionCall, rctx: Any) -> Any:
    _require_string(node, 0)
    schema = _json_schema(node)
    if not isinstance(schema, (StructType, MapType, ArrayType)):
        raise TypeError(
            f"[DATATYPE_MISMATCH.INVALID_JSON_SCHEMA] Input schema {schema.simpleString()} must be a struct, an "
            f"array or a map."
        )
    if _json_parse_mode(node) not in ("PERMISSIVE", "FAILFAST"):
        raise ValueError(
            f"[PARSE_MODE_UNSUPPORTED] The function `from_json` doesn't support the {_json_parse_mode(node)} mode. "
            f"Acceptable modes are PERMISSIVE and FAILFAST."
        )
    return schema


def _json_parse_mode(node: FunctionCall) -> str:
    return str((node.options.get("options") or {}).get("mode", "PERMISSIVE")).upper()


class _JsonConversionError(Exception):
    pass


def _json_to_value(node: Any, dtype: Any) -> Any:
    if node is None:
        return None
    if isinstance(dtype, StringType):
        return node if isinstance(node, str) else json_text(node)
    if isinstance(dtype, BooleanType):
        if isinstance(node, bool):
            return node
        raise _JsonConversionError
    if isinstance(dtype, _INTEGRAL):
        if (
            isinstance(node, JsonNumber)
            and node == node.to_integral_value()
            and "." not in node.text
            and "e" not in (node.text.lower())
        ):
            value = int(node)
            low, high = _INTEGRAL_RANGE[type(dtype)]
            if low <= value <= high:
                return value
        raise _JsonConversionError
    if isinstance(dtype, (FloatType, DoubleType)):
        if isinstance(node, JsonNumber):
            return float(node)
        if node in ("NaN", "Infinity", "-Infinity"):
            return float(node.replace("Infinity", "inf"))
        raise _JsonConversionError
    if isinstance(dtype, DecimalType):
        if isinstance(node, JsonNumber):
            try:
                return to_decimal(Decimal(node), dtype)
            except _Overflow:
                raise _JsonConversionError from None
        raise _JsonConversionError
    if isinstance(dtype, (DateType, TimestampType)):
        if isinstance(node, str):
            result = cast_value(node, dtype, False)
            if result is None:
                parsed = parse_datetime(node, "yyyy-MM-dd'T'HH:mm:ss.SSSXXX") or parse_datetime(
                    node, "yyyy-MM-dd'T'HH:mm:ssXXX"
                )
                if parsed is None:
                    raise _JsonConversionError
                result = parsed.date() if isinstance(dtype, DateType) else parsed
            return result
        raise _JsonConversionError
    if isinstance(dtype, ArrayType):
        if not isinstance(node, list):
            raise _JsonConversionError
        return [_json_field(v, dtype.elementType) for v in node]
    if isinstance(dtype, MapType):
        if not isinstance(node, dict):
            raise _JsonConversionError
        return {k: _json_field(v, dtype.valueType) for k, v in node.items()}
    if isinstance(dtype, StructType):
        if not isinstance(node, dict):
            raise _JsonConversionError
        return Row.from_fields(dtype.names, [_json_field(node.get(f.name), f.dataType) for f in dtype.fields])
    not_implemented_yet(f"from_json into {dtype.simpleString()}")


def _json_field(node: Any, dtype: Any) -> Any:
    """Convert a nested value; a field that does not fit its type becomes null (Spark's partial results)."""
    try:
        return _json_to_value(node, dtype)
    except _JsonConversionError:
        return None


def _eval_from_json(values: list, node: FunctionCall, ctx: Any) -> Any:
    schema = node.data_type
    if not values[0].strip():
        return None  # no JSON token at all: null, not a struct of nulls
    try:
        parsed = load_json(values[0])
        return _json_to_value(parsed, schema)
    except (ValueError, _JsonConversionError):
        if _json_parse_mode(node) == "FAILFAST":
            raise ValueError(
                f"[MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION] Malformed records are detected in record "
                f"parsing: [{values[0]}]. Parse Mode: FAILFAST."
            ) from None
        if isinstance(schema, StructType):
            return Row.from_fields(schema.names, [None] * len(schema.fields))
        return None


register("from_json", FunctionSpec(_resolve_from_json, _eval_from_json))


def _resolve_to_json(node: FunctionCall, rctx: Any) -> Any:
    dtype = _arg_type(node, 0)
    if not isinstance(dtype, (StructType, MapType, ArrayType)):
        raise mismatch("to_json", 0, '"STRUCT" or "MAP" or "ARRAY"', dtype)
    return StringType()


def _option_flag(options: Dict[str, Any], name: str, default: bool) -> bool:
    for key, value in (options or {}).items():
        if key.lower() == name.lower():
            return value if isinstance(value, bool) else str(value).lower() == "true"
    return default


def _option_value(options: Dict[str, Any], name: str, default: str) -> str:
    for key, value in (options or {}).items():
        if key.lower() == name.lower():
            return str(value)
    return default


def to_json_text(value: Any, dtype: Any, options: Dict[str, Any]) -> str:
    """Write ``value`` the way Spark's ``JacksonGenerator`` does (``to_json``)."""
    if value is None:
        return "null"
    if isinstance(dtype, StructType):
        ignore_nulls = _option_flag(options, "ignoreNullFields", True)
        parts = [
            f"{_json_string(field.name)}:{to_json_text(item, field.dataType, options)}"
            for field, item in zip(dtype.fields, value)
            if not (item is None and ignore_nulls)
        ]
        return "{" + ",".join(parts) + "}"
    if isinstance(dtype, MapType):
        parts = [
            f"{_json_string(to_spark_string(k, dtype.keyType))}:{to_json_text(v, dtype.valueType, options)}"
            for k, v in value.items()
        ]
        return "{" + ",".join(parts) + "}"
    if isinstance(dtype, ArrayType):
        return "[" + ",".join(to_json_text(v, dtype.elementType, options) for v in value) + "]"
    if isinstance(dtype, StringType):
        return _json_string(value)
    if isinstance(dtype, BooleanType):
        return "true" if value else "false"
    if isinstance(dtype, _INTEGRAL):
        return str(value)
    if isinstance(dtype, (FloatType, DoubleType)):
        if math.isnan(value) or math.isinf(value):
            return _json_string(java_float_string(value))
        return java_float_string(value, single=isinstance(dtype, FloatType))
    if isinstance(dtype, DecimalType):
        return str(value)
    if isinstance(dtype, TimestampType):
        pattern = _option_value(options, "timestampFormat", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX")
        return _json_string(format_datetime(value, pattern))
    if isinstance(dtype, DateType):
        pattern = _option_value(options, "dateFormat", "yyyy-MM-dd")
        return _json_string(format_datetime(datetime.combine(value, datetime.min.time()), pattern))
    if isinstance(dtype, BinaryType):
        return _json_string(base64.b64encode(bytes(value)).decode("ascii"))
    not_implemented_yet(f"to_json of {dtype.simpleString()}")


def _json_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


register(
    "to_json",
    FunctionSpec(
        _resolve_to_json,
        lambda v, n, c: to_json_text(v[0], n.children[0].data_type, n.options.get("options") or {}),
    ),
)


# --- non-deterministic ---------------------------------------------------------


def _eval_rand(values: list, node: FunctionCall, ctx: Any) -> Any:
    generator = ctx.random_state.get(id(node))
    if generator is None:
        seed = node.options.get("seed")
        generator = random.Random(seed)
        ctx.random_state[id(node)] = generator
    return generator.random()


register("rand", FunctionSpec(_fixed(DoubleType()), _eval_rand, null_propagating=False))
register("uuid", FunctionSpec(_fixed(StringType()), lambda v, n, c: str(_uuid.uuid4()), null_propagating=False))
register(
    "monotonically_increasing_id",
    FunctionSpec(_fixed(LongType()), lambda v, n, c: c.row_index, null_propagating=False),
)


# --- aggregates ----------------------------------------------------------------


def _resolve_sum(node: FunctionCall, rctx: Any) -> Any:
    _require_numeric(node, 0)
    dtype = _arg_type(node, 0)
    if isinstance(dtype, _INTEGRAL):
        return LongType()
    if isinstance(dtype, DecimalType):
        return DecimalType(min(dtype.precision + 10, 38), dtype.scale)
    return DoubleType()


def _agg_sum(rows: List[tuple], node: FunctionCall) -> Any:
    values = [r[0] for r in rows if r[0] is not None]
    if not values:
        return None
    result_type = node.data_type
    if isinstance(result_type, DecimalType):
        total = sum((Decimal(v) for v in values), Decimal(0))
        return arithmetic("+", total, Decimal(0), result_type)
    total = sum(values)
    if isinstance(result_type, LongType):
        low, high = _INTEGRAL_RANGE[LongType]
        if not low <= total <= high:
            raise ArithmeticError("[ARITHMETIC_OVERFLOW] long overflow. Use 'try_sum' to tolerate overflow.")
    return total


register("sum", FunctionSpec(_resolve_sum, kind="aggregate", aggregate=_agg_sum))


def _resolve_avg(node: FunctionCall, rctx: Any) -> Any:
    _require_numeric(node, 0)
    dtype = _arg_type(node, 0)
    if isinstance(dtype, DecimalType):
        return DecimalType(min(dtype.precision + 4, 38), min(dtype.scale + 4, 38))
    return DoubleType()


def _agg_avg(rows: List[tuple], node: FunctionCall) -> Any:
    values = [r[0] for r in rows if r[0] is not None]
    if not values:
        return None
    if isinstance(node.data_type, DecimalType):
        return arithmetic("/", sum((Decimal(v) for v in values), Decimal(0)), Decimal(len(values)), node.data_type)
    return sum(float(v) for v in values) / len(values)


register("avg", FunctionSpec(_resolve_avg, kind="aggregate", aggregate=_agg_avg))


def _resolve_min_max(node: FunctionCall, rctx: Any) -> Any:
    dtype = _arg_type(node, 0)
    if isinstance(dtype, MapType):
        raise mismatch(node.name, 0, "orderable", dtype)
    return dtype


def _agg_min_max(rows: List[tuple], node: FunctionCall) -> Any:
    values = [r[0] for r in rows if r[0] is not None]
    if not values:
        return None
    sign = 1 if node.name == "max" else -1
    best = values[0]
    for value in values[1:]:
        if sign * spark_order(value, best, node.data_type) > 0:
            best = value
    return best


register("max", FunctionSpec(_resolve_min_max, kind="aggregate", aggregate=_agg_min_max))
register("min", FunctionSpec(_resolve_min_max, kind="aggregate", aggregate=_agg_min_max))


def _resolve_count(node: FunctionCall, rctx: Any) -> Any:
    node.children[:] = [_one() if isinstance(c, Star) else c for c in node.children]
    return LongType()


def _one() -> Literal:
    literal = Literal(1)
    literal.data_type = IntegerType()
    return literal


register(
    "count",
    FunctionSpec(
        _resolve_count,
        kind="aggregate",
        aggregate=lambda rows, node: sum(1 for r in rows if all(v is not None for v in r)),
    ),
)


def _agg_first(rows: List[tuple], node: FunctionCall) -> Any:
    for row in rows:
        if row[0] is not None or not node.options.get("ignorenulls"):
            return row[0]
    return None


register("first", FunctionSpec(_same_as_first, kind="aggregate", aggregate=_agg_first))


def _resolve_collect(node: FunctionCall, rctx: Any) -> Any:
    return ArrayType(_arg_type(node, 0), False)


def _agg_collect_set(rows: List[tuple], node: FunctionCall) -> Any:
    seen = set()
    out = []
    for (value,) in rows:
        if value is None:
            continue
        key = _hashable(value)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


register(
    "collect_list",
    FunctionSpec(
        _resolve_collect, kind="aggregate", aggregate=lambda rows, node: [r[0] for r in rows if r[0] is not None]
    ),
)
register("collect_set", FunctionSpec(_resolve_collect, kind="aggregate", aggregate=_agg_collect_set))


# --- window ranking functions ------------------------------------------------------

register("row_number", FunctionSpec(_fixed(IntegerType()), kind="window"))
register("rank", FunctionSpec(_fixed(IntegerType()), kind="window"))
register("dense_rank", FunctionSpec(_fixed(IntegerType()), kind="window"))
