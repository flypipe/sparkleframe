"""Pure-Python lazy Column expression tree with Spark-compatible null semantics."""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional, Union

from sparkleframe.pythondf.types import (
    BooleanType,
    DataType,
    spark_type_name_to_type,
)


Data = dict  # dict[str, list[Any]]
EvalFn = Callable[[Data], list]


def _row_count(data: Data) -> int:
    if not data:
        return 0
    return len(next(iter(data.values())))


def _null_safe_binop(op: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def f(a: Any, b: Any) -> Any:
        if a is None or b is None:
            return None
        return op(a, b)
    return f


def _kleene_and(a: Any, b: Any) -> Any:
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return bool(a) and bool(b)


def _kleene_or(a: Any, b: Any) -> Any:
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return bool(a) or bool(b)


def _kleene_not(a: Any) -> Any:
    if a is None:
        return None
    return not a


class Column:
    """Lazy column expression. Evaluates against a dict[str, list] on demand."""

    def __init__(self, eval_fn: EvalFn, name: Optional[str] = None):
        self._eval = eval_fn
        self._name = name
        # Sort metadata for orderBy — populated by .asc()/.desc()
        self._sort_key: Optional[EvalFn] = None
        self._sort_descending: bool = False
        self._sort_nulls_last: bool = False
        # Aggregate metadata (populated by functions.sum/count/mean/...).
        # When set, this Column reduces a list-of-values to a scalar.
        # ``_agg_source_name`` is "*" for COUNT(*), a column name like "a",
        # or None when the source is an arbitrary expression (then
        # ``_agg_source_eval`` is used to produce the per-row values to reduce).
        self._agg_fn: Optional[Callable[[list], Any]] = None
        self._agg_source_name: Optional[str] = None
        self._agg_source_eval: Optional[EvalFn] = None

    def eval(self, data: Data) -> list:
        return self._eval(data)

    @property
    def name(self) -> Optional[str]:
        return self._name

    def alias(self, name: str) -> "Column":
        out = Column(self._eval, name=name)
        # Preserve aggregate metadata across alias (needed for groupBy.agg + standalone agg).
        if self._agg_fn is not None:
            out._agg_fn = self._agg_fn
            out._agg_source_name = self._agg_source_name
            out._agg_source_eval = self._agg_source_eval
        # Preserve explode metadata (used by DataFrame.select / withColumn).
        if getattr(self, "_is_explode", False):
            out._is_explode = True
            out._explode_source_eval = getattr(self, "_explode_source_eval", None)
            out._explode_source_name = getattr(self, "_explode_source_name", None)
        return out

    # --- arithmetic (null-propagating) ---
    def _binop(self, other: Any, op: Callable[[Any, Any], Any]) -> "Column":
        other_col = _to_column(other)
        left = self._eval
        right = other_col._eval
        ns = _null_safe_binop(op)
        return Column(lambda d: [ns(a, b) for a, b in zip(left(d), right(d))])

    def _rbinop(self, other: Any, op: Callable[[Any, Any], Any]) -> "Column":
        other_col = _to_column(other)
        left = other_col._eval
        right = self._eval
        ns = _null_safe_binop(op)
        return Column(lambda d: [ns(a, b) for a, b in zip(left(d), right(d))])

    def __add__(self, other): return self._binop(other, lambda a, b: a + b)
    def __radd__(self, other): return self._rbinop(other, lambda a, b: a + b)
    def __sub__(self, other): return self._binop(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._rbinop(other, lambda a, b: a - b)
    def __mul__(self, other): return self._binop(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._rbinop(other, lambda a, b: a * b)
    def __truediv__(self, other): return self._binop(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return self._rbinop(other, lambda a, b: a / b)
    def __mod__(self, other): return self._binop(other, lambda a, b: a % b)
    def __rmod__(self, other): return self._rbinop(other, lambda a, b: a % b)
    def __pow__(self, other): return self._binop(other, lambda a, b: float(a) ** float(b))
    def __rpow__(self, other): return self._rbinop(other, lambda a, b: float(a) ** float(b))

    def __neg__(self):
        fn = self._eval
        return Column(lambda d: [None if v is None else -v for v in fn(d)])

    def __pos__(self):
        return Column(self._eval)

    # --- comparisons (null-propagating) ---
    def __eq__(self, other): return self._binop(other, lambda a, b: a == b)  # type: ignore[override]
    def __ne__(self, other): return self._binop(other, lambda a, b: a != b)  # type: ignore[override]
    def __lt__(self, other): return self._binop(other, lambda a, b: a < b)
    def __le__(self, other): return self._binop(other, lambda a, b: a <= b)
    def __gt__(self, other): return self._binop(other, lambda a, b: a > b)
    def __ge__(self, other): return self._binop(other, lambda a, b: a >= b)

    __hash__ = None  # type: ignore[assignment]

    # --- boolean ops with Kleene 3-valued logic ---
    def __and__(self, other):
        other_col = _to_column(other)
        left = self._eval
        right = other_col._eval
        return Column(lambda d: [_kleene_and(a, b) for a, b in zip(left(d), right(d))])

    def __rand__(self, other):
        return _to_column(other).__and__(self)

    def __or__(self, other):
        other_col = _to_column(other)
        left = self._eval
        right = other_col._eval
        return Column(lambda d: [_kleene_or(a, b) for a, b in zip(left(d), right(d))])

    def __ror__(self, other):
        return _to_column(other).__or__(self)

    def __invert__(self):
        fn = self._eval
        return Column(lambda d: [_kleene_not(v) for v in fn(d)])

    # --- null predicates (always return concrete True/False, never null) ---
    def isNull(self) -> "Column":
        fn = self._eval
        return Column(lambda d: [v is None for v in fn(d)])

    def isNotNull(self) -> "Column":
        fn = self._eval
        return Column(lambda d: [v is not None for v in fn(d)])

    # --- containment ---
    def isin(self, *values: Any) -> "Column":
        # Accept either isin(a, b) or isin([a, b])
        if len(values) == 1 and isinstance(values[0], Iterable) and not isinstance(values[0], (str, bytes)):
            value_list = list(values[0])
        else:
            value_list = list(values)
        value_set = set()
        has_unhashable = False
        for v in value_list:
            try:
                value_set.add(v)
            except TypeError:
                has_unhashable = True
                break
        has_null = None in value_set if not has_unhashable else (None in value_list)

        def _ev(d):
            fn = self._eval
            out = []
            if has_unhashable:
                for v in fn(d):
                    if v is None:
                        out.append(None)
                    elif v in value_list:
                        out.append(True)
                    elif has_null:
                        out.append(None)
                    else:
                        out.append(False)
            else:
                for v in fn(d):
                    if v is None:
                        out.append(None)
                    elif v in value_set:
                        out.append(True)
                    elif has_null:
                        out.append(None)
                    else:
                        out.append(False)
            return out
        return Column(_ev)

    def between(self, lower: Any, upper: Any) -> "Column":
        return (self >= lower) & (self <= upper)

    # --- string predicates ---
    def rlike(self, pattern: str) -> "Column":
        if not isinstance(pattern, str):
            raise TypeError(f"rlike() expects a string pattern, got {type(pattern).__name__}")
        compiled = re.compile(pattern)
        fn = self._eval
        return Column(lambda d: [None if v is None else bool(compiled.search(v)) for v in fn(d)])

    def contains(self, substring: str) -> "Column":
        if not isinstance(substring, str):
            raise TypeError(f"contains() expects a string substring, got {type(substring).__name__}")
        fn = self._eval
        return Column(lambda d: [None if v is None else (substring in v) for v in fn(d)])

    def startswith(self, prefix: str) -> "Column":
        fn = self._eval
        return Column(lambda d: [None if v is None else v.startswith(prefix) for v in fn(d)])

    def endswith(self, suffix: str) -> "Column":
        fn = self._eval
        return Column(lambda d: [None if v is None else v.endswith(suffix) for v in fn(d)])

    def like(self, pattern: str) -> "Column":
        """SQL LIKE: ``%`` matches any sequence, ``_`` matches any single char.

        Backslash (``\\``) is the default escape: ``\\%`` matches a literal ``%``,
        ``\\_`` matches a literal ``_``, and ``\\\\`` matches a literal ``\\``.
        The pattern is anchored to the full string (Spark semantics).
        """
        if not isinstance(pattern, str):
            raise TypeError(f"like() expects a string pattern, got {type(pattern).__name__}")
        parts: list[str] = ["^"]
        i = 0
        n = len(pattern)
        while i < n:
            ch = pattern[i]
            if ch == "\\" and i + 1 < n:
                nxt = pattern[i + 1]
                if nxt in ("%", "_", "\\"):
                    parts.append(re.escape(nxt))
                    i += 2
                    continue
                # Unknown escape: emit backslash literally then continue.
                parts.append(re.escape(ch))
                i += 1
                continue
            if ch == "%":
                parts.append(".*")
            elif ch == "_":
                parts.append(".")
            else:
                parts.append(re.escape(ch))
            i += 1
        parts.append("$")
        regex = "".join(parts)
        fn = self._eval
        compiled = re.compile(regex, re.DOTALL)
        return Column(lambda d: [None if v is None else bool(compiled.match(v)) for v in fn(d)])

    # --- cast ---
    def cast(self, data_type: Union[DataType, str]) -> "Column":
        dt = self._resolve_type(data_type)
        fn = self._eval
        return Column(lambda d: [dt._cast(v, strict=True) for v in fn(d)])

    def try_cast(self, data_type: Union[DataType, str]) -> "Column":
        dt = self._resolve_type(data_type)
        fn = self._eval
        return Column(lambda d: [dt._cast(v, strict=False) for v in fn(d)])

    @staticmethod
    def _resolve_type(data_type: Union[DataType, str]) -> DataType:
        if isinstance(data_type, DataType):
            return data_type
        if isinstance(data_type, str):
            return spark_type_name_to_type(data_type)
        raise TypeError(f"cast() expects a DataType or str, got {type(data_type).__name__}")

    # --- sort helpers (return self with sort metadata attached) ---
    def asc(self) -> "Column":
        out = Column(self._eval, name=self._name)
        out._sort_key = self._eval
        out._sort_descending = False
        out._sort_nulls_last = False
        return out

    def desc(self) -> "Column":
        out = Column(self._eval, name=self._name)
        out._sort_key = self._eval
        out._sort_descending = True
        out._sort_nulls_last = True
        return out

    def asc_nulls_first(self) -> "Column":
        return self.asc()

    def asc_nulls_last(self) -> "Column":
        out = Column(self._eval, name=self._name)
        out._sort_key = self._eval
        out._sort_descending = False
        out._sort_nulls_last = True
        return out

    def desc_nulls_first(self) -> "Column":
        out = Column(self._eval, name=self._name)
        out._sort_key = self._eval
        out._sort_descending = True
        out._sort_nulls_last = False
        return out

    def desc_nulls_last(self) -> "Column":
        return self.desc()

    # --- struct / map / array indexing ---
    def getItem(self, key: Union[str, int]) -> "Column":
        if not isinstance(key, (str, int)) or isinstance(key, bool):
            raise TypeError(f"getItem expects str or int, got {type(key).__name__}")
        fn = self._eval

        def _ev(d):
            out = []
            for v in fn(d):
                if v is None:
                    out.append(None)
                elif isinstance(v, dict):
                    out.append(v.get(key))
                elif isinstance(v, (list, tuple)):
                    try:
                        out.append(v[key])
                    except (IndexError, TypeError):
                        out.append(None)
                else:
                    out.append(None)
            return out

        return Column(_ev, name=str(key) if isinstance(key, str) else self._name)

    def __getitem__(self, key: Union[str, int]) -> "Column":
        return self.getItem(key)

    def getField(self, name: str) -> "Column":
        return self.getItem(name)

    # --- window: applies an aggregate (or ranking) over a partitioned window ---
    def over(self, windowspec) -> "Column":  # type: ignore[no-untyped-def]
        """Apply this Column over a :class:`WindowSpec`.

        Three dispatch modes:

        1. Ranking functions (``F.rank()``, ``F.dense_rank()``, ``F.row_number()``)
           — the Column carries ``_window_rank_kind`` set by the factory; we
           compute the rank inside each partition using the spec's order keys.
        2. Aggregate Columns (``F.sum``, ``F.count``, ``F.mean``, ``F.min``,
           ``F.max``, ``F.first``, ``F.collect_list``, ``F.collect_set``) —
           apply the aggregate over the appropriate frame:
              - No ``orderBy`` and no explicit frame → reduce the whole partition.
              - ``orderBy`` present, no explicit frame → running reduction
                (Spark default RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW;
                we approximate via ROWS).
              - Explicit ``rowsBetween`` / ``rangeBetween`` → use the offsets.
        3. Anything else → ``NotImplementedError`` (Spark allows arbitrary
           expressions over a window with a frame, but our v1 scope is the
           cases above).
        """
        from sparkleframe.pythondf.window import WindowSpec, Window

        if not isinstance(windowspec, WindowSpec):
            raise TypeError(
                f"over() expects a WindowSpec, got {type(windowspec).__name__}"
            )

        partition_keys = windowspec._partition_by
        order_specs = windowspec._order_specs
        frame_start = windowspec._frame_start
        frame_end = windowspec._frame_end
        has_explicit_frame = windowspec._has_explicit_frame

        # Ranking dispatch.
        rank_kind: Optional[str] = getattr(self, "_window_rank_kind", None)
        agg_fn = self._agg_fn
        agg_source_eval = self._agg_source_eval
        agg_source_name = self._agg_source_name
        result_name = self._name

        if rank_kind is None and agg_fn is None:
            raise NotImplementedError(
                "Column.over() only supports aggregate or ranking expressions in pythondf"
            )

        def _partition_rows(d: dict) -> tuple[list[list[int]], list[int]]:
            """Return (partition_indices, partition_order) where:
              - partition_indices[g] is the list of row indices in input order
                that belong to partition ``g``.
              - partition_order is a list of group-ids per partition in
                first-seen order (used to map back).
            (We just return the list of index-lists directly; that's enough.)
            """
            n = _row_count(d)
            if not partition_keys:
                return [list(range(n))]
            key_columns = [list(d[k]) for k in partition_keys]
            groups: dict = {}
            order: list = []
            for i in range(n):
                key = tuple(_hashable(col[i]) for col in key_columns)
                bucket = groups.get(key)
                if bucket is None:
                    bucket = []
                    groups[key] = bucket
                    order.append(key)
                bucket.append(i)
            return [groups[k] for k in order]

        def _sorted_within_partition(d: dict, indices: list[int]) -> list[int]:
            if not order_specs:
                return list(indices)
            # Evaluate each order key once over the entire frame, then index.
            key_values: list[tuple[list, bool, bool]] = []
            for fn, desc, nulls_last, _ in order_specs:
                vals = fn(d)
                key_values.append((vals, desc, nulls_last))

            def _key(i: int):
                parts = []
                for vals, desc, nulls_last in key_values:
                    v = vals[i]
                    is_null = v is None
                    if nulls_last:
                        null_marker = 1 if is_null else 0
                    else:
                        null_marker = 0 if is_null else 1
                    if is_null:
                        parts.append((null_marker, 0))
                    else:
                        if desc:
                            parts.append((null_marker, _ReverseKey(v)))
                        else:
                            parts.append((null_marker, v))
                return tuple(parts)

            return sorted(indices, key=_key)

        def _evaluate(d: dict) -> list:
            n = _row_count(d)
            out: list = [None] * n
            partitions = _partition_rows(d)

            # Pre-evaluate the aggregate source once for the whole frame; we'll
            # index into it per row.
            if agg_fn is not None:
                if agg_source_name == "*":
                    source_values: list = [1] * n
                elif agg_source_eval is not None:
                    source_values = agg_source_eval(d)
                else:
                    source_values = [None] * n
            else:
                source_values = []

            for indices in partitions:
                sorted_indices = _sorted_within_partition(d, indices)

                if rank_kind is not None:
                    # We sort with the *same* key tuple used in ordering; rows
                    # that compare equal under all order keys are ties.
                    key_columns: list[tuple[list, bool, bool]] = []
                    for fn, desc, nulls_last, _ in order_specs:
                        vals = fn(d)
                        key_columns.append((vals, desc, nulls_last))

                    def _key_tuple(idx: int) -> tuple:
                        parts = []
                        for vals, desc, nulls_last in key_columns:
                            v = vals[idx]
                            is_null = v is None
                            if nulls_last:
                                null_marker = 1 if is_null else 0
                            else:
                                null_marker = 0 if is_null else 1
                            if is_null:
                                parts.append((null_marker, None))
                            else:
                                if desc:
                                    parts.append((null_marker, _ReverseKeyVal(v)))
                                else:
                                    parts.append((null_marker, v))
                        return tuple(parts)

                    if rank_kind == "row_number":
                        for pos, idx in enumerate(sorted_indices, start=1):
                            out[idx] = pos
                    elif rank_kind == "rank":
                        # 1, 2, 2, 4 (skip after ties).
                        prev_key = object()
                        current_rank = 0
                        for pos, idx in enumerate(sorted_indices, start=1):
                            k = _key_tuple(idx) if order_specs else (pos,)
                            if order_specs and k == prev_key:
                                pass  # same rank
                            else:
                                current_rank = pos
                                prev_key = k
                            out[idx] = current_rank
                    elif rank_kind == "dense_rank":
                        prev_key = object()
                        current_rank = 0
                        for pos, idx in enumerate(sorted_indices, start=1):
                            k = _key_tuple(idx) if order_specs else (pos,)
                            if order_specs and k == prev_key:
                                pass
                            else:
                                current_rank += 1
                                prev_key = k
                            out[idx] = current_rank
                    else:  # pragma: no cover — defensive
                        raise ValueError(f"unknown rank kind: {rank_kind}")
                    continue

                # Aggregate-over-window dispatch.
                m = len(sorted_indices)
                partition_source = [source_values[i] for i in sorted_indices]
                # Cumulative whole-partition (no orderBy, no explicit frame):
                #   every row in partition gets the same aggregate over the
                #   whole partition.
                # Cumulative running (orderBy, no explicit frame): each row
                #   gets the aggregate over [0..pos].
                # Explicit frame: clamp [pos+start, pos+end].
                if has_explicit_frame:
                    # Treat values close to Java MIN/MAX longs as unbounded.
                    _UNBOUNDED_THRESHOLD = 1 << 62
                    for pos, idx in enumerate(sorted_indices):
                        if frame_start is not None and frame_start <= -_UNBOUNDED_THRESHOLD:
                            lo = 0
                        else:
                            lo = pos + (frame_start or 0)
                        if frame_end is not None and frame_end >= _UNBOUNDED_THRESHOLD:
                            hi = m - 1
                        else:
                            hi = pos + (frame_end or 0)
                        lo = max(0, lo)
                        hi = min(m - 1, hi)
                        if lo > hi:
                            window_vals = []
                        else:
                            window_vals = partition_source[lo:hi + 1]
                        out[idx] = agg_fn(window_vals)
                elif order_specs:
                    # Running aggregation: [0..pos].
                    for pos, idx in enumerate(sorted_indices):
                        out[idx] = agg_fn(partition_source[:pos + 1])
                else:
                    # Whole-partition aggregation.
                    reduced = agg_fn(partition_source)
                    for idx in sorted_indices:
                        out[idx] = reduced
            return out

        result = Column(_evaluate, name=result_name)
        # The result of .over() is no longer an aggregate; it produces one
        # value per row. Do not propagate _agg_* metadata so downstream
        # .select() treats it as a regular projection.
        return result


def _hashable(v: Any) -> Any:
    """Hashable representation for partition-key bucketing."""
    if v is None or isinstance(v, (str, int, float, bool, bytes)):
        return v
    try:
        hash(v)
        return v
    except TypeError:
        import json as _json
        return ("__json__", _json.dumps(v, sort_keys=True, default=str))


class _ReverseKey:
    """Wraps a sort value so ascending compare reverses to descending."""

    __slots__ = ("v",)

    def __init__(self, v: Any) -> None:
        self.v = v

    def __lt__(self, other: "_ReverseKey") -> bool:
        return other.v < self.v

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        return isinstance(other, _ReverseKey) and self.v == other.v


class _ReverseKeyVal(_ReverseKey):
    """Same as :class:`_ReverseKey` but also supports ``==`` for tie detection
    in rank()/dense_rank().
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReverseKey) and self.v == other.v

    def __hash__(self) -> int:  # noqa: D401 — needed for tuple-equality fallback
        try:
            return hash(self.v)
        except TypeError:
            return id(self)


def _to_column(x: Any) -> Column:
    if isinstance(x, Column):
        return x
    return Column(lambda d, v=x: [v] * _row_count(d))
