from __future__ import annotations

import datetime as _dt
import decimal as _decimal
import json
from typing import Any, Iterable, Iterator, Optional, Union

from sparkleframe.pythondf.column import Column
from sparkleframe.pythondf.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def _infer_type_from_value(value: Any) -> DataType:
    """Infer a Spark DataType from a single Python value (PySpark inference rules).

    Mirrors PySpark's _infer_type defaults: Python int -> LongType (bigint),
    float -> DoubleType, Decimal -> DecimalType(38, 18), datetime -> TimestampType,
    date -> DateType, bool -> BooleanType, str -> StringType, bytes -> BinaryType,
    list/tuple -> ArrayType(inferred element), dict -> MapType.
    """
    if isinstance(value, bool):
        return BooleanType()
    if isinstance(value, int):
        return LongType()
    if isinstance(value, float):
        return DoubleType()
    if isinstance(value, _decimal.Decimal):
        return DecimalType(38, 18)
    if isinstance(value, _dt.datetime):
        return TimestampType()
    if isinstance(value, _dt.date):
        return DateType()
    if isinstance(value, (bytes, bytearray)):
        return BinaryType()
    if isinstance(value, str):
        return StringType()
    if isinstance(value, (list, tuple)):
        # Find first non-null element for element-type inference.
        element_type: DataType = StringType()
        for elem in value:
            if elem is not None:
                element_type = _infer_type_from_value(elem)
                break
        return ArrayType(element_type)
    if isinstance(value, dict):
        # PySpark defaults dict to MapType<inferred key, inferred value>; pick
        # MapType for parity simplicity.
        key_type: DataType = StringType()
        value_type: DataType = StringType()
        for k, v in value.items():
            if k is not None:
                key_type = _infer_type_from_value(k)
                break
        for v in value.values():
            if v is not None:
                value_type = _infer_type_from_value(v)
                break
        return MapType(key_type, value_type)
    # Fallback: treat as string.
    return StringType()


class _ReverseKey:
    """Wraps a value so ascending sort compares it as if descending.

    Used by ``orderBy`` to mix ASC and DESC sort keys in a single composite key.
    """

    __slots__ = ("v",)

    def __init__(self, v: Any) -> None:
        self.v = v

    def __lt__(self, other: "_ReverseKey") -> bool:
        return other.v < self.v

    def __eq__(self, other: object) -> bool:  # pragma: no cover — only ``<`` is used by sorted()
        return isinstance(other, _ReverseKey) and self.v == other.v


def _make_hashable(v: Any) -> Any:
    """Return a hashable representation of ``v`` for set-based dedup."""
    if v is None or isinstance(v, (str, int, float, bool, bytes)):
        return v
    try:
        hash(v)
        return v
    except TypeError:
        # Stable JSON-based fallback for dict/list-shaped values.
        return ("__json__", json.dumps(v, sort_keys=True, default=str))


class DataFrame:
    """Pure-Python DataFrame backed by a dict-of-lists.

    Layout: ``self._data`` maps column name -> list of values. ``self._columns``
    preserves column order. All ops return a new DataFrame; no mutation.
    """

    def __init__(
        self,
        data: Union[dict, list, "DataFrame"],
        schema: Optional[Union[Iterable[str], StructType]] = None,
    ):
        # Capture the original schema (if a StructType) so we can preserve typed
        # field metadata on .schema/.dtypes; resolve a list of column names for
        # downstream layout code.
        self._typed_schema: Optional[StructType] = None
        if isinstance(schema, StructType):
            self._typed_schema = schema
            schema = schema.fieldNames()

        if isinstance(data, DataFrame):
            self._data = {k: list(v) for k, v in data._data.items()}
            self._columns = list(data._columns)
            return

        if isinstance(data, dict):
            self._data = {k: list(v) for k, v in data.items()}
            self._columns = list(data.keys())
            return

        if isinstance(data, list):
            schema_list = list(schema) if schema is not None else None
            if not data:
                self._data = {c: [] for c in (schema_list or [])}
                self._columns = list(schema_list or [])
                return

            first = data[0]
            if isinstance(first, dict):
                cols: list[str] = []
                seen: set[str] = set()
                source = schema_list if schema_list is not None else None
                if source is None:
                    for row in data:
                        for k in row.keys():
                            if k not in seen:
                                seen.add(k)
                                cols.append(k)
                else:
                    cols = list(source)
                self._data = {c: [row.get(c) for row in data] for c in cols}
                self._columns = cols
                return

            if isinstance(first, (list, tuple)):
                if schema_list is None:
                    raise ValueError("schema (list of column names) required for row-tuple input")
                self._data = {name: [row[i] for row in data] for i, name in enumerate(schema_list)}
                self._columns = list(schema_list)
                return

            raise TypeError(f"Unsupported row type: {type(first).__name__}")

        raise TypeError(f"Unsupported data type: {type(data).__name__}")

    @classmethod
    def _from_internal(cls, data: dict, columns: list[str]) -> "DataFrame":
        out = cls.__new__(cls)
        out._data = data
        out._columns = columns
        return out

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def __len__(self) -> int:
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def withColumn(self, name: str, col: Column) -> "DataFrame":
        if getattr(col, "_is_explode", False):
            return self._explode_into(name, col)
        new_vals = col.eval(self._data)
        new_data = {**self._data, name: new_vals}
        new_columns = list(self._columns) if name in self._data else [*self._columns, name]
        return DataFrame._from_internal(new_data, new_columns)

    def _explode_into(self, name: str, col: Column) -> "DataFrame":
        """Row-multiplying explode: replace each input row with one row per
        element of ``col`` evaluated against that row. Null/empty arrays
        produce a single row with ``None`` in the exploded column (matches
        polars/explode_outer semantics — see :func:`functions.explode`).
        """
        arrays = col._explode_source_eval(self._data)
        source_name = getattr(col, "_explode_source_name", None)
        replace_existing = source_name is not None and source_name == name and name in self._data
        # Decide output column layout.
        if replace_existing:
            new_columns = list(self._columns)
        elif name in self._data:
            new_columns = list(self._columns)
        else:
            new_columns = [*self._columns, name]

        new_data: dict[str, list] = {c: [] for c in new_columns}
        n = len(self)
        for i in range(n):
            arr = arrays[i]
            if arr is None or (isinstance(arr, (list, tuple)) and len(arr) == 0):
                # Emit one null row (Polars/explode_outer behaviour).
                for c in self._columns:
                    if c == name:
                        new_data[c].append(None)
                    else:
                        new_data[c].append(self._data[c][i])
                if name not in self._columns:
                    new_data[name].append(None)
            elif isinstance(arr, (list, tuple)):
                for elem in arr:
                    for c in self._columns:
                        if c == name:
                            new_data[c].append(elem)
                        else:
                            new_data[c].append(self._data[c][i])
                    if name not in self._columns:
                        new_data[name].append(elem)
            else:
                # Non-array value — leave as-is (one row).
                for c in self._columns:
                    if c == name:
                        new_data[c].append(arr)
                    else:
                        new_data[c].append(self._data[c][i])
                if name not in self._columns:
                    new_data[name].append(arr)
        return DataFrame._from_internal(new_data, new_columns)

    def select(self, *cols: Union[str, Column, list, tuple]) -> "DataFrame":
        # PySpark accepts a single list/tuple argument as well as varargs.
        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = tuple(cols[0])
        # Detect explode in selection: it changes row count and is restricted
        # to one explode per select() in PySpark. We support a single explode
        # combined with passthrough columns sourced from the original frame.
        explode_targets = [
            (i, c) for i, c in enumerate(cols)
            if isinstance(c, Column) and getattr(c, "_is_explode", False)
        ]
        if explode_targets:
            if len(explode_targets) > 1:
                raise ValueError("Only one explode is allowed per select()")
            return self._select_with_explode(cols, explode_targets[0])

        # Detect aggregate expressions: PySpark collapses to a single row when
        # any selected Column is an aggregate (and disallows mixing with
        # non-aggregate plain columns — we follow that).
        has_agg = any(isinstance(c, Column) and c._agg_fn is not None for c in cols)
        if has_agg:
            new_data: dict[str, list] = {}
            new_columns: list[str] = []
            for c in cols:
                if isinstance(c, str):
                    raise TypeError(
                        "select() cannot mix a plain column name with aggregate expressions"
                    )
                if not isinstance(c, Column):
                    raise TypeError(f"select expects str or Column, got {type(c).__name__}")
                if c._agg_fn is None:
                    raise TypeError(
                        "select() cannot mix non-aggregate Column with aggregate expressions"
                    )
                name = c.name or "expr"
                # Reduce over the entire frame.
                if c._agg_source_name == "*":
                    reduced = c._agg_fn([1] * len(self))
                elif c._agg_source_eval is not None:
                    vals = c._agg_source_eval(self._data)
                    reduced = c._agg_fn(vals)
                else:
                    reduced = c._agg_fn([])
                new_data[name] = [reduced]
                new_columns.append(name)
            return DataFrame._from_internal(new_data, new_columns)

        new_data = {}
        new_columns = []
        for c in cols:
            if isinstance(c, str):
                name = c
                vals = list(self._data[c])
            elif isinstance(c, Column):
                name = c.name or "expr"
                vals = c.eval(self._data)
            else:
                raise TypeError(f"select expects str or Column, got {type(c).__name__}")
            new_data[name] = vals
            new_columns.append(name)
        return DataFrame._from_internal(new_data, new_columns)

    def _select_with_explode(
        self, cols: tuple, exploded: tuple[int, Column]
    ) -> "DataFrame":
        """Materialise a select() that contains exactly one explode column.

        Other selected columns are evaluated against the *original* frame and
        then expanded row-by-row in lockstep with the exploded values.
        """
        _, explode_col = exploded
        arrays = explode_col._explode_source_eval(self._data)

        # Determine the exploded column's output name (alias takes precedence).
        explode_name = explode_col.name or getattr(explode_col, "_explode_source_name", None) or "col"

        # Evaluate all non-explode columns once against the original frame.
        passthrough: list[tuple[str, list]] = []
        for c in cols:
            if isinstance(c, Column) and getattr(c, "_is_explode", False):
                continue
            if isinstance(c, str):
                if c not in self._data:
                    raise KeyError(c)
                passthrough.append((c, list(self._data[c])))
            elif isinstance(c, Column):
                vals = c.eval(self._data)
                name = c.name or "expr"
                passthrough.append((name, vals))
            else:
                raise TypeError(f"select expects str or Column, got {type(c).__name__}")

        # Build output column order matching the original select() argument order.
        out_columns: list[str] = []
        pass_iter = iter(passthrough)
        for c in cols:
            if isinstance(c, Column) and getattr(c, "_is_explode", False):
                out_columns.append(explode_name)
            else:
                out_columns.append(next(pass_iter)[0])

        new_data: dict[str, list] = {name: [] for name in out_columns}
        n = len(self)
        for i in range(n):
            arr = arrays[i]
            if arr is None or (isinstance(arr, (list, tuple)) and len(arr) == 0):
                # Emit one null row (explode_outer semantics).
                for out_name, c in zip(out_columns, cols):
                    if isinstance(c, Column) and getattr(c, "_is_explode", False):
                        new_data[out_name].append(None)
                # Fill passthrough columns from row i.
                for (p_name, p_vals), out_name, c in zip(
                    passthrough, out_columns, cols
                ):
                    pass
                # Re-iterate to fill passthrough values.
                k = 0
                for j, c in enumerate(cols):
                    if isinstance(c, Column) and getattr(c, "_is_explode", False):
                        continue
                    out_name = out_columns[j]
                    new_data[out_name].append(passthrough[k][1][i])
                    k += 1
            elif isinstance(arr, (list, tuple)):
                for elem in arr:
                    k = 0
                    for j, c in enumerate(cols):
                        out_name = out_columns[j]
                        if isinstance(c, Column) and getattr(c, "_is_explode", False):
                            new_data[out_name].append(elem)
                        else:
                            new_data[out_name].append(passthrough[k][1][i])
                            k += 1
            else:
                # Non-array — treat as single-row scalar in the exploded position.
                k = 0
                for j, c in enumerate(cols):
                    out_name = out_columns[j]
                    if isinstance(c, Column) and getattr(c, "_is_explode", False):
                        new_data[out_name].append(arr)
                    else:
                        new_data[out_name].append(passthrough[k][1][i])
                        k += 1
        return DataFrame._from_internal(new_data, out_columns)

    def filter(self, condition: Column) -> "DataFrame":
        mask = condition.eval(self._data)
        new_data = {c: [v for v, keep in zip(self._data[c], mask) if keep is True] for c in self._columns}
        return DataFrame._from_internal(new_data, list(self._columns))

    where = filter

    def withColumns(self, colsMap: dict[str, Column]) -> "DataFrame":  # noqa: N803 — PySpark API
        """Batched ``withColumn``. Entries are applied in dict-iteration order,
        so a later entry may reference a column added by an earlier entry
        (matches PySpark 3.4+ and polarsdf behavior).
        """
        if not isinstance(colsMap, dict):
            raise TypeError(f"withColumns expects a dict, got {type(colsMap).__name__}")
        result = self
        for name, col_expr in colsMap.items():
            if not isinstance(col_expr, Column):
                raise TypeError(
                    f"withColumns values must be Column expressions; got {type(col_expr).__name__} for {name!r}"
                )
            result = result.withColumn(name, col_expr)
        return result

    def withColumnRenamed(self, existing: str, new: str) -> "DataFrame":  # noqa: N802 — PySpark API
        """Rename a single column. No-op if ``existing`` is not present.

        Preserves column order; the renamed column occupies the same position.
        """
        if existing not in self._data:
            return DataFrame._from_internal(
                {k: list(v) for k, v in self._data.items()}, list(self._columns)
            )
        new_columns = [new if c == existing else c for c in self._columns]
        new_data: dict[str, list] = {}
        for old_name, target in zip(self._columns, new_columns):
            new_data[target] = list(self._data[old_name])
        return DataFrame._from_internal(new_data, new_columns)

    def drop(self, *cols: Union[str, Column]) -> "DataFrame":
        """Remove columns by name. Absent names are ignored (PySpark semantics).

        Column references must have a ``.name`` (e.g., a bare ``col("x")``); an
        anonymous expression raises ``TypeError``.
        """
        if not cols:
            return DataFrame._from_internal(
                {k: list(v) for k, v in self._data.items()}, list(self._columns)
            )

        to_drop: list[str] = []
        for c in cols:
            if isinstance(c, str):
                to_drop.append(c)
            elif isinstance(c, Column):
                if not c.name:
                    raise TypeError("drop() Column expression must reference a named column")
                to_drop.append(c.name)
            else:
                raise TypeError(f"drop() expected str or Column, got {type(c).__name__}")

        drop_set = set(to_drop)
        new_columns = [c for c in self._columns if c not in drop_set]
        new_data = {c: list(self._data[c]) for c in new_columns}
        return DataFrame._from_internal(new_data, new_columns)

    def collect(self) -> list[dict]:
        n = len(self)
        return [{c: self._data[c][i] for c in self._columns} for i in range(n)]

    # ----- ordering -----

    def orderBy(self, *cols: Union[str, Column]) -> "DataFrame":  # noqa: N802 — PySpark API
        """Sort rows by one or more keys.

        Accepted argument forms (mirrors PySpark):
          * ``str`` column name — ASC, NULLS FIRST (PySpark default).
          * Bare ``Column`` expression (no ``.asc()/.desc()`` attached) — ASC,
            NULLS FIRST.
          * Sort-keyed ``Column`` (``F.col("x").desc()``, ``.asc_nulls_last()``,
            etc.) — uses the metadata captured by ``Column.asc/desc/...``.
        """
        if len(cols) == 1 and isinstance(cols[0], list):
            cols = tuple(cols[0])

        # Build (eval_fn, descending, nulls_last) for each key.
        keys: list[tuple] = []
        for c in cols:
            if isinstance(c, str):
                if c not in self._data:
                    raise KeyError(c)
                col_name = c
                eval_fn = (lambda d, n=col_name: list(d[n]))
                keys.append((eval_fn, False, False))
            elif isinstance(c, Column):
                if c._sort_key is not None:
                    keys.append((c._sort_key, c._sort_descending, c._sort_nulls_last))
                else:
                    keys.append((c._eval, False, False))
            else:
                raise TypeError(f"orderBy expects str or Column, got {type(c).__name__}")

        n = len(self)
        if n == 0 or not keys:
            return DataFrame._from_internal(
                {k: list(v) for k, v in self._data.items()}, list(self._columns)
            )

        # Pre-compute each key column once.
        key_columns: list[tuple[list, bool, bool]] = [
            (eval_fn(self._data), desc, nulls_last) for eval_fn, desc, nulls_last in keys
        ]

        def _row_key(i: int):
            parts: list = []
            for vals, desc, nulls_last in key_columns:
                v = vals[i]
                is_null = v is None
                # Push nulls to the appropriate end. Tuple component 0 controls
                # null placement: 0 sorts before 1 in ascending tuple-compare.
                # NULLS FIRST -> nulls get 0, non-nulls get 1.
                # NULLS LAST  -> nulls get 1, non-nulls get 0.
                if nulls_last:
                    null_marker = 1 if is_null else 0
                else:
                    null_marker = 0 if is_null else 1
                if is_null:
                    # Placeholder value for None (avoids cross-type compare crashes).
                    parts.append((null_marker, 0))
                else:
                    if desc:
                        parts.append((null_marker, _ReverseKey(v)))
                    else:
                        parts.append((null_marker, v))
            return tuple(parts)

        order = sorted(range(n), key=_row_key)
        new_data = {c: [self._data[c][i] for i in order] for c in self._columns}
        return DataFrame._from_internal(new_data, list(self._columns))

    sort = orderBy

    # ----- distinct / dedup -----

    def distinct(self) -> "DataFrame":
        """Drop duplicate rows. Preserves first-seen order."""
        return self.dropDuplicates()

    def dropDuplicates(self, subset: Optional[list[str]] = None) -> "DataFrame":  # noqa: N802 — PySpark API
        """Drop duplicate rows keyed on ``subset`` (or every column if None).

        First occurrence is kept; subsequent duplicates are dropped.
        """
        if subset is not None:
            for s in subset:
                if s not in self._data:
                    raise KeyError(s)
            key_cols = list(subset)
        else:
            key_cols = list(self._columns)

        seen: set = set()
        kept_indices: list[int] = []
        n = len(self)
        for i in range(n):
            key_parts = []
            for c in key_cols:
                v = self._data[c][i]
                key_parts.append(_make_hashable(v))
            key = tuple(key_parts)
            if key in seen:
                continue
            seen.add(key)
            kept_indices.append(i)

        new_data = {c: [self._data[c][i] for i in kept_indices] for c in self._columns}
        return DataFrame._from_internal(new_data, list(self._columns))

    # ----- slicing -----

    def limit(self, n: int) -> "DataFrame":
        """Return a new DataFrame with the first ``n`` rows."""
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f"limit() expects int, got {type(n).__name__}")
        n = max(0, n)
        new_data = {c: self._data[c][:n] for c in self._columns}
        return DataFrame._from_internal(new_data, list(self._columns))

    def head(self, n: Optional[int] = None):
        """Return the first row (as dict) or first ``n`` rows (as list of dicts).

        Matches PySpark's overload: ``head()`` and ``head(1)`` return a single
        row (dict here; ``None`` if the frame is empty); ``head(n)`` for any
        other ``n`` returns a list of dicts (possibly empty).
        """
        if n is None:
            rows = self.collect()[:1]
            return rows[0] if rows else None
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f"head() expects int, got {type(n).__name__}")
        return self.collect()[: max(0, n)]

    def take(self, n: int) -> list[dict]:
        """Return the first ``n`` rows as a list of dicts (alias for ``head(n)``)."""
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f"take() expects int, got {type(n).__name__}")
        return self.collect()[: max(0, n)]

    def first(self) -> Optional[dict]:
        """Return the first row as a dict, or ``None`` if the frame is empty."""
        rows = self.collect()[:1]
        return rows[0] if rows else None

    # ----- metadata -----

    def count(self) -> int:
        """Return the total row count."""
        return len(self)

    def isEmpty(self) -> bool:  # noqa: N802 — PySpark API
        """Return True if the DataFrame contains no rows."""
        return len(self) == 0

    @property
    def schema(self) -> StructType:
        """Return a ``StructType`` describing the frame's columns.

        If a typed ``StructType`` was supplied to ``createDataFrame`` it is
        echoed back verbatim (filtered to columns still present). Otherwise we
        infer types from cell values: for each column, sample the first
        non-null value (PySpark ``_infer_type`` rules). Empty / all-null
        columns default to ``StringType`` (PySpark's inferred-empty default).
        """
        if getattr(self, "_typed_schema", None) is not None:
            # Preserve declared field metadata for columns that survive any
            # downstream projection.
            declared = {f.name: f for f in self._typed_schema.fields}
            fields = [declared[c] for c in self._columns if c in declared]
            if len(fields) == len(self._columns):
                return StructType(fields)
        fields: list[StructField] = []
        for col_name in self._columns:
            values = self._data.get(col_name, [])
            inferred: DataType = StringType()
            for v in values:
                if v is not None:
                    inferred = _infer_type_from_value(v)
                    break
            fields.append(StructField(col_name, inferred, nullable=True))
        return StructType(fields)

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        """Return ``[(name, simpleString)]`` for each column (PySpark format)."""
        return [(f.name, f.dataType.simpleString()) for f in self.schema]

    def printSchema(self) -> None:  # noqa: N802 — PySpark API
        """Best-effort tree-style schema print (matches PySpark's format)."""
        print("root")
        for f in self.schema:
            print(f" |-- {f.name}: {f.dataType.simpleString()} (nullable = {str(f.nullable).lower()})")

    # ----- set operations -----

    def union(self, other: "DataFrame") -> "DataFrame":
        """UNION ALL by position. Column ordering follows ``self``."""
        if not isinstance(other, DataFrame):
            raise TypeError("union() expects a DataFrame")
        left_cols = self._columns
        right_cols = other._columns
        if len(left_cols) != len(right_cols):
            raise ValueError(
                f"union() requires same number of columns; left={len(left_cols)}, right={len(right_cols)}"
            )
        new_data: dict[str, list] = {}
        for i, name in enumerate(left_cols):
            right_name = right_cols[i]
            new_data[name] = list(self._data[name]) + list(other._data[right_name])
        return DataFrame._from_internal(new_data, list(left_cols))

    unionAll = union

    def unionByName(self, other: "DataFrame", allowMissingColumns: bool = False) -> "DataFrame":  # noqa: N802, N803
        """UNION ALL by column name. Column ordering follows ``self``.

        When ``allowMissingColumns=False`` (default), both frames must share the
        same set of column names. With ``allowMissingColumns=True``, missing
        columns are filled with ``None`` on either side.
        """
        if not isinstance(other, DataFrame):
            raise TypeError("unionByName() expects a DataFrame")

        left_cols = list(self._columns)
        left_set = set(left_cols)
        right_set = set(other._columns)

        n_left = len(self)
        n_right = len(other)

        if not allowMissingColumns:
            if left_set != right_set:
                missing_in_right = sorted(left_set - right_set)
                extra_in_right = sorted(right_set - left_set)
                msg_parts: list[str] = []
                if missing_in_right:
                    msg_parts.append(f"columns not in the right frame: {missing_in_right}")
                if extra_in_right:
                    msg_parts.append(f"columns only in the right frame: {extra_in_right}")
                raise ValueError(
                    "unionByName() requires the same column names when allowMissingColumns=False ("
                    + "; ".join(msg_parts)
                    + "). Use allowMissingColumns=True to union with null padding."
                )
            new_data = {c: list(self._data[c]) + list(other._data[c]) for c in left_cols}
            return DataFrame._from_internal(new_data, list(left_cols))

        # allowMissingColumns=True: include union of column names; preserve self
        # order, then extra right-only columns in their original right-side order.
        extra_right = [c for c in other._columns if c not in left_set]
        all_cols = left_cols + extra_right
        new_data = {}
        for c in all_cols:
            left_vals = list(self._data[c]) if c in self._data else [None] * n_left
            right_vals = list(other._data[c]) if c in other._data else [None] * n_right
            new_data[c] = left_vals + right_vals
        return DataFrame._from_internal(new_data, all_cols)

    # ----- groupBy -----

    def groupBy(self, *cols: Union[str, Column]):  # noqa: N802 — PySpark API
        """Return a :class:`~sparkleframe.pythondf.group.GroupedData` for the
        given grouping columns. With zero arguments, behaves like a global
        aggregation (single group containing all rows).
        """
        from sparkleframe.pythondf.group import GroupedData

        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = tuple(cols[0])
        return GroupedData(self, list(cols))

    groupby = groupBy

    # ----- join -----

    _VALID_JOIN_HOWS = {
        "inner",
        "cross",
        "outer", "full", "fullouter", "full_outer",
        "left", "leftouter", "left_outer",
        "right", "rightouter", "right_outer",
        "semi", "leftsemi", "left_semi",
        "anti", "leftanti", "left_anti",
    }

    def join(
        self,
        other: "DataFrame",
        on: Optional[Union[str, list, Column]] = None,
        how: str = "inner",
    ) -> "DataFrame":
        """Join two DataFrames.

        Supports ``on`` as a single column name (``str``) or a list of column
        names. Predicate (``Column``) joins are not supported (raises
        ``NotImplementedError``) — pythondf Columns are independent of any
        specific DataFrame, so resolving a side requires extra plumbing that's
        out of scope for the v1 parity work.

        ``how`` accepts every PySpark variant: ``inner``, ``cross``,
        ``left``/``leftouter``/``left_outer``, ``right``/``rightouter``/
        ``right_outer``, ``outer``/``full``/``fullouter``/``full_outer``,
        ``semi``/``leftsemi``/``left_semi``, ``anti``/``leftanti``/``left_anti``.
        """
        if not isinstance(other, DataFrame):
            raise TypeError(f"join() expects a DataFrame, got {type(other).__name__}")

        how_normalized = how.lower()
        if how_normalized not in self._VALID_JOIN_HOWS:
            raise ValueError(f"Unsupported join type: '{how}'")

        # Normalize ``on`` into a list[str] of key column names (or [] for cross/None).
        if on is None:
            on_keys: list[str] = []
        elif isinstance(on, str):
            on_keys = [on]
        elif isinstance(on, Column):
            raise NotImplementedError(
                "pythondf does not support predicate (Column) joins; pass a key "
                "column name or list of names instead."
            )
        elif isinstance(on, list):
            if not on:
                on_keys = []
            else:
                first_type = type(on[0])
                for n in on:
                    if type(n) is not first_type:
                        raise TypeError(
                            "On columns must have the same type. str or List[str] or Column or List[Column], None)"
                        )
                if isinstance(on[0], Column):
                    raise NotImplementedError(
                        "pythondf does not support predicate (Column) joins; pass a key "
                        "column name or list of names instead."
                    )
                if not all(isinstance(n, str) for n in on):
                    raise TypeError(f"join() expects on as str, list[str], or Column; got {type(on[0]).__name__}")
                on_keys = list(on)
        else:
            raise TypeError(f"join() expects on as str, list[str], or Column; got {type(on).__name__}")

        # Cross join: cartesian product.
        if how_normalized == "cross":
            if on_keys:
                raise ValueError("cross join does not accept 'on'")
            return self._cross_join(other)

        # Semi / anti: only left rows; only left columns.
        if how_normalized in {"semi", "leftsemi", "left_semi"}:
            if not on_keys:
                raise ValueError("semi join requires 'on'")
            return self._semi_anti_join(other, on_keys, anti=False)
        if how_normalized in {"anti", "leftanti", "left_anti"}:
            if not on_keys:
                raise ValueError("anti join requires 'on'")
            return self._semi_anti_join(other, on_keys, anti=True)

        # Equi-joins: inner / left / right / outer.
        if not on_keys:
            # Spark allows omitted ``on`` to mean "common columns"; mirror that.
            common = [c for c in self._columns if c in other._data]
            if not common:
                raise ValueError("join() with no 'on' requires common columns")
            on_keys = common

        for k in on_keys:
            if k not in self._data:
                raise KeyError(f"join key {k!r} not in left frame")
            if k not in other._data:
                raise KeyError(f"join key {k!r} not in right frame")

        # Detect non-key name collisions (PySpark allows this but creates ambiguous refs;
        # we follow polarsdf's behaviour of producing ``_right``-suffixed columns).
        left_non_keys = [c for c in self._columns if c not in on_keys]
        right_non_keys = [c for c in other._columns if c not in on_keys]
        right_rename: dict[str, str] = {}
        left_non_key_set = set(left_non_keys)
        for c in right_non_keys:
            if c in left_non_key_set:
                right_rename[c] = c + "_right"

        return self._equi_join(other, on_keys, how_normalized, left_non_keys, right_non_keys, right_rename)

    def _cross_join(self, other: "DataFrame") -> "DataFrame":
        left_cols = list(self._columns)
        right_cols = list(other._columns)
        # Disambiguate name collisions.
        right_rename: dict[str, str] = {}
        left_set = set(left_cols)
        for c in right_cols:
            if c in left_set:
                right_rename[c] = c + "_right"
        out_left_cols = left_cols
        out_right_cols = [right_rename.get(c, c) for c in right_cols]
        out_columns = out_left_cols + out_right_cols

        n_left = len(self)
        n_right = len(other)
        new_data: dict[str, list] = {c: [] for c in out_columns}
        for i in range(n_left):
            for j in range(n_right):
                for c in left_cols:
                    new_data[c].append(self._data[c][i])
                for c in right_cols:
                    new_data[right_rename.get(c, c)].append(other._data[c][j])
        return DataFrame._from_internal(new_data, out_columns)

    def _semi_anti_join(
        self, other: "DataFrame", on_keys: list[str], anti: bool
    ) -> "DataFrame":
        for k in on_keys:
            if k not in self._data:
                raise KeyError(f"join key {k!r} not in left frame")
            if k not in other._data:
                raise KeyError(f"join key {k!r} not in right frame")
        right_keys: set = set()
        n_right = len(other)
        for j in range(n_right):
            key = tuple(_make_hashable(other._data[k][j]) for k in on_keys)
            right_keys.add(key)
        kept: list[int] = []
        n_left = len(self)
        for i in range(n_left):
            key = tuple(_make_hashable(self._data[k][i]) for k in on_keys)
            present = key in right_keys
            if anti and not present:
                kept.append(i)
            elif (not anti) and present:
                kept.append(i)
        new_data = {c: [self._data[c][i] for i in kept] for c in self._columns}
        return DataFrame._from_internal(new_data, list(self._columns))

    def _equi_join(
        self,
        other: "DataFrame",
        on_keys: list[str],
        how: str,
        left_non_keys: list[str],
        right_non_keys: list[str],
        right_rename: dict[str, str],
    ) -> "DataFrame":
        # Output column ordering: PySpark's equi-join (on=str|list[str]) puts the
        # key columns first (single, coalesced), then the remaining left columns
        # in original order, then the remaining right columns in original order.
        out_right_non_key_names = [right_rename.get(c, c) for c in right_non_keys]
        out_columns = list(on_keys) + list(left_non_keys) + out_right_non_key_names
        new_data: dict[str, list] = {c: [] for c in out_columns}

        # Build a hash table on the right side keyed by the join keys.
        right_index: dict[tuple, list[int]] = {}
        n_right = len(other)
        for j in range(n_right):
            key = tuple(_make_hashable(other._data[k][j]) for k in on_keys)
            right_index.setdefault(key, []).append(j)

        n_left = len(self)
        matched_right: set[int] = set()

        is_left = how in {"left", "leftouter", "left_outer", "outer", "full", "fullouter", "full_outer"}
        is_right = how in {"right", "rightouter", "right_outer", "outer", "full", "fullouter", "full_outer"}

        for i in range(n_left):
            key = tuple(_make_hashable(self._data[k][i]) for k in on_keys)
            matches = right_index.get(key)
            if matches:
                for j in matches:
                    matched_right.add(j)
                    for idx, k in enumerate(on_keys):
                        # For inner/left/right/outer with a key match, both sides
                        # have the same key value — pick from left for symmetry.
                        new_data[k].append(self._data[k][i])
                    for c in left_non_keys:
                        new_data[c].append(self._data[c][i])
                    for c, out_c in zip(right_non_keys, out_right_non_key_names):
                        new_data[out_c].append(other._data[c][j])
            else:
                # Left row had no right match: emit only for left/outer.
                if is_left:
                    for k in on_keys:
                        new_data[k].append(self._data[k][i])
                    for c in left_non_keys:
                        new_data[c].append(self._data[c][i])
                    for out_c in out_right_non_key_names:
                        new_data[out_c].append(None)

        if is_right:
            # Emit right rows that had no left match.
            for j in range(n_right):
                if j in matched_right:
                    continue
                # Use right's key values (left side had no row to source from).
                for k in on_keys:
                    new_data[k].append(other._data[k][j])
                for c in left_non_keys:
                    new_data[c].append(None)
                for c, out_c in zip(right_non_keys, out_right_non_key_names):
                    new_data[out_c].append(other._data[c][j])

        return DataFrame._from_internal(new_data, out_columns)

    # ----- JSON export -----

    def toJSON(self) -> Iterator[str]:  # noqa: N802 — PySpark API
        """Return an iterable of JSON strings, one per row.

        PySpark's ``toJSON`` returns an RDD of JSON strings; we return a stdlib
        iterator (no Spark-RDD parity).
        """
        def _default(o: Any) -> Any:
            if isinstance(o, (_dt.datetime, _dt.date)):
                return o.isoformat()
            if isinstance(o, _decimal.Decimal):
                return str(o)
            if isinstance(o, (bytes, bytearray)):
                return o.decode("utf-8", errors="replace")
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
        for row in self.collect():
            yield json.dumps(row, default=_default)

    def toPandas(self):  # noqa: N802 — PySpark API
        import pandas as pd
        return pd.DataFrame(self._data, columns=self._columns)

    def show(self, n: int = 20) -> None:
        print(self._format(n))

    def _format(self, n: int = 20) -> str:
        cols = self._columns
        if not cols:
            return "(empty DataFrame)"
        rows = [[str(self._data[c][i]) for c in cols] for i in range(min(n, len(self)))]
        widths = [max(len(c), *(len(row[i]) for row in rows)) if rows else len(c) for i, c in enumerate(cols)]
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        header = "|" + "|".join(f" {c:<{widths[i]}} " for i, c in enumerate(cols)) + "|"
        lines = [sep, header, sep]
        for row in rows:
            lines.append("|" + "|".join(f" {row[i]:<{widths[i]}} " for i in range(len(cols))) + "|")
        lines.append(sep)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DataFrame[{', '.join(self._columns)}]"
