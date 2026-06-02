"""GroupedData for pythondf DataFrame.

Produced by ``DataFrame.groupBy(*cols)``. Reduces each group via aggregate
:class:`~sparkleframe.pythondf.column.Column` expressions carrying
``_agg_fn`` / ``_agg_source_*`` metadata (see ``functions.sum``,
``functions.count``, etc.).

PySpark output naming: aggregate columns produced without ``.alias(...)``
get a name like ``sum(a)`` / ``count(1)`` (for ``F.count("*")``). With
``.alias("x")`` the user-supplied name wins.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable, Union

from sparkleframe.pythondf.column import Column

if TYPE_CHECKING:
    from sparkleframe.pythondf.dataframe import DataFrame


def _hashable_key(value: Any):
    """Hashable representation of a group-key cell value.

    Lists/dicts (PySpark group keys are typically scalar but we don't enforce)
    are normalized via JSON for stable equality.
    """
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    try:
        hash(value)
        return value
    except TypeError:
        return ("__json__", json.dumps(value, sort_keys=True, default=str))


class GroupedData:
    """The result of ``DataFrame.groupBy(*cols)``.

    Internal layout: keep the source DataFrame plus the resolved list of group
    column names. Aggregations partition the rows by group key and apply
    aggregate functions to each partition.
    """

    def __init__(self, df: "DataFrame", group_cols: Iterable[Union[str, Column]]):
        self._df = df
        names: list[str] = []
        for c in group_cols:
            if isinstance(c, str):
                names.append(c)
            elif isinstance(c, Column):
                if not c.name:
                    raise TypeError("groupBy() Column expression must have a name (use .alias(...) or F.col(...))")
                names.append(c.name)
            else:
                raise TypeError(f"groupBy expects str or Column, got {type(c).__name__}")
        for n in names:
            if n not in df._data:
                raise KeyError(n)
        self._group_cols: list[str] = names

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def _partition(self):
        """Return ``(group_keys_ordered, indices_per_group)`` where keys are
        tuples of raw cell values (in ``self._group_cols`` order) and
        ``indices_per_group`` lists the row indices belonging to each key.
        """
        n = len(self._df)
        # Special case: groupBy with no keys = one global group.
        if not self._group_cols:
            return [()], {(): list(range(n))}

        key_to_indices: dict = {}
        ordered_keys: list = []
        for i in range(n):
            raw_key = tuple(self._df._data[c][i] for c in self._group_cols)
            hash_key = tuple(_hashable_key(v) for v in raw_key)
            if hash_key not in key_to_indices:
                key_to_indices[hash_key] = (raw_key, [])
                ordered_keys.append(hash_key)
            key_to_indices[hash_key][1].append(i)

        keys_ordered = [key_to_indices[k][0] for k in ordered_keys]
        indices_ordered = [key_to_indices[k][1] for k in ordered_keys]
        return keys_ordered, indices_ordered

    # ------------------------------------------------------------------
    # agg(*exprs)
    # ------------------------------------------------------------------

    def agg(self, *exprs: Column) -> "DataFrame":
        from sparkleframe.pythondf.dataframe import DataFrame as _DF

        if not exprs:
            raise ValueError("agg() requires at least one expression")
        for e in exprs:
            if not isinstance(e, Column):
                raise TypeError(f"agg() expects Column expressions, got {type(e).__name__}")
            if e._agg_fn is None:
                raise TypeError(
                    "agg() expects aggregate expressions (F.sum, F.count, F.mean, ...); "
                    f"got non-aggregate Column {e.name!r}"
                )

        keys, indices_per_group = self._partition()

        # Output columns: group cols (raw key values), then agg columns.
        out_data: dict[str, list] = {c: [] for c in self._group_cols}
        agg_names: list[str] = []
        agg_outputs: list[list] = []
        for e in exprs:
            agg_names.append(e.name or "expr")
            agg_outputs.append([])

        for key_tuple, idxs in zip(keys, indices_per_group):
            # Append group keys.
            for i, gc in enumerate(self._group_cols):
                out_data[gc].append(key_tuple[i])
            # For each agg, slice the source column to the group's rows and reduce.
            for k, e in enumerate(exprs):
                if e._agg_source_name == "*":
                    reduced = e._agg_fn([1] * len(idxs))
                elif e._agg_source_eval is not None:
                    # Evaluate the source expression against a sub-data dict
                    # restricted to this group's rows.
                    sub = {c: [self._df._data[c][i] for i in idxs] for c in self._df._columns}
                    vals = e._agg_source_eval(sub)
                    reduced = e._agg_fn(vals)
                else:
                    reduced = e._agg_fn([])
                agg_outputs[k].append(reduced)

        for name, vals in zip(agg_names, agg_outputs):
            out_data[name] = vals

        out_columns = list(self._group_cols) + agg_names
        return _DF._from_internal(out_data, out_columns)

    # ------------------------------------------------------------------
    # Shortcut aggregations
    # ------------------------------------------------------------------

    def _shortcut(self, cols: tuple[str, ...], fn_name: str, agg_fn, label_fn=None):
        from sparkleframe.pythondf.functions import _make_agg_column

        label_fn = label_fn or (lambda c: f"{fn_name}({c})")
        if not cols:
            # PySpark fans out across all numeric (for sum/mean/min/max) columns.
            # We don't track numeric vs not; pick every non-group column.
            cols = tuple(c for c in self._df._columns if c not in self._group_cols)

        exprs = []
        for c in cols:
            col = _make_agg_column(c, agg_fn, fn_name, display_label=c)
            # The default label format is fn_name(display); already correct.
            exprs.append(col)
        return self.agg(*exprs)

    def sum(self, *cols: str) -> "DataFrame":  # noqa: A003
        from sparkleframe.pythondf.functions import _sum_ignore_nulls
        return self._shortcut(cols, "sum", _sum_ignore_nulls)

    def mean(self, *cols: str) -> "DataFrame":
        from sparkleframe.pythondf.functions import _mean_ignore_nulls
        return self._shortcut(cols, "avg", _mean_ignore_nulls)

    avg = mean

    def min(self, *cols: str) -> "DataFrame":  # noqa: A003
        from sparkleframe.pythondf.functions import _min_ignore_nulls
        return self._shortcut(cols, "min", _min_ignore_nulls)

    def max(self, *cols: str) -> "DataFrame":  # noqa: A003
        from sparkleframe.pythondf.functions import _max_ignore_nulls
        return self._shortcut(cols, "max", _max_ignore_nulls)

    def count(self) -> "DataFrame":
        """Per-group row count. Single output column ``count`` (long)."""
        from sparkleframe.pythondf.dataframe import DataFrame as _DF

        keys, indices_per_group = self._partition()
        out_data: dict[str, list] = {c: [] for c in self._group_cols}
        out_data["count"] = []
        for key_tuple, idxs in zip(keys, indices_per_group):
            for i, gc in enumerate(self._group_cols):
                out_data[gc].append(key_tuple[i])
            out_data["count"].append(len(idxs))
        out_columns = list(self._group_cols) + ["count"]
        return _DF._from_internal(out_data, out_columns)
