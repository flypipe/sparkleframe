"""``DataFrame`` for the pure-Python engine.

A frame is a list of row tuples plus an engine ``StructType`` schema. Every transformation is
eager (see ``docs/design/python-engine-ast.md``, Q3): its column expressions are resolved through
``analyzer.analyze`` (type resolution against the schema) then run row by row through the
evaluator, producing a new frame.

Each frame has a ``plan id``; ``df["x"]`` builds a column reference qualified by it, and every
output column remembers the ids of the frames it can be referenced through (``_plan_sets``). That
is how ``a.join(b, a["id"] == b["id"])`` tells the two ``id`` columns apart, as Spark does.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sparkleframe.base.dataframe import DataFrame as BaseDataFrame
from sparkleframe.python._errors import not_implemented_yet
from sparkleframe.python.ast import coercion
from sparkleframe.python.ast.analyzer import AnalysisException, analyze
from sparkleframe.python.ast.evaluator import EvalContext, eval_row
from sparkleframe.python.ast.expressions import (
    Alias,
    AttributeReference,
    Expression,
    FunctionCall,
    SortOrder,
    Star,
    WindowExpression,
)
from sparkleframe.python.ast.function_registry import explode_rows, lookup
from sparkleframe.python.column import Column, _column_ref
from sparkleframe.python.column_helpers import cast_value
from sparkleframe.python.dataframe_helpers import (
    find_nodes,
    group_key,
    infer_schema,
    normalize_rows,
    output_name,
    parse_sql_predicate,
    rows_as_dicts,
    sort_rows,
    tabular_input,
    window_values,
)
from sparkleframe.python.types import (
    BooleanType,
    ByteType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NullType,
    Row,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    parse_ddl_schema,
)

_PLAN_IDS = itertools.count(1)

_JOIN_TYPES = {
    "inner": "inner",
    "cross": "inner",
    "left": "left",
    "leftouter": "left",
    "left_outer": "left",
    "right": "right",
    "rightouter": "right",
    "right_outer": "right",
    "outer": "full",
    "full": "full",
    "fullouter": "full",
    "full_outer": "full",
    "semi": "left_semi",
    "leftsemi": "left_semi",
    "left_semi": "left_semi",
    "anti": "left_anti",
    "leftanti": "left_anti",
    "left_anti": "left_anti",
}


class DataFrame(BaseDataFrame):
    def __init__(self, data: Any, schema: Optional[Any] = None, _plan_sets: Optional[List[frozenset]] = None) -> None:
        super().__init__(df=None)
        rows, schema = tabular_input(
            data, parse_ddl_schema(schema, pyspark_colons=True) if isinstance(schema, str) else schema
        )
        if isinstance(schema, DataType) and not isinstance(schema, StructType):
            # A single type names one ``value`` column; each input item is that column's value.
            rows, schema = [(value,) for value in rows], StructType([StructField("value", schema, True)])
        if schema is not None and not isinstance(schema, StructType):
            schema = infer_schema(rows, list(schema))
        elif schema is None and rows:
            schema = infer_schema(rows, None)
        if isinstance(schema, StructType) and _plan_sets is None:
            rows = normalize_rows(rows, schema)  # user input; derived frames already hold normalized tuples
        self._rows: List[tuple] = rows
        self._schema = schema
        self._plan_id = next(_PLAN_IDS)
        width = len(schema.fields) if isinstance(schema, StructType) else 0
        base = _plan_sets or [frozenset()] * width
        self._plan_sets: List[frozenset] = [s | {self._plan_id} for s in base]

    # --- metadata ---------------------------------------------------------------
    @property
    def schema(self) -> StructType:
        return self._schema

    @property
    def columns(self) -> List[str]:
        return [f.name for f in self._schema.fields]

    @property
    def dtypes(self) -> List[Tuple[str, str]]:
        return [(f.name, f.dataType.simpleString()) for f in self._schema.fields]

    def to_records(self) -> list:
        """Engine → ``list[dict]`` for the parity oracle: zip schema names with each row tuple."""
        return rows_as_dicts(self._rows, self._schema)

    def __getitem__(self, item: Union[int, str, Column, List, Tuple]) -> Union[Column, "DataFrame"]:
        if isinstance(item, str):
            column = Column(AttributeReference(item, self._plan_id))
            analyze(column._expr, self._schema, self._plan_sets)  # Spark resolves df["x"] eagerly
            return column
        if isinstance(item, int):
            return self[self._schema.fields[item].name]
        if isinstance(item, Column):
            return self.filter(item)
        if isinstance(item, (list, tuple)):
            return self.select(*item)
        raise TypeError(f"[NOT_COLUMN_OR_INT_OR_LIST_OR_STR_OR_TUPLE] Unexpected key {item!r}.")

    def __getattr__(self, item: str) -> Column:
        if item.startswith("_"):
            raise AttributeError(item)
        if item not in self.columns:
            raise AttributeError(f"'DataFrame' object has no attribute '{item}'")
        return self[item]

    def _derive(self, rows: List[tuple], fields: List[StructField], plan_sets: List[frozenset]) -> "DataFrame":
        return DataFrame(rows, StructType(fields), _plan_sets=plan_sets)

    def _bound(self, index: int) -> AttributeReference:
        field = self._schema.fields[index]
        ref = AttributeReference(field.name)
        ref.index = index
        ref.data_type = field.dataType
        return ref

    def _resolve(self, expr: Expression) -> Expression:
        return analyze(expr, self._schema, self._plan_sets)

    # --- projection -------------------------------------------------------------
    def select(self, *cols: Any) -> "DataFrame":
        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = tuple(cols[0])
        items: List[Expression] = []
        for col in cols:
            if isinstance(col, str):
                import sparkleframe.python.functions as F

                col = F.col(col)
            if not isinstance(col, Column):
                raise TypeError(
                    f"[NOT_LIST_OF_COLUMN_OR_STR] Argument `cols` should be a list of Column or str, got "
                    f"{type(col).__name__}."
                )
            if isinstance(col._expr, Star):
                items.extend(self._bound(i) for i in self._star_indices(col._expr.qualifier))
            else:
                items.append(col._expr)
        return self._project(items)

    def _star_indices(self, qualifier: Optional[str]) -> List[int]:
        indices = [i for i, s in enumerate(self._plan_sets) if qualifier is None or qualifier in s]
        if qualifier is not None and not indices:
            raise AnalysisException(
                f"[CANNOT_RESOLVE_STAR_EXPAND] Cannot resolve `{qualifier}`.* given input columns "
                f"{', '.join(f'`{name}`' for name in self.columns)}."
            )
        return indices

    def _project(self, items: List[Expression], names: Optional[List[Optional[str]]] = None) -> "DataFrame":
        """Evaluate ``items`` per row into a new frame (``names`` overrides output names)."""
        resolved = [self._resolve(item) for item in items]
        if any(_aggregates_in(r) for r in resolved):
            return self._aggregate([], [Column(item) for item in items])
        names = names or [None] * len(items)
        ctx = EvalContext()
        self._precompute_windows(resolved, ctx)

        generator_at = [i for i, r in enumerate(resolved) if _generator_of(r) is not None]
        if len(generator_at) > 1:
            raise AnalysisException("[UNSUPPORTED_GENERATOR.MULTI_GENERATOR] Only one generator allowed per select.")

        fields: List[StructField] = []
        plan_sets: List[frozenset] = []
        for position, (item, expr) in enumerate(zip(items, resolved)):
            if position in generator_at:
                generator = _generator_of(expr)
                generated_fields = generator.data_type.fields
                alias = names[position] or (expr.name if isinstance(expr, Alias) else None)
                if alias is not None and len(generated_fields) != 1:
                    raise AnalysisException(
                        f"[UDTF_ALIAS_NUMBER_MISMATCH] The number of aliases supplied in the AS clause does not "
                        f"match the number of columns output by the UDTF. Expected {len(generated_fields)} aliases, "
                        f"but got {alias}."
                    )
                for field in generated_fields:
                    fields.append(StructField(alias or field.name, field.dataType, field.nullable))
                    plan_sets.append(frozenset())
                continue
            fields.append(StructField(names[position] or output_name(item), expr.data_type))
            plan_sets.append(
                self._plan_sets[expr.index]
                if isinstance(expr, AttributeReference) and expr.index is not None
                else frozenset()
            )

        out_rows: List[tuple] = []
        for index, row in enumerate(self._rows):
            ctx.row_index = index
            values: List[Any] = []
            generated: List[tuple] = [()]
            for position, expr in enumerate(resolved):
                if position in generator_at:
                    generator = _generator_of(expr)
                    generated = explode_rows(eval_row(generator.children[0], row, ctx))
                    values.append(_GENERATED)
                else:
                    values.append(eval_row(expr, row, ctx))
            for produced in generated:
                out: List[Any] = []
                for value in values:
                    if value is _GENERATED:
                        out.extend(produced)
                    else:
                        out.append(value)
                out_rows.append(tuple(out))
        return self._derive(out_rows, fields, plan_sets)

    def _precompute_windows(self, resolved: List[Expression], ctx: EvalContext) -> None:
        for expr in resolved:
            for window in find_nodes(expr, lambda n: isinstance(n, WindowExpression)):
                ctx.window_values[id(window)] = window_values(window, self._rows, ctx)

    def _replace_columns(self, replacements: Dict[str, Column]) -> "DataFrame":
        """``withColumn(s)``: replace same-named columns in place, append the rest (all resolved on ``self``)."""
        for name, col in replacements.items():
            if not isinstance(col, Column):
                raise TypeError(f"[NOT_COLUMN] Argument `col` should be a Column, got {type(col).__name__}.")
        if _uses_lateral_alias(replacements, {f.name.lower() for f in self._schema.fields}):
            return self._replace_columns_lateral(replacements)
        lowered = {name.lower(): name for name in replacements}
        items: List[Expression] = []
        names: List[Optional[str]] = []
        used = set()
        for index, field in enumerate(self._schema.fields):
            key = field.name.lower()
            if key in lowered:
                name = lowered[key]
                items.append(replacements[name]._expr)
                names.append(name)
                used.add(name)
            else:
                items.append(self._bound(index))
                names.append(field.name)
        for name, col in replacements.items():
            if name not in used:
                items.append(col._expr)
                names.append(name)
        return self._project(items, names)

    def _replace_columns_lateral(self, replacements: Dict[str, Column]) -> "DataFrame":
        """Spark's lateral column alias in ``withColumns``: a name absent from the input binds to an earlier entry.

        New columns are appended one stage at a time so each can see the ones before it; names already in the
        input keep meaning the input column, so replacements are computed last, against the widened frame.
        """
        input_width = len(self._schema.fields)
        existing = {f.name.lower() for f in self._schema.fields}
        frame = self
        for name, col in replacements.items():
            if name.lower() not in existing:
                width = len(frame._schema.fields)
                items = [frame._bound(i) for i in range(width)] + [col._expr]
                frame = frame._project(items, frame.columns + [name])
        lowered = {name.lower(): name for name in replacements if name.lower() in existing}
        items = []
        names = []
        for index, field in enumerate(frame._schema.fields):
            name = lowered.get(field.name.lower()) if index < input_width else None
            items.append(replacements[name]._expr if name else frame._bound(index))
            names.append(name or field.name)
        return frame._project(items, names)

    def withColumn(self, colName: str, col: Column) -> "DataFrame":
        return self._replace_columns({colName: col})

    def withColumns(self, *colsMap: Dict[str, Column]) -> "DataFrame":
        if len(colsMap) != 1 or not isinstance(colsMap[0], dict):
            raise TypeError("[NOT_DICT] Argument `colsMap` should be a dict.")
        return self._replace_columns(dict(colsMap[0]))

    def withColumnRenamed(self, existing: str, new: str) -> "DataFrame":
        fields = [
            StructField(new if f.name.lower() == existing.lower() else f.name, f.dataType, f.nullable)
            for f in self._schema.fields
        ]
        return self._derive(list(self._rows), fields, list(self._plan_sets))

    def drop(self, *cols: Union[str, Column]) -> "DataFrame":
        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = tuple(cols[0])
        dropped = set()
        for col in cols:
            if isinstance(col, str):
                dropped.update(i for i, f in enumerate(self._schema.fields) if f.name.lower() == col.lower())
            elif isinstance(col, Column) and isinstance(col._expr, AttributeReference):
                try:
                    resolved = self._resolve(col._expr)
                except AnalysisException:
                    continue
                if isinstance(resolved, AttributeReference):
                    dropped.add(resolved.index)
            elif not isinstance(col, Column):
                raise TypeError(
                    f"[NOT_COLUMN_OR_STR] Argument `col` should be a Column or str, got {type(col).__name__}."
                )
        keep = [i for i in range(len(self._schema.fields)) if i not in dropped]
        return self._derive(
            [tuple(row[i] for i in keep) for row in self._rows],
            [self._schema.fields[i] for i in keep],
            [self._plan_sets[i] for i in keep],
        )

    # --- filtering / dedup ---------------------------------------------------------
    def filter(self, condition: Union[str, Column]) -> "DataFrame":
        if isinstance(condition, str):
            expr = parse_sql_predicate(condition)
        elif isinstance(condition, Column):
            expr = condition._expr
        else:
            raise TypeError(
                f"[NOT_COLUMN_OR_STR] Argument `condition` should be a Column or str, got "
                f"{type(condition).__name__}."
            )
        resolved = self._resolve(expr)
        if not isinstance(resolved.data_type, (BooleanType, NullType)):
            raise AnalysisException(
                f'[DATATYPE_MISMATCH.FILTER_NOT_BOOLEAN] Cannot resolve "{output_name(expr)}" due to data type '
                f'mismatch: Filter expression "{output_name(expr)}" of type '
                f'"{resolved.data_type.simpleString().upper()}" is not a boolean.'
            )
        ctx = EvalContext()
        self._precompute_windows([resolved], ctx)
        kept = []
        for index, row in enumerate(self._rows):
            ctx.row_index = index
            if eval_row(resolved, row, ctx) is True:
                kept.append(row)
        return self._derive(kept, list(self._schema.fields), list(self._plan_sets))

    where = filter

    def distinct(self) -> "DataFrame":
        return self.dropDuplicates()

    def dropDuplicates(self, subset: Optional[List[str]] = None) -> "DataFrame":
        if subset is None:
            positions = list(range(len(self._schema.fields)))
        else:
            positions = [self._resolve(AttributeReference(name)).index for name in subset]
        for i in positions:
            if coercion.contains_map(self._schema.fields[i].dataType):
                field = self._schema.fields[i]
                raise AnalysisException(
                    f"[UNSUPPORTED_FEATURE.SET_OPERATION_ON_MAP_TYPE] The feature is not supported: Cannot have MAP "
                    f"type columns in DataFrame which calls set operations (INTERSECT, EXCEPT, etc.), but the type "
                    f'of column `{field.name}` is "{field.dataType.simpleString().upper()}".'
                )
        seen = set()
        kept = []
        for row in self._rows:
            key = group_key([row[i] for i in positions])
            if key not in seen:
                seen.add(key)
                kept.append(row)
        return self._derive(kept, list(self._schema.fields), list(self._plan_sets))

    drop_duplicates = dropDuplicates

    def limit(self, num: int) -> "DataFrame":
        return self._derive(self._rows[:num], list(self._schema.fields), list(self._plan_sets))

    # --- set operations ----------------------------------------------------------------
    def union(self, other: "DataFrame") -> "DataFrame":
        if len(self._schema.fields) != len(other._schema.fields):
            raise AnalysisException(
                f"[NUM_COLUMNS_MISMATCH] UNION can only be performed on inputs with the same number of columns, but "
                f"the first input has {len(self._schema.fields)} columns and the second input has "
                f"{len(other._schema.fields)} columns."
            )
        fields = []
        for left, right in zip(self._schema.fields, other._schema.fields):
            wider = coercion.find_wider_type(left.dataType, right.dataType)
            if wider is None:
                raise AnalysisException(
                    f"[INCOMPATIBLE_COLUMN_TYPE] UNION can only be performed on tables with compatible column types. "
                    f'The column `{left.name}` has "{right.dataType.simpleString().upper()}" which is not compatible '
                    f'with "{left.dataType.simpleString().upper()}".'
                )
            fields.append(StructField(left.name, wider, left.nullable or right.nullable))
        rows = [_cast_row(row, self._schema.fields, fields) for row in self._rows]
        rows += [_cast_row(row, other._schema.fields, fields) for row in other._rows]
        return self._derive(rows, fields, [frozenset()] * len(fields))

    unionAll = union

    def unionByName(self, other: "DataFrame", allowMissingColumns: bool = False) -> "DataFrame":
        other_index = {f.name.lower(): i for i, f in enumerate(other._schema.fields)}
        own_names = {f.name.lower() for f in self._schema.fields}
        missing = [f.name for f in self._schema.fields if f.name.lower() not in other_index]
        extra = [f for f in other._schema.fields if f.name.lower() not in own_names]
        if (missing or extra) and not allowMissingColumns:
            name = (missing or [extra[0].name])[0]
            raise AnalysisException(
                f"[NUM_COLUMNS_MISMATCH] UNION can only be performed on inputs with the same number of columns; "
                f"cannot resolve column `{name}`."
            )
        null_field = StructField("", NullType())
        right_fields = [
            other._schema.fields[other_index[f.name.lower()]] if f.name.lower() in other_index else null_field
            for f in self._schema.fields
        ]
        left_fields = list(self._schema.fields) + [null_field] * len(extra)
        right_fields += list(extra)
        names = self.columns + [f.name for f in extra]
        fields = []
        for name, left, right in zip(names, left_fields, right_fields):
            wider = coercion.find_wider_type(left.dataType, right.dataType)
            if wider is None:
                raise AnalysisException(
                    f"[INCOMPATIBLE_COLUMN_TYPE] UNION can only be performed on tables with compatible column types; "
                    f"column `{name}`."
                )
            fields.append(StructField(name, wider, True))
        rows = [_cast_row(row + (None,) * len(extra), left_fields, fields) for row in self._rows]
        for row in other._rows:
            aligned = tuple(
                row[other_index[f.name.lower()]] if f.name.lower() in other_index else None
                for f in self._schema.fields
            ) + tuple(row[other_index[f.name.lower()]] for f in extra)
            rows.append(_cast_row(aligned, right_fields, fields))
        return self._derive(rows, fields, [frozenset()] * len(fields))

    # --- joins ----------------------------------------------------------------------------
    def join(self, other: "DataFrame", on: Any = None, how: Optional[str] = None) -> "DataFrame":
        join_type = _JOIN_TYPES.get((how or "inner").lower().replace(" ", ""))
        if join_type is None:
            raise ValueError(f"[UNSUPPORTED_JOIN_TYPE] Unsupported join type '{how}'.")
        if isinstance(on, str):
            on = [on]
        if on is None or (isinstance(on, list) and not on):
            return self._condition_join(other, None, join_type)
        if isinstance(on, Column):
            return self._condition_join(other, on._expr, join_type)
        if isinstance(on, (list, tuple)) and all(isinstance(c, Column) for c in on):
            condition = on[0]
            for extra in on[1:]:
                condition = condition & extra
            return self._condition_join(other, condition._expr, join_type)
        if isinstance(on, (list, tuple)) and all(isinstance(c, str) for c in on):
            return self._using_join(other, list(on), join_type)
        raise TypeError(f"[NOT_STR_OR_LIST_OF_STR_OR_COLUMN] Unsupported join condition {on!r}.")

    def _using_join(self, other: "DataFrame", keys: List[str], join_type: str) -> "DataFrame":
        left_idx = [_using_index(self, key, "left") for key in keys]
        right_idx = [_using_index(other, key, "right") for key in keys]
        key_types = []
        for li, ri in zip(left_idx, right_idx):
            lt, rt = self._schema.fields[li].dataType, other._schema.fields[ri].dataType
            common = coercion.find_wider_type(lt, rt)
            if common is None:
                _, lc, rc = coercion.coerce_comparison(lt, rt)
                common = lc or lt
            key_types.append(common)

        def join_key(row: tuple, idx: List[int], fields: List[StructField]) -> Optional[tuple]:
            values = [cast_value(row[i], t, True, fields[i].dataType) for i, t in zip(idx, key_types)]
            return None if any(v is None for v in values) else group_key(values)

        right_by_key: Dict[tuple, List[int]] = {}
        for position, row in enumerate(other._rows):
            key = join_key(row, right_idx, other._schema.fields)
            if key is not None:
                right_by_key.setdefault(key, []).append(position)

        left_rest = [i for i in range(len(self._schema.fields)) if i not in left_idx]
        right_rest = [i for i in range(len(other._schema.fields)) if i not in right_idx]
        key_source = "right" if join_type == "right" else "left"
        key_fields = []
        for li, ri, common in zip(left_idx, right_idx, key_types):
            source = other._schema.fields[ri] if key_source == "right" else self._schema.fields[li]
            dtype = common if join_type == "full" else source.dataType
            key_fields.append(StructField(source.name, dtype, True))
        key_sets = [self._plan_sets[li] | other._plan_sets[ri] for li, ri in zip(left_idx, right_idx)]

        if join_type in ("left_semi", "left_anti"):
            fields = key_fields + [self._schema.fields[i] for i in left_rest]
            rows = []
            for row in self._rows:
                key = join_key(row, left_idx, self._schema.fields)
                matched = key is not None and key in right_by_key
                if matched == (join_type == "left_semi"):
                    rows.append(tuple(row[i] for i in left_idx) + tuple(row[i] for i in left_rest))
            return self._derive(rows, fields, key_sets + [self._plan_sets[i] for i in left_rest])

        fields = (
            key_fields + [self._schema.fields[i] for i in left_rest] + [other._schema.fields[i] for i in right_rest]
        )
        plan_sets = key_sets + [self._plan_sets[i] for i in left_rest] + [other._plan_sets[i] for i in right_rest]
        empty_left, empty_right = (None,) * len(self._schema.fields), (None,) * len(other._schema.fields)
        rows = []
        matched_right = set()
        for row in self._rows:
            key = join_key(row, left_idx, self._schema.fields)
            matches = right_by_key.get(key, []) if key is not None else []
            for position in matches:
                matched_right.add(position)
                rows.append(
                    _using_row(
                        row,
                        other._rows[position],
                        left_idx,
                        right_idx,
                        left_rest,
                        right_rest,
                        key_source,
                        join_type,
                        key_types,
                        self,
                        other,
                    )
                )
            if not matches and join_type in ("left", "full"):
                rows.append(
                    _using_row(
                        row,
                        empty_right,
                        left_idx,
                        right_idx,
                        left_rest,
                        right_rest,
                        "left",
                        join_type,
                        key_types,
                        self,
                        other,
                    )
                )
        if join_type in ("right", "full"):
            for position, row in enumerate(other._rows):
                if position not in matched_right:
                    rows.append(
                        _using_row(
                            empty_left,
                            row,
                            left_idx,
                            right_idx,
                            left_rest,
                            right_rest,
                            "right",
                            join_type,
                            key_types,
                            self,
                            other,
                        )
                    )
        return self._derive(rows, fields, plan_sets)

    def _condition_join(self, other: "DataFrame", condition: Optional[Expression], join_type: str) -> "DataFrame":
        fields = list(self._schema.fields) + list(other._schema.fields)
        plan_sets = list(self._plan_sets) + list(other._plan_sets)
        resolved = None
        if condition is not None:
            resolved = analyze(condition, StructType(fields), plan_sets)
            if not isinstance(resolved.data_type, (BooleanType, NullType)):
                raise AnalysisException("[JOIN_CONDITION_IS_NOT_BOOLEAN_TYPE] The join condition must be a boolean.")
        ctx = EvalContext()
        rows = []
        matched_right = set()
        empty_left, empty_right = (None,) * len(self._schema.fields), (None,) * len(other._schema.fields)
        for left in self._rows:
            matched = False
            for position, right in enumerate(other._rows):
                if resolved is None or eval_row(resolved, left + right, ctx) is True:
                    matched = True
                    matched_right.add(position)
                    if join_type in ("left_semi", "left_anti"):
                        break
                    rows.append(left + right)
            if join_type == "left_semi" and matched:
                rows.append(left)
            elif join_type == "left_anti" and not matched:
                rows.append(left)
            elif not matched and join_type in ("left", "full"):
                rows.append(left + empty_right)
        if join_type in ("right", "full"):
            rows.extend(empty_left + right for p, right in enumerate(other._rows) if p not in matched_right)
        if join_type in ("left_semi", "left_anti"):
            return self._derive(rows, list(self._schema.fields), list(self._plan_sets))
        return self._derive(rows, fields, plan_sets)

    def crossJoin(self, other: "DataFrame") -> "DataFrame":
        return self._condition_join(other, None, "inner")

    # --- aggregation ---------------------------------------------------------------------------
    def groupBy(self, *cols: Union[str, Column]) -> Any:
        from sparkleframe.python.group import GroupedData

        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = tuple(cols[0])
        return GroupedData(self, list(cols))

    groupby = groupBy

    def agg(self, *exprs: Any) -> "DataFrame":
        return self.groupBy().agg(*exprs)

    def _aggregate(self, group_cols: List[Union[str, Column]], exprs: List[Column]) -> "DataFrame":
        group_items = [_column_ref(c) for c in group_cols]
        agg_items = [c._expr if isinstance(c, Column) else _column_ref(c) for c in exprs]
        group_resolved = [self._resolve(g) for g in group_items]
        agg_resolved = [self._resolve(a) for a in agg_items]

        groups: Dict[tuple, List[tuple]] = {}
        ctx = EvalContext()
        for row in self._rows:
            key = group_key([eval_row(g, row, ctx) for g in group_resolved])
            groups.setdefault(key, []).append(row)
        if not group_items and not groups:
            groups[()] = []

        aggregates = [
            node
            for expr in agg_resolved
            for node in find_nodes(expr, lambda n: isinstance(n, FunctionCall) and lookup(n.name).kind == "aggregate")
        ]
        fields = [StructField(output_name(item), r.data_type) for item, r in zip(group_items, group_resolved)]
        fields += [StructField(output_name(item), r.data_type) for item, r in zip(agg_items, agg_resolved)]
        plan_sets = [
            self._plan_sets[r.index] if isinstance(r, AttributeReference) else frozenset() for r in group_resolved
        ] + [frozenset()] * len(agg_resolved)

        out_rows = []
        empty_row = (None,) * len(self._schema.fields)
        for rows in groups.values():
            representative = rows[0] if rows else empty_row
            for node in aggregates:
                arg_rows = [tuple(eval_row(arg, row, ctx) for arg in node.children) for row in rows]
                ctx.agg_values[id(node)] = lookup(node.name).aggregate(arg_rows, node)
            values = [eval_row(g, representative, ctx) for g in group_resolved]
            values += [eval_row(a, representative, ctx) for a in agg_resolved]
            out_rows.append(tuple(values))
        return self._derive(out_rows, fields, plan_sets)

    # --- ordering ---------------------------------------------------------------------------------
    def orderBy(self, *cols: Any, ascending: Union[bool, List[bool], None] = None) -> "DataFrame":
        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
            cols = tuple(cols[0])
        flags = ascending if isinstance(ascending, list) else [ascending] * len(cols)
        orders = []
        for col, flag in zip(cols, flags):
            if isinstance(col, int) and not isinstance(col, bool):
                expr = self._ordinal(col)
            else:
                expr = _column_ref(col)
            if not isinstance(expr, SortOrder):
                asc = True if flag is None else bool(flag)
                expr = SortOrder(expr, ascending=asc, nulls_first=asc)
            orders.append(self._resolve(expr))
        rows = sort_rows(list(self._rows), orders, EvalContext())
        return self._derive(rows, list(self._schema.fields), list(self._plan_sets))

    sort = orderBy

    def _ordinal(self, ordinal: int) -> Expression:
        """PySpark's ``orderBy(int)``: a 1-based column ordinal; negative sorts that column descending."""
        if ordinal == 0:
            raise IndexError("[ZERO_INDEX] Index must be non-zero.")
        if abs(ordinal) > len(self._schema.fields):
            raise IndexError(f"[INDEX_OUT_OF_RANGE] Column ordinal {ordinal} is out of range.")
        bound = self._bound(abs(ordinal) - 1)
        return bound if ordinal > 0 else SortOrder(bound, ascending=False, nulls_first=False)

    # --- actions ------------------------------------------------------------------------------------
    def count(self) -> int:
        return len(self._rows)

    def isEmpty(self) -> bool:
        return not self._rows

    def collect(self) -> List[Row]:
        names = self.columns
        return [Row.from_fields(names, list(row)) for row in self._rows]

    def first(self) -> Optional[Row]:
        rows = self.collect()
        return rows[0] if rows else None

    def head(self, n: Optional[int] = None) -> Any:
        if n is None:
            return self.first()
        return self.collect()[:n]

    def take(self, num: int) -> List[Row]:
        return self.collect()[:num]

    def cache(self) -> "DataFrame":
        return self

    persist = cache

    def show(self, n: int = 20, truncate: bool = True, vertical: bool = False) -> None:
        import pandas as pd

        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(self.toPandas().head(n))

    def toPandas(self) -> Any:
        """A pandas frame with PySpark's ``toPandas`` dtypes (timestamps as ``datetime64``, nulls as NaN/None)."""
        import pandas as pd

        columns = {}
        for index, field in enumerate(self._schema.fields):
            values = [row[index] for row in self._rows]
            columns[index] = _pandas_series(values, field.dataType)
        frame = pd.DataFrame(columns, index=range(len(self._rows)))
        frame.columns = self.columns
        return frame

    def fillna(self, value: Union[Any, dict], subset: Union[str, List[str], None] = None) -> "DataFrame":
        if isinstance(value, dict):
            replacements = dict(value)
        else:
            names = [subset] if isinstance(subset, str) else subset or self.columns
            replacements = {name: value for name in names if _fillable(self._schema[name].dataType, value)}
        items: List[Expression] = []
        for index, field in enumerate(self._schema.fields):
            if field.name in replacements:
                import sparkleframe.python.functions as F

                fill = F.lit(replacements[field.name]).cast(field.dataType)
                items.append(F.coalesce(Column(self._bound(index)), fill)._expr)
            else:
                items.append(self._bound(index))
        return self._project(items, self.columns)

    def alias(self, alias: str) -> "DataFrame":
        """Name the frame so ``alias.column`` and ``alias.*`` reference its columns (an earlier alias is hidden)."""
        plan_sets = [frozenset(p for p in s if not isinstance(p, str)) | {alias} for s in self._plan_sets]
        return self._derive(list(self._rows), list(self._schema.fields), plan_sets)

    def to_arrow(self) -> Any:
        not_implemented_yet("DataFrame.to_arrow")


class _Generated:
    """Placeholder for the generator column while assembling ``explode`` output rows."""


_GENERATED = _Generated()


def _aggregates_in(expr: Expression) -> bool:
    return bool(
        find_nodes(
            expr,
            lambda n: isinstance(n, FunctionCall) and lookup(n.name).kind == "aggregate",
            stop_at=(WindowExpression,),
        )
    )


def _generator_of(expr: Expression) -> Optional[FunctionCall]:
    node = expr.child if isinstance(expr, Alias) else expr
    if isinstance(node, FunctionCall) and lookup(node.name).kind == "generator":
        return node
    return None


def _uses_lateral_alias(replacements: Dict[str, Column], existing: set) -> bool:
    """Whether an entry references a column that only an earlier entry of the same ``withColumns`` defines."""
    defined: set = set()
    for name, col in replacements.items():
        refs = find_nodes(col._expr, lambda n: isinstance(n, AttributeReference) and n.plan_id is None)
        for ref in refs:
            base = ref.name.split(".", 1)[0].strip("`").lower()
            if base in defined and base not in existing:
                return True
        defined.add(name.lower())
    return False


def _cast_row(row: tuple, source: Sequence[StructField], target: Sequence[StructField]) -> tuple:
    return tuple(
        value if s.dataType == t.dataType else cast_value(value, t.dataType, True, s.dataType)
        for value, s, t in zip(row, source, target)
    )


def _using_index(df: DataFrame, key: str, side: str) -> int:
    matches = [i for i, f in enumerate(df._schema.fields) if f.name == key] or [
        i for i, f in enumerate(df._schema.fields) if f.name.lower() == key.lower()
    ]
    if len(matches) != 1:
        raise AnalysisException(
            f"[UNRESOLVED_USING_COLUMN_FOR_JOIN] USING column `{key}` cannot be resolved on the {side} side of the "
            f"join. The {side}-side columns: [{', '.join(f'`{c}`' for c in df.columns)}]."
        )
    return matches[0]


def _using_row(
    left: tuple,
    right: tuple,
    left_idx: List[int],
    right_idx: List[int],
    left_rest: List[int],
    right_rest: List[int],
    key_source: str,
    join_type: str,
    key_types: List[Any],
    left_df: DataFrame,
    right_df: DataFrame,
) -> tuple:
    keys = []
    for li, ri, common in zip(left_idx, right_idx, key_types):
        if join_type == "full":
            value = left[li] if left[li] is not None else right[ri]
            source = left_df._schema.fields[li] if left[li] is not None else right_df._schema.fields[ri]
            keys.append(cast_value(value, common, True, source.dataType) if source.dataType != common else value)
        else:
            keys.append(right[ri] if key_source == "right" else left[li])
    return tuple(keys) + tuple(left[i] for i in left_rest) + tuple(right[i] for i in right_rest)


def _fillable(dtype: Any, value: Any) -> bool:
    if isinstance(value, bool):
        return isinstance(dtype, BooleanType)
    if isinstance(value, (int, float)):
        return isinstance(dtype, (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType))
    if isinstance(value, str):
        return isinstance(dtype, StringType)
    return False


def _pandas_series(values: List[Any], dtype: Any) -> Any:
    import pandas as pd

    if isinstance(dtype, TimestampType):
        return pd.to_datetime(pd.Series(values, dtype=object)).astype("datetime64[ns]")
    if isinstance(dtype, StringType):
        # PySpark builds the frame from records, so pandas infers the dtype (``str`` with NaN on pandas 3).
        return pd.Series(values) if values else pd.Series(values, dtype=object)
    if isinstance(dtype, (FloatType, DoubleType)):
        return pd.Series([float("nan") if v is None else v for v in values], dtype="float64")
    if isinstance(dtype, (ByteType, ShortType, IntegerType, LongType)):
        if any(v is None for v in values):
            return pd.Series([float("nan") if v is None else v for v in values], dtype="float64")
        width = {ByteType: "int8", ShortType: "int16", IntegerType: "int32", LongType: "int64"}[type(dtype)]
        return pd.Series(values, dtype=width)
    if isinstance(dtype, BooleanType) and all(v is not None for v in values):
        return pd.Series(values, dtype="bool")
    if isinstance(dtype, DateType):
        return pd.Series(values, dtype=object)
    return pd.Series(values, dtype=object)
