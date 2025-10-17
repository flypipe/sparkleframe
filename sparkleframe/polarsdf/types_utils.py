from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Any, Dict
from sparkleframe.polarsdf import types as sft
import polars as pl

from sparkleframe.polarsdf.types import StructType


@dataclass
class _KeyUnion:
    """Union of keys for a (possibly nested) MapType column."""

    keys: List[str] = field(default_factory=list)  # ordered list of keys at this node
    _seen: set = field(default_factory=set, repr=False)  # for order-preserving uniqueness
    children: Dict[str, "_KeyUnion"] = field(default_factory=dict)  # per-key subtree (for nested maps)

    def add_key(self, k: str) -> None:
        if k not in self._seen:
            self._seen.add(k)
            self.keys.append(k)

    def child_for(self, k: str) -> "_KeyUnion":
        if k not in self.children:
            self.children[k] = _KeyUnion()
        return self.children[k]


class _MapTypeUtils:

    @staticmethod
    def is_map_dtype(dtype: pl.DataType) -> bool:
        """
        Recognize your logical MapType's native Polars layout:
        List(Struct([Field('key', ...), Field('value', ...)]))
        """
        if isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct):
            fields = dtype.inner.fields
            return len(fields) == 2 and fields[0].name == "key" and fields[1].name == "value"
        return False

    @staticmethod
    def infer_map_keys(df: pl.DataFrame, col: str) -> List[str]:
        if col not in df.columns:
            raise pl.ColumnNotFoundError(col)

        dt = df.schema[col]
        if not _MapTypeUtils.is_map_dtype(dt):
            raise TypeError(f"Column '{col}' is not a map-like List[Struct[key,value]]; got {dt}")

        if df.height == 0:
            return []

        keys_series = (
            df.select(pl.col(col).list.explode().alias("_kv"))
            .select(pl.col("_kv").struct.field("key").alias("_k"))
            .to_series()
        )
        seen, keys = set(), []
        for k in keys_series:
            if k is not None and k not in seen:
                seen.add(k)
                keys.append(k)
        return keys

    @staticmethod
    def map_to_struct(df: pl.DataFrame, col: str, *, keys: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Materialize a Struct column from a map-like column so that dot access works.
        Overwrites the same column by default.
        """
        if col not in df.columns:
            raise pl.ColumnNotFoundError(col)

        # Determine the field keys (infer if not provided)
        keys = keys or _MapTypeUtils.infer_map_keys(df, col)
        if not keys:
            # create an empty struct if no keys are present
            df = df.with_columns(pl.struct([]).alias(col))
            return df

        field_exprs = []
        for k in keys:
            val_expr = (
                pl.col(col)
                .list.eval(
                    pl.when(pl.element().struct.field("key") == pl.lit(k)).then(pl.element().struct.field("value"))
                )
                .list.drop_nulls()
                .list.first()
                .alias(k)
            )
            field_exprs.append(val_expr)

        struct_expr = pl.struct(field_exprs).alias(col)
        df = df.with_columns(struct_expr)
        return df

    @staticmethod
    def collect_map_keys_for_fields(rows: Iterable[Any], schema: StructType) -> Dict[str, _KeyUnion]:
        roots: Dict[str, _KeyUnion] = {}

        # --- discover all MapType paths in the declared schema ---
        def iter_map_paths(prefix: str, dt: sft.DataType):
            if isinstance(dt, sft.MapType):
                yield prefix
            elif isinstance(dt, sft.ArrayType) and isinstance(dt.elementType, sft.MapType):
                # treat array<map> as a "map-like" node for key union
                yield prefix
            elif isinstance(dt, sft.StructType):
                for sf in dt.fields:
                    child_prefix = f"{prefix}.{sf.name}" if prefix else sf.name
                    yield from iter_map_paths(child_prefix, sf.dataType)

        map_paths = list(iter_map_paths("", schema))
        for p in map_paths:
            roots[p] = _KeyUnion()
        if not roots:
            return roots

        # Build top-level field index map once (for tuple/list rows)
        top_index = {f.name: i for i, f in enumerate(schema.fields)}

        def as_dict_or_none(obj):
            if obj is None:
                return None
            if isinstance(obj, list):
                try:
                    return {kv.get("key"): kv.get("value") for kv in obj}
                except Exception:
                    return None
            if isinstance(obj, dict):
                return obj
            return None

        def get_nested_cell(row_obj, path: str):
            """Fetch the value at a schema path from a row (dict or tuple/list)."""
            if path == "":
                return row_obj

            parts = path.split(".")
            first = parts[0]

            # Get top-level object for `first`
            if isinstance(row_obj, dict):
                cur = row_obj.get(first, None)
            elif isinstance(row_obj, (list, tuple)):
                # map top-level schema field name -> index (precomputed as top_index)
                idx = top_index.get(first, None)
                if idx is None or idx >= len(row_obj):
                    return None
                cur = row_obj[idx]
                # IMPORTANT:
                # If the path is just "first" and the tuple element is a dict, we have two valid shapes:
                #   1) row[idx] == {first: {...}}  -> unwrap to the inner value
                #   2) row[idx] == {...}           -> it's already the inner map; keep as-is
                if len(parts) == 1 and isinstance(cur, dict):
                    if first in cur:
                        cur = cur[first]
                    # else: keep cur unchanged
            else:
                return None

            # Traverse remaining segments inside dicts
            for seg in parts[1:]:
                if cur is None:
                    return None
                if isinstance(cur, dict):
                    cur = cur.get(seg, None)
                else:
                    return None
            return cur

        def walk(node: _KeyUnion, map_dtype: sft.MapType, value) -> None:
            d = as_dict_or_none(value)
            if d is None:
                return
            for k, v in d.items():
                node.add_key(k)
                if isinstance(map_dtype.valueType, sft.MapType):
                    child = node.child_for(k)
                    walk(child, map_dtype.valueType, v)

        # Collect unions per map path
        for row in rows:
            for path in map_paths:
                node = roots[path]
                # resolve dtype at path
                dt = schema
                for seg in path.split("."):
                    if seg == "":
                        continue
                    dt = next(f.dataType for f in dt.fields if f.name == seg) if isinstance(dt, sft.StructType) else dt
                if not isinstance(dt, sft.MapType):
                    continue

                cell = get_nested_cell(row, path)
                if isinstance(dt, sft.MapType):
                    walk(node, dt, cell)
                elif isinstance(dt, sft.ArrayType) and isinstance(dt.elementType, sft.MapType):
                    if isinstance(cell, list):
                        for elem in cell:
                            walk(node, dt.elementType, elem)

        return roots

    # types_utils.py

    @classmethod
    def build_df_from_struct_rows(cls, rows: Iterable[Any], schema: StructType) -> pl.DataFrame:
        roots_by_path = cls.collect_map_keys_for_fields(rows, schema)

        # -------- helpers --------
        def _top_index():
            return {f.name: i for i, f in enumerate(schema.fields)}

        top_idx = _top_index()

        def _resolve_dtype_at_path(path: str, dt=None):
            dt = schema if dt is None else dt
            if path == "":
                return dt
            for seg in path.split("."):
                if seg == "":
                    continue
                if isinstance(dt, sft.StructType):
                    dt = next(sf.dataType for sf in dt.fields if sf.name == seg)
                else:
                    break
            return dt

        def _get_cell(row_obj, path: str):
            """Fetch value at a schema path from a row (dict or tuple/list)."""
            if path == "":
                return row_obj
            parts = path.split(".")
            first = parts[0]

            if isinstance(row_obj, dict):
                cur = row_obj.get(first, None)
            elif isinstance(row_obj, (list, tuple)):
                idx = top_idx.get(first, None)
                if idx is None or idx >= len(row_obj):
                    return None
                cur = row_obj[idx]
                # support shape: ({col: ...},) and ({...},)
                if len(parts) == 1 and isinstance(cur, dict) and first in cur:
                    cur = cur[first]
            else:
                return None

            for seg in parts[1:]:
                if cur is None:
                    return None
                if isinstance(cur, dict):
                    cur = cur.get(seg, None)
                else:
                    return None
            return cur

        def _as_dict_or_none(obj):
            if obj is None:
                return None
            if isinstance(obj, list):
                # [{"key":k,"value":v}, ...] -> {k:v}
                try:
                    return {kv.get("key"): kv.get("value") for kv in obj}
                except Exception:
                    return None
            if isinstance(obj, dict):
                return obj
            return None

        # -------- backfill unions for array<map<…>> if empty --------
        for path, node in list(roots_by_path.items()):
            dt = _resolve_dtype_at_path(path)
            if isinstance(dt, sft.ArrayType) and isinstance(dt.elementType, sft.MapType) and not node.keys:
                seen = set()
                for row in rows:
                    cell = _get_cell(row, path)
                    if isinstance(cell, list):
                        for elem in cell:
                            d = elem if isinstance(elem, dict) else _as_dict_or_none(elem)
                            if isinstance(d, dict):
                                for k in d.keys():
                                    if k not in seen:
                                        node.add_key(k)
                                        seen.add(k)

        # -------- dtype construction --------
        def dtype_from_path(dt: sft.DataType, path: str, node: Optional["_KeyUnion"]) -> pl.DataType:
            # map<k,v> -> Struct(unioned keys)
            if isinstance(dt, sft.MapType):
                node = node if node is not None else roots_by_path.get(path)
                keys_here = [] if node is None else node.keys
                if isinstance(dt.valueType, sft.MapType):
                    fields = []
                    for k in keys_here:
                        child_node = None if node is None else node.children.get(k)
                        child_dtype = dtype_from_path(dt.valueType, f"{path}.{k}", child_node)
                        fields.append(pl.Field(k, child_dtype))
                    return pl.Struct(fields)
                else:
                    return pl.Struct([pl.Field(k, dt.valueType.to_native()) for k in keys_here])

            # array<map<k,v>> -> List(Struct(unioned keys))
            if isinstance(dt, sft.ArrayType) and isinstance(dt.elementType, sft.MapType):
                node = node if node is not None else roots_by_path.get(path)
                keys_here = [] if node is None else node.keys
                elem_fields = [pl.Field(k, dt.elementType.valueType.to_native()) for k in keys_here]
                return pl.List(pl.Struct(elem_fields))

            # generic arrays
            if isinstance(dt, sft.ArrayType):
                inner = dtype_from_path(dt.elementType, path, None)
                return pl.List(inner)

            # struct
            if isinstance(dt, sft.StructType):
                return pl.Struct(
                    [
                        pl.Field(
                            sf.name,
                            dtype_from_path(
                                sf.dataType,
                                f"{path}.{sf.name}" if path else sf.name,
                                roots_by_path.get(f"{path}.{sf.name}" if path else sf.name),
                            ),
                        )
                        for sf in dt.fields
                    ]
                )

            # leaf
            return dt.to_native()

        colnames = [f.name for f in schema.fields]
        # IMPORTANT: pass the node for each top-level field
        coltypes = [dtype_from_path(f.dataType, f.name, roots_by_path.get(f.name)) for f in schema.fields]

        # -------- value coercion --------
        def coerce_value(val, dt: sft.DataType, path: str, node: Optional["_KeyUnion"]):
            # map<k,v> -> dict with unioned keys
            if isinstance(dt, sft.MapType):
                node = node if node is not None else roots_by_path.get(path)
                keys_here = [] if node is None else node.keys
                d = _as_dict_or_none(val) or {}
                out = {}
                for k in keys_here:
                    if isinstance(dt.valueType, sft.MapType):
                        child_node = None if node is None else node.children.get(k)
                        out[k] = coerce_value(d.get(k, None), dt.valueType, f"{path}.{k}", child_node)
                    else:
                        out[k] = d.get(k, None)
                return out

            # array<map<k,v>> -> list of struct dicts with unioned keys
            if isinstance(dt, sft.ArrayType) and isinstance(dt.elementType, sft.MapType):
                node = node if node is not None else roots_by_path.get(path)
                keys_here = [] if node is None else node.keys
                if val is None:
                    return None
                items = val if isinstance(val, list) else [val]
                out_list = []
                for elem in items:
                    d = elem if isinstance(elem, dict) else _as_dict_or_none(elem) or {}
                    out_elem = {k: d.get(k, None) for k in keys_here}
                    out_list.append(out_elem)
                return out_list

            # generic array
            if isinstance(dt, sft.ArrayType):
                if val is None:
                    return None
                items = val if isinstance(val, list) else [val]
                return [coerce_value(elem, dt.elementType, path, None) for elem in items]

            # struct
            if isinstance(dt, sft.StructType):
                src = val if isinstance(val, dict) else {}
                return {
                    sf.name: coerce_value(
                        src.get(sf.name, None),
                        sf.dataType,
                        f"{path}.{sf.name}" if path else sf.name,
                        roots_by_path.get(f"{path}.{sf.name}" if path else sf.name),
                    )
                    for sf in dt.fields
                }

            # leaf
            return val

        cols = {name: [] for name in colnames}
        for row in rows:
            if isinstance(row, dict):
                for f in schema.fields:
                    cols[f.name].append(
                        coerce_value(row.get(f.name, None), f.dataType, f.name, roots_by_path.get(f.name))
                    )
            elif isinstance(row, (list, tuple)):
                if len(schema.fields) == 1 and len(row) == 1:
                    f = schema.fields[0]
                    raw = row[0]
                    if isinstance(raw, dict) and f.name in raw:
                        raw = raw[f.name]
                    cols[f.name].append(coerce_value(raw, f.dataType, f.name, roots_by_path.get(f.name)))
                else:
                    if len(row) != len(schema.fields):
                        raise ValueError(f"Row length {len(row)} does not match schema length {len(schema.fields)}")
                    for f, raw in zip(schema.fields, row):
                        cols[f.name].append(coerce_value(raw, f.dataType, f.name, roots_by_path.get(f.name)))
            else:
                if len(schema.fields) != 1:
                    raise TypeError(f"Scalar row provided but schema has {len(schema.fields)} fields")
                f = schema.fields[0]
                cols[f.name].append(coerce_value(row, f.dataType, f.name, roots_by_path.get(f.name)))

        schema_pl = list(zip(colnames, coltypes))
        return pl.DataFrame(cols, schema=schema_pl)

    @staticmethod
    def apply_schema_casts(df: pl.DataFrame, schema: StructType) -> pl.DataFrame:
        """
        Enforce the provided StructType's dtypes on self.df (post-construction).
        Cast only primitive leaf types; skip Struct, Map, and Array (already materialized correctly).
        """
        if not isinstance(schema, StructType):
            return df

        for field_ in schema.fields:
            name = field_.name
            if name not in df.columns:
                continue

            dt = field_.dataType

            # Skip complex or nested types (Struct, Map, Array)
            if isinstance(dt, (sft.StructType, sft.MapType, sft.ArrayType)):
                continue

            target = dt.to_native()
            current = df.schema[name]

            # Only cast if the type truly differs and both are scalar types
            if current != target:
                df = df.with_columns(pl.col(name).cast(target).alias(name))

        return df
