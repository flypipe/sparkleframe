import pytest
import polars as pl

from sparkleframe.polarsdf.types import StructType, StructField, StringType, IntegerType, MapType
from sparkleframe.polarsdf.types_utils import _MapTypeUtils


class TestMapTypeUtils:
    """Test suite for MapTypeUtils class that handles map-type data operations."""

    # ---------- is_map_dtype ----------
    def test_is_map_dtype_true(self):
        """Test valid map data type detection.
        Verifies that a List containing a Struct with 'key' and 'value' fields
        is correctly identified as a map data type."""
        dt = pl.List(pl.Struct([pl.Field("key", pl.Utf8), pl.Field("value", pl.Int32)]))
        assert _MapTypeUtils.is_map_dtype(dt) is True

    @pytest.mark.parametrize(
        "dtype",
        [
            pl.List(pl.Struct([pl.Field("k", pl.Utf8), pl.Field("value", pl.Int32)])),  # wrong field name
            pl.List(pl.Utf8),  # not struct in list
            pl.Struct([pl.Field("key", pl.Utf8), pl.Field("value", pl.Int32)]),  # not a list
            pl.List(pl.Struct([pl.Field("key", pl.Utf8)])),  # only one field
        ],
    )
    def test_is_map_dtype_false(self, dtype):
        """Test invalid map data type detection.
        Tests various invalid configurations that should not be considered map types."""
        assert _MapTypeUtils.is_map_dtype(dtype) is False

    # ---------- infer_map_keys ----------
    def test_infer_map_keys_happy_path_and_order_preserved(self):
        """Test key inference and order preservation in maps.
        Verifies that keys are extracted in first-appearance order from map data."""
        df = pl.DataFrame(
            {
                "col": [
                    [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
                    [{"key": "b", "value": 3}, {"key": "c", "value": 4}],
                ]
            }
        )
        keys = _MapTypeUtils.infer_map_keys(df, "col")
        assert keys == ["a", "b", "c"]

    def test_infer_map_keys_empty_df(self):
        """Test key inference from empty DataFrame.
        Ensures proper handling of empty DataFrames with correct map schema."""
        df = pl.DataFrame(
            {"col": pl.Series([], dtype=pl.List(pl.Struct([pl.Field("key", pl.Utf8), pl.Field("value", pl.Int32)])))}
        )
        assert _MapTypeUtils.infer_map_keys(df, "col") == []

    def test_infer_map_keys_errors(self):
        """Test error conditions in key inference.
        Verifies appropriate error handling for missing columns and invalid data types."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(pl.ColumnNotFoundError):
            _MapTypeUtils.infer_map_keys(df, "col")
        with pytest.raises(TypeError):
            _MapTypeUtils.infer_map_keys(pl.DataFrame({"col": [[1, 2], [3, 4]]}), "col")

    # ---------- map_to_struct ----------
    def test_map_to_struct_overwrite(self):
        """Test conversion from map to struct with column overwrite.
        Verifies that a map column can be correctly converted to a struct column
        while maintaining the original column name."""
        df = pl.DataFrame(
            {
                "col": [
                    [{"key": "id", "value": 1}, {"key": "m", "value": 10}],
                    [{"key": "id", "value": 2}, {"key": "m", "value": 20}],
                ]
            }
        )
        out = _MapTypeUtils.map_to_struct(df, "col")
        assert isinstance(out.schema["col"], pl.Struct)
        assert [f.name for f in out.schema["col"].fields] == ["id", "m"]
        assert out.select("col").to_dicts() == [{"col": {"id": 1, "m": 10}}, {"col": {"id": 2, "m": 20}}]

    # ---------- collect_map_keys_for_fields ----------
    @pytest.mark.parametrize(
        "rows",
        [
            [({"id": {"id2": 1, "m2": 1}},)],
            [({"id": {"id2": 5}},)],
        ],
    )
    def test_collect_map_keys_for_fields_nested(self, rows):
        """Test collection of keys from nested map structures.
        Validates proper key collection from multi-level nested maps."""
        schema = StructType([StructField("col", MapType(StringType(), MapType(StringType(), IntegerType())))])
        roots = _MapTypeUtils.collect_map_keys_for_fields(rows, schema)
        assert "col" in roots
        assert "id" in roots["col"].keys
        child = roots["col"].children.get("id")
        assert child is not None
        assert set(child.keys) >= {"id2"}

    # ---------- build_df_from_struct_rows ----------
    @pytest.mark.parametrize(
        "rows, schema, expected",
        [
            (
                [
                    ({"id": 1, "m": 1},),
                ],
                StructType([StructField("col", MapType(StringType(), IntegerType()))]),
                [{"col": {"id": 1, "m": 1}}],
            ),
            (
                [
                    ({"id": {"id2": 1, "m2": 1}},),
                ],
                StructType([StructField("col", MapType(StringType(), MapType(StringType(), IntegerType())))]),
                [{"col": {"id": {"id2": 1, "m2": 1}}}],
            ),
        ],
    )
    def test_build_df_from_struct_rows_tuple_rows(self, rows, schema, expected):
        """Test DataFrame construction from tuple rows.
        Verifies correct DataFrame creation from tuple-based row data with nested structures."""
        df = _MapTypeUtils.build_df_from_struct_rows(rows, schema)
        assert df.to_dicts() == expected
        assert isinstance(df.schema["col"], pl.Struct)

    def test_build_df_from_struct_rows_dict_rows(self):
        """Test DataFrame construction from dictionary rows.
        Validates DataFrame creation from dictionary-format input data."""
        rows = [{"col": {"id": 3, "m": 4}}]
        schema = StructType([StructField("col", MapType(StringType(), IntegerType()))])
        df = _MapTypeUtils.build_df_from_struct_rows(rows, schema)
        assert df.to_dicts() == [{"col": {"id": 3, "m": 4}}]

    def test_build_df_from_struct_rows_missing_keys_fill_null(self):
        """Test handling of missing keys in struct rows.
        Ensures proper null-filling behavior when keys are missing in some rows."""
        rows = [
            ({"k": {"a": 1}},),
            ({"k": {"b": 2}},),
        ]
        schema = StructType([StructField("k", MapType(StringType(), IntegerType()))])
        df = _MapTypeUtils.build_df_from_struct_rows(rows, schema)
        assert [f.name for f in df.schema["k"].fields] == ["a", "b"]
        assert df.to_dicts() == [{"k": {"a": 1, "b": None}}, {"k": {"a": None, "b": 2}}]

    # ---------- apply_schema_casts ----------
    def test_apply_schema_casts_top_level_primitive(self):
        """Test casting of primitive type columns.
        Verifies correct type casting for primitive columns according to schema."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        schema = StructType([StructField("x", IntegerType())])
        out = _MapTypeUtils.apply_schema_casts(df, schema)
        assert isinstance(out.schema["x"], pl.Int32)

    def test_apply_schema_casts_skips_map_and_struct(self):
        """Test selective casting behavior.
        Ensures that complex types are preserved while primitive types are cast."""
        df = pl.DataFrame(
            {
                "x": [1],
                "col": [{"a": 1, "b": 2}],
            }
        )
        schema = StructType(
            [
                StructField("x", IntegerType()),
                StructField("col", MapType(StringType(), IntegerType())),
            ]
        )
        out = _MapTypeUtils.apply_schema_casts(df, schema)
        assert isinstance(out.schema["x"], pl.Int32)
        assert isinstance(out.schema["col"], pl.Struct)

    # ---------- errors / guards ----------
    def test_map_to_struct_raises_column_not_found(self):
        """Test error handling for missing columns.
        Verifies appropriate error is raised when attempting to convert non-existent column."""
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(pl.ColumnNotFoundError):
            _MapTypeUtils.map_to_struct(df, "missing")

    def test_map_to_struct_raises_if_not_map_dtype(self):
        """Test error handling for invalid data types.
        Ensures appropriate error is raised when attempting to convert invalid column type."""
        df = pl.DataFrame({"col": [[1, 2], [3, 4]]})
        with pytest.raises(TypeError):
            _MapTypeUtils.map_to_struct(df, "col")

    def test_build_df_from_struct_rows_dict_rows_with_nulls(self):
        """Ensure MapType with a None value is preserved in the materialized struct."""
        rows = [{"col": {"a": 1, "b": None}}]
        schema = StructType([StructField("col", MapType(StringType(), IntegerType()))])

        df = _MapTypeUtils.build_df_from_struct_rows(rows, schema)

        # Underlying column should be a Struct with the inferred keys in order of first appearance
        assert isinstance(df.schema["col"], pl.Struct)
        assert [f.name for f in df.schema["col"].fields] == ["a", "b"]

        # Data should round-trip exactly, preserving None
        assert df.to_dicts() == [{"col": {"a": 1, "b": None}}]
