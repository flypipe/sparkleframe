"""Batch 10 parity tests: DataFrame metadata + set operations.

Covers:
    - ``count()`` — total row count
    - ``columns`` — column name order; defensive copy
    - ``schema`` — inferred ``StructType``, field names + Spark-compatible
      ``simpleString`` types
    - ``dtypes`` — list of ``(name, simpleString)`` tuples
    - ``printSchema()`` — best-effort smoke (no PySpark parity)
    - ``union`` / ``unionAll`` — positional row concat
    - ``unionByName`` — name-aligned concat (incl. ``allowMissingColumns``)
    - ``isEmpty()`` — emptiness check
    - ``toJSON()`` — one JSON string per row (no PySpark RDD parity)
"""
from __future__ import annotations

import json

import pytest

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


# -----------------------------------------------------------------------------
# count / isEmpty
# -----------------------------------------------------------------------------


class TestCount:
    def test_empty_zero_columns(self, session, F, spark):
        # An empty dict yields a frame with no columns and no rows.
        df = session.createDataFrame([{"a": 1}]).filter(F.col("a") > F.lit(100))
        assert df.count() == 0

    def test_single_row(self, session, F, spark):
        df = session.createDataFrame([{"a": 1, "b": "x"}])
        assert df.count() == 1

    def test_many_rows(self, session, F, spark):
        data = [{"a": i, "b": str(i)} for i in range(50)]
        df = session.createDataFrame(data)
        assert df.count() == 50


class TestIsEmpty:
    def test_empty(self, session, F, spark):
        df = session.createDataFrame([{"a": 1}]).filter(F.col("a") > F.lit(100))
        assert df.isEmpty() is True

    def test_non_empty(self, session, F, spark):
        df = session.createDataFrame([{"a": 1}])
        assert df.isEmpty() is False


# -----------------------------------------------------------------------------
# columns
# -----------------------------------------------------------------------------


class TestColumns:
    def test_columns_match_input_order(self, session, F, spark):
        data = [{"a": 1, "b": 2, "c": 3}]
        df = session.createDataFrame(data)
        spark_df = spark.createDataFrame(data)
        assert df.columns == spark_df.columns

    def test_columns_is_defensive_copy(self, session, F, spark):
        df = session.createDataFrame([{"a": 1, "b": 2}])
        cols = df.columns
        cols.append("c")
        # Mutating the returned list must not change the DataFrame.
        assert df.columns == ["a", "b"]


# -----------------------------------------------------------------------------
# schema / dtypes
# -----------------------------------------------------------------------------


class TestSchema:
    _DATA = [{"a": 1, "b": 1.5, "c": "x", "d": True}]

    def test_field_names_match_columns(self, session, F, spark):
        df = session.createDataFrame(self._DATA)
        assert df.schema.fieldNames() == df.columns

    def test_field_names_match_spark(self, session, F, spark):
        df = session.createDataFrame(self._DATA)
        spark_df = spark.createDataFrame(self._DATA)
        assert df.schema.fieldNames() == spark_df.schema.fieldNames()

    def test_simple_string_types_match_spark(self, session, F, spark):
        df = session.createDataFrame(self._DATA)
        spark_df = spark.createDataFrame(self._DATA)
        sf_types = [f.dataType.simpleString() for f in df.schema]
        sp_types = [f.dataType.simpleString() for f in spark_df.schema]
        assert sf_types == sp_types

    def test_long_for_python_int(self, session, F, spark):
        df = session.createDataFrame([{"a": 1}])
        spark_df = spark.createDataFrame([{"a": 1}])
        assert df.schema.fieldNames() == ["a"]
        assert df.schema.fields[0].dataType.simpleString() == "bigint"
        assert spark_df.schema.fields[0].dataType.simpleString() == "bigint"

    def test_double_for_python_float(self, session, F, spark):
        df = session.createDataFrame([{"a": 1.5}])
        spark_df = spark.createDataFrame([{"a": 1.5}])
        assert df.schema.fields[0].dataType.simpleString() == "double"
        assert spark_df.schema.fields[0].dataType.simpleString() == "double"

    def test_string_for_python_str(self, session, F, spark):
        df = session.createDataFrame([{"a": "hi"}])
        spark_df = spark.createDataFrame([{"a": "hi"}])
        assert df.schema.fields[0].dataType.simpleString() == "string"
        assert spark_df.schema.fields[0].dataType.simpleString() == "string"

    def test_boolean_for_python_bool(self, session, F, spark):
        df = session.createDataFrame([{"a": True}])
        spark_df = spark.createDataFrame([{"a": True}])
        assert df.schema.fields[0].dataType.simpleString() == "boolean"
        assert spark_df.schema.fields[0].dataType.simpleString() == "boolean"

    def test_all_null_column_defaults_to_string(self, session, F, spark):
        # PySpark defaults an all-null inferred column to StringType.
        df = session.createDataFrame([{"a": None}])
        assert df.schema.fields[0].dataType.simpleString() == "string"


class TestDtypes:
    def test_dtypes_shape(self, session, F, spark):
        df = session.createDataFrame([{"a": 1, "b": "x"}])
        dts = df.dtypes
        assert isinstance(dts, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in dts)

    def test_dtypes_match_spark(self, session, F, spark):
        data = [{"a": 1, "b": 1.5, "c": "x", "d": True}]
        df = session.createDataFrame(data)
        spark_df = spark.createDataFrame(data)
        assert df.dtypes == [(f.name, f.dataType.simpleString()) for f in spark_df.schema]


class TestPrintSchema:
    def test_print_schema_smoke(self, session, F, spark, capsys):
        df = session.createDataFrame([{"a": 1, "b": "x"}])
        df.printSchema()
        captured = capsys.readouterr()
        # Best-effort: should at least contain the column names and 'root'.
        assert "root" in captured.out
        assert "a" in captured.out
        assert "b" in captured.out


# -----------------------------------------------------------------------------
# union / unionAll
# -----------------------------------------------------------------------------


class TestUnion:
    def test_union_same_schema(self, session, F, spark):
        left_data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        right_data = [{"a": 3, "b": "z"}]
        sf = session.createDataFrame(left_data).union(session.createDataFrame(right_data))
        sp = spark.createDataFrame(left_data).union(spark.createDataFrame(right_data))
        assert_sparkle_spark_frame_are_equal(sf, sp)

    def test_union_all_alias(self, session, F, spark):
        left_data = [{"a": 1, "b": "x"}]
        right_data = [{"a": 2, "b": "y"}]
        sf = session.createDataFrame(left_data).unionAll(session.createDataFrame(right_data))
        sp = spark.createDataFrame(left_data).unionAll(spark.createDataFrame(right_data))
        assert_sparkle_spark_frame_are_equal(sf, sp)

    def test_union_mismatched_column_count_raises(self, session, F, spark):
        left = session.createDataFrame([{"a": 1, "b": "x"}])
        right = session.createDataFrame([{"a": 2}])
        with pytest.raises(ValueError):
            left.union(right)

    def test_union_rejects_non_dataframe(self, session, F, spark):
        left = session.createDataFrame([{"a": 1}])
        with pytest.raises(TypeError):
            left.union([{"a": 2}])  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# unionByName
# -----------------------------------------------------------------------------


class TestUnionByName:
    def test_same_columns_different_order(self, session, F, spark):
        left_data = [{"x": 1, "y": 10}, {"x": 2, "y": 20}]
        right_data = [{"y": 30, "x": 3}, {"y": 40, "x": 4}]
        sf = session.createDataFrame(left_data).unionByName(session.createDataFrame(right_data))
        sp = spark.createDataFrame(left_data).unionByName(spark.createDataFrame(right_data))
        assert_sparkle_spark_frame_are_equal(sf, sp)

    def test_allow_missing_columns(self, session, F, spark):
        left_data = [{"x": 1, "y": 2}]
        right_data = [{"x": 3, "z": 4}]
        sf = session.createDataFrame(left_data).unionByName(
            session.createDataFrame(right_data), allowMissingColumns=True
        )
        sp = spark.createDataFrame(left_data).unionByName(
            spark.createDataFrame(right_data), allowMissingColumns=True
        )
        assert_sparkle_spark_frame_are_equal(sf, sp)

    def test_disjoint_schemas_raise_when_strict(self, session, F, spark):
        left = session.createDataFrame([{"x": 1, "y": 2}])
        right = session.createDataFrame([{"x": 3, "z": 4}])
        with pytest.raises(ValueError):
            left.unionByName(right, allowMissingColumns=False)

    def test_rejects_non_dataframe(self, session, F, spark):
        left = session.createDataFrame([{"x": 1}])
        with pytest.raises(TypeError):
            left.unionByName([{"x": 2}])  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# toJSON
# -----------------------------------------------------------------------------


class TestToJSON:
    def test_one_json_string_per_row(self, session, F, spark):
        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
        df = session.createDataFrame(data)
        rows = list(df.toJSON())
        assert len(rows) == 3
        assert all(isinstance(r, str) for r in rows)

    def test_round_trips_through_json_load(self, session, F, spark):
        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        df = session.createDataFrame(data)
        decoded = [json.loads(s) for s in df.toJSON()]
        # Compare as a set of (key,value) tuples — order across rows is
        # preserved by collect, but field key order inside dicts may vary.
        assert sorted([sorted(d.items()) for d in decoded]) == sorted(
            [sorted(d.items()) for d in data]
        )

    def test_empty_frame(self, session, F, spark):
        df = session.createDataFrame([{"a": 1}]).filter(F.col("a") > F.lit(100))
        assert list(df.toJSON()) == []
