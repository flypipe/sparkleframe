"""Unit tests for ``DataFrame``: the constructor stores state; every action is a slot that raises.

Scope rules for this file:
- Tests the orchestration contract that only ``select`` owns: projection, output order,
  output names, resolved output dtypes, argument handling, and unimplemented slots.
- Does NOT re-assert arithmetic or coercion *values* — those are owned by ``evaluator_test``,
  ``coercion_test``, and the engine-vs-Spark ``arithmetic_parity_test`` (which is strictly
  stronger than a colocated value assertion).
- Fixtures use ≥2 columns whenever a property could otherwise pass vacuously.
"""

import pytest
from pyspark.sql.types import IntegerType, LongType, StructField, StructType

import sparkleframe.python.functions as F
from sparkleframe.python.dataframe import DataFrame


class TestDataFrameConstruction:
    def test_stores_rows_and_schema(self):
        rows = [(1, 2), (3, 4)]
        df = DataFrame(rows, schema="schema-sentinel")
        assert df._rows == rows
        assert df._schema == "schema-sentinel"

    def test_none_data_becomes_empty(self):
        df = DataFrame(None)
        assert df._rows == []
        assert df._schema is None


class TestDataFrameActionsRaise:
    @pytest.mark.parametrize(
        "call",
        [
            lambda df: df.columns(),
            lambda df: df.schema(),
            lambda df: df.dtypes(),
            lambda df: df["a"],
            lambda df: df.alias("x"),
            lambda df: df.filter("a > 1"),
            lambda df: df.where("a > 1"),
            lambda df: df.withColumn("b", None),
            lambda df: df.withColumns({}),
            lambda df: df.withColumnRenamed("a", "b"),
            lambda df: df.drop("a"),
            lambda df: df.distinct(),
            lambda df: df.dropDuplicates(),
            lambda df: df.union(df),
            lambda df: df.unionByName(df),
            lambda df: df.join(df),
            lambda df: df.groupBy("a"),
            lambda df: df.groupby("a"),
            lambda df: df.sort("a"),
            lambda df: df.orderBy("a"),
            lambda df: df.fillna(0),
            lambda df: df.count(),
            lambda df: df.show(),
            lambda df: df.toPandas(),
            lambda df: df.to_arrow(),
        ],
    )
    def test_action_raises(self, call):
        df = DataFrame([(1,)], schema=None)
        with pytest.raises(NotImplementedError):
            call(df)


class TestToRecords:
    def test_to_records_zips_schema_field_names_to_row_tuples(self):
        schema = StructType([StructField("a", IntegerType()), StructField("b", IntegerType())])
        df = DataFrame([(1, 10), (2, 20)], schema)
        assert df.to_records() == [{"a": 1, "b": 10}, {"a": 2, "b": 20}]


class TestSelectOrchestration:
    """``select`` owns: projection, output order, output names, output dtypes, arg handling.

    Value-level correctness is owned by ``arithmetic_parity_test`` (vs real Spark) and the
    AST-phase unit tests. These tests must not re-assert arithmetic or coercion values.
    """

    @staticmethod
    def _three_col_int_df() -> DataFrame:
        schema = StructType(
            [
                StructField("a", IntegerType()),
                StructField("b", IntegerType()),
                StructField("c", IntegerType()),
            ]
        )
        return DataFrame([(1, 10, 100), (2, 20, 200)], schema)

    def test_select_subset_drops_unreferenced_columns(self):
        out = self._three_col_int_df().select("a", "c")
        assert [f.name for f in out._schema.fields] == ["a", "c"]
        assert out.to_records() == [{"a": 1, "c": 100}, {"a": 2, "c": 200}]

    def test_select_output_order_follows_select_args_not_schema_order(self):
        out = self._three_col_int_df().select("c", "a")
        assert [f.name for f in out._schema.fields] == ["c", "a"]

    def test_select_single_string_arg_projects_named_column(self):
        out = self._three_col_int_df().select("b")
        assert [f.name for f in out._schema.fields] == ["b"]
        assert out.to_records() == [{"b": 10}, {"b": 20}]

    def test_select_alias_overrides_field_name_in_output_schema(self):
        out = self._three_col_int_df().select(F.col("a").alias("x"), "b")
        assert [f.name for f in out._schema.fields] == ["x", "b"]

    def test_select_mixes_expression_and_bare_column(self):
        out = self._three_col_int_df().select((F.col("a") + F.col("b")).alias("sum"), "c")
        assert [f.name for f in out._schema.fields] == ["sum", "c"]

    def test_select_resolves_output_dtype_from_analyze(self):
        """Int + Long → Long: proves ``select`` actually ran the analyze phase.

        The *value* of the sum is owned by parity / evaluator tests. We only assert dtype
        resolution here, and we pick a coercion case (Int+Long) distinct from coercion_test's
        primary table so this isn't a duplicate of that file either.
        """
        schema = StructType([StructField("i", IntegerType()), StructField("l", LongType())])
        df = DataFrame([(1, 2)], schema)
        out = df.select((F.col("i") + F.col("l")).alias("r"))
        assert isinstance(out._schema["r"].dataType, LongType)

    def test_select_unsupported_arg_raises(self):
        df = self._three_col_int_df()
        with pytest.raises(NotImplementedError):
            df.select(["a"])
