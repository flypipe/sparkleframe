"""Batch 8 parity tests: DataFrame core mutations.

Covers the column-shaping surface of the DataFrame API:
    - ``select(*cols)`` with names, Column expressions, and ``.alias()``
    - ``withColumn(name, col)`` for add / replace / chained / dependent
    - ``filter(condition)`` / ``where(condition)`` with compound predicates,
      ``isin``, ``isNull``
    - ``withColumns(dict)`` batched add/replace, with cross-column references
    - ``withColumnRenamed(old, new)`` rename + no-op on absent
    - ``drop(*cols)`` by name, by Column ref, multi-drop, absent name

Every test runs against both backends (``polarsdf``, ``pythondf``) and asserts
parity against native PySpark via ``assert_sparkle_spark_frame_are_equal``.
"""
from __future__ import annotations

import pyspark.sql.functions as SF

from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal


_SAMPLE = [
    {"a": 1, "b": 10, "c": "x"},
    {"a": 2, "b": 20, "c": "y"},
    {"a": 3, "b": 30, "c": "z"},
    {"a": 4, "b": 40, "c": "y"},
]


# -----------------------------------------------------------------------------
# select
# -----------------------------------------------------------------------------


class TestSelect:
    def test_select_by_names(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).select("a", "c")
        spark_df = spark.createDataFrame(_SAMPLE).select("a", "c")
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_select_by_column_expr(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).select(F.col("a"), F.col("b"))
        spark_df = spark.createDataFrame(_SAMPLE).select(SF.col("a"), SF.col("b"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_select_with_alias(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).select((F.col("a") + F.lit(100)).alias("a_plus"))
        spark_df = spark.createDataFrame(_SAMPLE).select((SF.col("a") + SF.lit(100)).alias("a_plus"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_select_mixed_name_and_expr(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).select("c", (F.col("a") * F.col("b")).alias("prod"))
        spark_df = spark.createDataFrame(_SAMPLE).select(
            "c", (SF.col("a") * SF.col("b")).alias("prod")
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_select_lit_only(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).select(F.lit(1).alias("k"))
        spark_df = spark.createDataFrame(_SAMPLE).select(SF.lit(1).alias("k"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# withColumn
# -----------------------------------------------------------------------------


class TestWithColumn:
    def test_with_column_add(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumn("d", F.col("a") + F.col("b"))
        spark_df = spark.createDataFrame(_SAMPLE).withColumn("d", SF.col("a") + SF.col("b"))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_with_column_replace_existing(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumn("b", F.col("b") * F.lit(2))
        spark_df = spark.createDataFrame(_SAMPLE).withColumn("b", SF.col("b") * SF.lit(2))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_with_column_chained(self, session, F, spark):
        df = (
            session.createDataFrame(_SAMPLE)
            .withColumn("d", F.col("a") + F.lit(1))
            .withColumn("e", F.col("b") - F.lit(1))
        )
        spark_df = (
            spark.createDataFrame(_SAMPLE)
            .withColumn("d", SF.col("a") + SF.lit(1))
            .withColumn("e", SF.col("b") - SF.lit(1))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_with_column_computed_from_new_column(self, session, F, spark):
        df = (
            session.createDataFrame(_SAMPLE)
            .withColumn("d", F.col("a") * F.lit(10))
            .withColumn("e", F.col("d") + F.lit(1))
        )
        spark_df = (
            spark.createDataFrame(_SAMPLE)
            .withColumn("d", SF.col("a") * SF.lit(10))
            .withColumn("e", SF.col("d") + SF.lit(1))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_with_column_preserves_order_on_replace(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumn("a", F.col("a") + F.lit(100))
        spark_df = spark.createDataFrame(_SAMPLE).withColumn("a", SF.col("a") + SF.lit(100))
        # Spark replaces in-place; column order should match.
        assert df.columns == spark_df.columns
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# filter / where
# -----------------------------------------------------------------------------


_FILTER_DATA = [
    {"a": 1, "b": "cat", "c": 10},
    {"a": 2, "b": "dog", "c": None},
    {"a": 3, "b": "cat", "c": 30},
    {"a": None, "b": "fish", "c": 40},
    {"a": 5, "b": None, "c": 50},
]


class TestFilter:
    def test_filter_simple(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).filter(F.col("a") > F.lit(2))
        spark_df = spark.createDataFrame(_SAMPLE).filter(SF.col("a") > SF.lit(2))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_where_alias_of_filter(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).where(F.col("a") <= F.lit(2))
        spark_df = spark.createDataFrame(_SAMPLE).where(SF.col("a") <= SF.lit(2))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_filter_compound_and(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).filter(
            (F.col("a") > F.lit(1)) & (F.col("b") < F.lit(40))
        )
        spark_df = spark.createDataFrame(_SAMPLE).filter(
            (SF.col("a") > SF.lit(1)) & (SF.col("b") < SF.lit(40))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_filter_compound_or(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).filter(
            (F.col("a") == F.lit(1)) | (F.col("a") == F.lit(4))
        )
        spark_df = spark.createDataFrame(_SAMPLE).filter(
            (SF.col("a") == SF.lit(1)) | (SF.col("a") == SF.lit(4))
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_filter_isin(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).filter(F.col("c").isin(["x", "z"]))
        spark_df = spark.createDataFrame(_SAMPLE).filter(SF.col("c").isin(["x", "z"]))
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_filter_isnotnull(self, session, F, spark):
        df = session.createDataFrame(_FILTER_DATA).filter(F.col("a").isNotNull())
        spark_df = spark.createDataFrame(_FILTER_DATA).filter(SF.col("a").isNotNull())
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_filter_isnull(self, session, F, spark):
        df = session.createDataFrame(_FILTER_DATA).filter(F.col("c").isNull())
        spark_df = spark.createDataFrame(_FILTER_DATA).filter(SF.col("c").isNull())
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_filter_gt_with_isnotnull_drops_nulls(self, session, F, spark):
        # Spark `>` returns null on null operand; filter only keeps True rows.
        df = session.createDataFrame(_FILTER_DATA).filter(
            (F.col("a") > F.lit(1)) & F.col("a").isNotNull()
        )
        spark_df = spark.createDataFrame(_FILTER_DATA).filter(
            (SF.col("a") > SF.lit(1)) & SF.col("a").isNotNull()
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_filter_null_predicate_drops_rows(self, session, F, spark):
        # `> 2` on a null value yields null; Spark filter drops null-result rows.
        df = session.createDataFrame(_FILTER_DATA).filter(F.col("a") > F.lit(2))
        spark_df = spark.createDataFrame(_FILTER_DATA).filter(SF.col("a") > SF.lit(2))
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# withColumns
# -----------------------------------------------------------------------------


class TestWithColumns:
    def test_with_columns_add_multiple(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumns(
            {
                "bonus": F.col("b") + F.lit(1),
                "tag": F.lit("v1"),
            }
        )
        spark_df = spark.createDataFrame(_SAMPLE).withColumns(
            {
                "bonus": SF.col("b") + SF.lit(1),
                "tag": SF.lit("v1"),
            }
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_with_columns_replace_existing(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumns(
            {
                "a": F.col("a") * F.lit(2),
                "b": F.col("b") + F.lit(1),
            }
        )
        spark_df = spark.createDataFrame(_SAMPLE).withColumns(
            {
                "a": SF.col("a") * SF.lit(2),
                "b": SF.col("b") + SF.lit(1),
            }
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_with_columns_mix_add_and_replace(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumns(
            {
                "a": F.col("a") + F.lit(1000),
                "new_flag": F.lit(True),
            }
        )
        spark_df = spark.createDataFrame(_SAMPLE).withColumns(
            {
                "a": SF.col("a") + SF.lit(1000),
                "new_flag": SF.lit(True),
            }
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_with_columns_single_entry(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumns({"doubled": F.col("a") * F.lit(2)})
        spark_df = spark.createDataFrame(_SAMPLE).withColumns(
            {"doubled": SF.col("a") * SF.lit(2)}
        )
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# withColumnRenamed
# -----------------------------------------------------------------------------


class TestWithColumnRenamed:
    def test_rename_existing(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumnRenamed("a", "alpha")
        spark_df = spark.createDataFrame(_SAMPLE).withColumnRenamed("a", "alpha")
        assert df.columns == spark_df.columns
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_rename_preserves_order(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumnRenamed("b", "beta")
        spark_df = spark.createDataFrame(_SAMPLE).withColumnRenamed("b", "beta")
        assert df.columns == spark_df.columns == ["a", "beta", "c"]
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_rename_absent_is_noop(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).withColumnRenamed("nope", "still_nope")
        spark_df = spark.createDataFrame(_SAMPLE).withColumnRenamed("nope", "still_nope")
        assert df.columns == spark_df.columns
        assert_sparkle_spark_frame_are_equal(df, spark_df)


# -----------------------------------------------------------------------------
# drop
# -----------------------------------------------------------------------------


class TestDrop:
    def test_drop_single_name(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).drop("b")
        spark_df = spark.createDataFrame(_SAMPLE).drop("b")
        assert df.columns == spark_df.columns
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_drop_multiple_names(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).drop("a", "c")
        spark_df = spark.createDataFrame(_SAMPLE).drop("a", "c")
        assert df.columns == spark_df.columns
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_drop_unpack_list(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).drop(*["a", "b"])
        spark_df = spark.createDataFrame(_SAMPLE).drop(*["a", "b"])
        assert df.columns == spark_df.columns
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_drop_column_expr(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).drop(F.col("a"))
        spark_df = spark.createDataFrame(_SAMPLE).drop(SF.col("a"))
        assert df.columns == spark_df.columns
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_drop_missing_is_ignored(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).drop("not_there", "a")
        spark_df = spark.createDataFrame(_SAMPLE).drop("not_there", "a")
        assert df.columns == spark_df.columns
        assert_sparkle_spark_frame_are_equal(df, spark_df)

    def test_drop_all_present_returns_empty_schema(self, session, F, spark):
        df = session.createDataFrame(_SAMPLE).drop("a", "b", "c")
        spark_df = spark.createDataFrame(_SAMPLE).drop("a", "b", "c")
        assert df.columns == spark_df.columns == []
