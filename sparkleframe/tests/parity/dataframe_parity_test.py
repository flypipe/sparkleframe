"""Shared Spark-parity tests for ``DataFrame`` operations.

Lifted from ``polarsdf/dataframe_test.py`` and converted to the engine-parametrized
harness: each test builds the actual frame via ``engine.build_df`` (or the shared
``sample`` fixture) and the expected frame via real Spark, then compares with
``assert_matches_spark``. Every expression is built from ``engine.functions`` so it
runs on the active engine. Ordering tests opt into ``check_row_order=True``.

Unit tests (polars dtype/schema assertions, ``.df`` identity, API-misuse guards) and
irregular construction tests (pandas/arrow round-trips, native polars map layouts)
stay co-located in ``polarsdf/dataframe_test.py``.
"""

from datetime import date, datetime
from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pyspark.sql.functions as SF
import pytest
from pyspark.sql.functions import col as spark_col
from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import BooleanType as SparkBooleanType
from pyspark.sql.types import DateType as SparkDateType
from pyspark.sql.types import DecimalType as SparkDecimalType
from pyspark.sql.types import DoubleType as SparkDoubleType
from pyspark.sql.types import FloatType as SparkFloatType
from pyspark.sql.types import IntegerType as SparkIntegerType
from pyspark.sql.types import LongType as SparkLongType
from pyspark.sql.types import StringType as SparkStringType
from pyspark.sql.types import StructField as SparkStructField
from pyspark.sql.types import StructType as SparkStructType
from pyspark.sql.types import TimestampType as SparkTimestampType

from sparkleframe.tests.parity.oracle import assert_matches_spark

_LONG = SparkLongType()
_STR = SparkStringType()


def _schema(*fields):
    return SparkStructType([SparkStructField(name, dtype, True) for name, dtype in fields])


# ----------------------------------------------------------------------------- #
# select
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.select.by_list_str")
@pytest.mark.parametrize("cols", ["name", ["name", "age"]])
def test_select_by_list_str(engine, spark, sample, cols):
    actual = sample.sparkle(engine).select(cols)
    expected = sample.spark(spark).select(cols)
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.select.by_list_columns")
@pytest.mark.parametrize("cols", ["name", ["name", "age"]])
def test_select_by_list_columns(engine, spark, sample, cols):
    F = engine.functions
    cols = [cols] if isinstance(cols, str) else cols
    actual = sample.sparkle(engine).select([F.col(c) for c in cols])
    expected = sample.spark(spark).select([SF.col(c) for c in cols])
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.select.by_column_name")
def test_select_by_column_name(engine, spark, sample):
    actual = sample.sparkle(engine).select("name")
    expected = sample.spark(spark).select("name")
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.select.by_pointer_list")
def test_select_by_pointer_list(engine, spark, sample):
    actual = sample.sparkle(engine).select(*["name", "age"])
    expected = sample.spark(spark).select(*["name", "age"])
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.select.by_expression")
def test_select_by_expression(engine, spark, sample):
    F = engine.functions
    actual = sample.sparkle(engine).select(F.col("name"), (F.col("salary") * 1.1).alias("salary"))
    expected = sample.spark(spark).select(spark_col("name"), (spark_col("salary") * 1.1).alias("salary"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.select.expression_auto_name")
def test_select_expression_auto_name(engine, spark, sample):
    """An unaliased expression is named after its SQL text, e.g. ``(salary * 1.1)``."""
    F = engine.functions
    actual = sample.sparkle(engine).select(F.col("name"), F.col("salary") * 1.1, F.col("age") - F.lit(1))
    expected = sample.spark(spark).select(spark_col("name"), spark_col("salary") * 1.1, spark_col("age") - SF.lit(1))
    assert actual.columns == expected.columns
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.select.all_columns_with_aliases")
def test_select_all_columns_with_aliases(engine, spark, sample):
    F = engine.functions
    aliases = {"name": "employee_name", "age": "employee_age", "salary": "employee_salary"}
    actual = sample.sparkle(engine).select(*(F.col(c).alias(a) for c, a in aliases.items()))
    expected = sample.spark(spark).select(*(spark_col(c).alias(a) for c, a in aliases.items()))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.select.literal_only_broadcasts")
def test_select_literal_only_broadcasts_to_source_row_count(engine, spark, sample):
    """Spark's ``df.select(lit(x))`` produces one row per source row (broadcast)."""
    F = engine.functions
    sf = sample.sparkle(engine)
    actual = sf.select(F.lit("constant").alias("c"), F.lit(42).alias("n"))
    expected = sample.spark(spark).select(SF.lit("constant").alias("c"), SF.lit(42).alias("n"))

    assert actual.count() == sf.count() == expected.count()
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.select.literal_broadcast_empty")
def test_select_literal_broadcast_with_empty_dataframe(engine, spark):
    """Spark parity edge case: selecting a literal against an empty frame yields zero rows."""
    F = engine.functions
    schema = _schema(("a", _LONG))
    actual = engine.build_df([], schema).select(F.lit("constant").alias("c"))
    expected = spark.createDataFrame([], schema).select(SF.lit("constant").alias("c"))

    assert actual.count() == 0 == expected.count()
    assert_matches_spark(actual, expected, engine)


# ----------------------------------------------------------------------------- #
# withColumn / withColumns
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.with_column.null_lit_empty_rows")
def test_with_column_null_lit_empty_rows_matches_spark(engine, spark):
    F = engine.functions
    schema = _schema(("id", _LONG))
    actual = engine.build_df([], schema).withColumn("n", F.lit(None))
    expected = spark.createDataFrame([], schema).withColumn("n", SF.lit(None).cast("string"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.with_column.add")
def test_with_column_add(engine, spark, sample):
    F = engine.functions
    actual = sample.sparkle(engine).withColumn("bonus", F.col("salary") * 0.1)
    expected = (
        sample.spark(spark)
        .withColumn("bonus", spark_col("salary") * 0.1)
        .withColumn("bonus", SF.col("bonus").cast(SparkFloatType()))
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.with_column.replace")
def test_with_column_replace(engine, spark, sample):
    F = engine.functions
    actual = sample.sparkle(engine).withColumn("salary", F.col("salary") * 2)
    expected = sample.spark(spark).withColumn("salary", spark_col("salary") * 2)
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.with_columns.add_multiple")
def test_with_columns_add_multiple_against_spark(engine, spark, sample):
    F = engine.functions
    actual = sample.sparkle(engine).withColumns({"bonus": F.col("salary") * 0.1, "tag": F.lit("v1")})
    expected = sample.spark(spark).withColumns({"bonus": SF.col("salary") * 0.1, "tag": SF.lit("v1")})
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.with_columns.replace_existing")
def test_with_columns_replace_existing_against_spark(engine, spark, sample):
    F = engine.functions
    actual = sample.sparkle(engine).withColumns({"salary": F.col("salary") * 2, "age": F.col("age") + 1})
    expected = sample.spark(spark).withColumns({"salary": SF.col("salary") * 2, "age": SF.col("age") + 1})
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.with_columns.mix_add_and_replace")
def test_with_columns_mix_add_and_replace_against_spark(engine, spark, sample):
    F = engine.functions
    actual = sample.sparkle(engine).withColumns({"salary": F.col("salary") + 1000, "new_flag": F.lit(True)})
    expected = sample.spark(spark).withColumns({"salary": SF.col("salary") + 1000, "new_flag": SF.lit(True)})
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.with_columns.single_entry")
def test_with_columns_single_entry_against_spark(engine, spark, sample):
    F = engine.functions
    actual = sample.sparkle(engine).withColumns({"doubled": F.col("age") * 2})
    expected = sample.spark(spark).withColumns({"doubled": SF.col("age") * 2})
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.with_columns.references_prior_column")
def test_with_columns_references_prior_column_against_spark(engine, spark):
    """A later entry can reference a column created by an earlier entry (Spark 3.4+ behavior)."""
    F = engine.functions
    schema = _schema(("x", _LONG))
    rows = [(10,), (20,)]
    actual = engine.build_df(rows, schema).withColumns({"y": F.col("x") + 1, "z": F.col("y") * 2})
    expected = spark.createDataFrame(rows, schema).withColumns({"y": SF.col("x") + 1, "z": SF.col("y") * 2})
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.with_columns.replaced_name_reads_input")
def test_with_columns_replaced_name_reads_input_against_spark(engine, spark):
    """A name already in the input keeps meaning the input column, even when an earlier entry replaces it."""
    F = engine.functions
    schema = _schema(("x", _LONG))
    rows = [(10,), (20,)]
    actual = engine.build_df(rows, schema).withColumns({"x": F.col("x") + 1, "z": F.col("x") * 2})
    expected = spark.createDataFrame(rows, schema).withColumns({"x": SF.col("x") + 1, "z": SF.col("x") * 2})
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.with_columns.forward_reference_raises")
def test_with_columns_forward_reference_raises(engine, spark):
    """An entry cannot see a column only a *later* entry of the same ``withColumns`` defines."""
    F = engine.functions
    schema = _schema(("x", _LONG))
    rows = [(10,), (20,)]
    with pytest.raises(Exception, match="UNRESOLVED_COLUMN"):
        spark.createDataFrame(rows, schema).withColumns({"z": SF.col("y") * 2, "y": SF.col("x") + 1}).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).withColumns({"z": F.col("y") * 2, "y": F.col("x") + 1}))


@pytest.mark.feature("dataframe.with_column.renamed")
def test_with_column_renamed(engine, spark, sample):
    actual = sample.sparkle(engine).withColumnRenamed("name", "employee_name")
    expected = sample.spark(spark).withColumnRenamed("name", "employee_name")
    assert_matches_spark(actual, expected, engine)


# ----------------------------------------------------------------------------- #
# drop
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.drop.columns")
def test_drop_columns(engine, spark, sample):
    actual = sample.sparkle(engine).drop("salary", "birth_date")
    expected = sample.spark(spark).drop("salary", "birth_date")
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.drop.unpack_list")
def test_drop_unpack_list(engine, spark, sample):
    actual = sample.sparkle(engine).drop(*["salary", "age"])
    expected = sample.spark(spark).drop(*["salary", "age"])
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.drop.column_expr")
def test_drop_column_expr(engine, spark, sample):
    F = engine.functions
    actual = sample.sparkle(engine).drop(F.col("salary"))
    expected = sample.spark(spark).drop(SF.col("salary"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.drop.missing_column_ignored")
def test_drop_missing_column_ignored(engine, spark, sample):
    actual = sample.sparkle(engine).drop("not_there", "age")
    expected = sample.spark(spark).drop("not_there", "age")
    assert_matches_spark(actual, expected, engine)


# ----------------------------------------------------------------------------- #
# column predicates / expressions
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.is_not_null")
def test_is_not_null(engine, spark):
    F = engine.functions
    schema = _schema(("id", _LONG), ("name", _STR))
    rows = [(1, "Alice"), (2, None), (3, "Charlie"), (4, None)]
    actual = engine.build_df(rows, schema).select(F.col("name").isNotNull().alias("not_null"))
    expected = spark.createDataFrame(rows, schema).select(SF.col("name").isNotNull().alias("not_null"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.reverse_arithmetic_operators")
@pytest.mark.parametrize(
    "op_name, expr_func",
    [
        ("+", lambda c: 10 + c),
        ("-", lambda c: 10 - c),
        ("*", lambda c: 10 * c),
        ("/", lambda c: 10 / c),
    ],
)
def test_reverse_arithmetic_operators(engine, spark, op_name, expr_func):
    F = engine.functions
    schema = _schema(("a", _LONG))
    rows = [(1,), (2,), (3,)]
    actual = engine.build_df(rows, schema).select(expr_func(F.col("a")).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(expr_func(SF.col("a")).alias("result"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.logical_operations")
@pytest.mark.parametrize(
    "description, expr_func",
    [
        ("AND", lambda a, b, c: (a > 1) & (b < 6)),
        ("OR", lambda a, b, c: (a > 2) | (b < 6)),
        ("chained AND-OR", lambda a, b, c: ((a > 1) & (b < 6)) | (c > 7)),
        ("chained OR-AND", lambda a, b, c: (a < 2) | ((b == 5) & (c < 9))),
    ],
)
def test_logical_operations(engine, spark, description, expr_func):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG), ("c", _LONG))
    rows = list(zip([1, 2, 3, 4], [10, 5, 3, 8], [7, 12, 9, 4]))
    actual = engine.build_df(rows, schema).select(expr_func(F.col("a"), F.col("b"), F.col("c")).alias("result"))
    expected = spark.createDataFrame(rows, schema).select(
        expr_func(SF.col("a"), SF.col("b"), SF.col("c")).alias("result")
    )
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.filter_and_where")
@pytest.mark.parametrize(
    "description, expr_func",
    [
        ("filter by single column", lambda a, b, c: a > 2),
        ("filter with AND", lambda a, b, c: (a > 1) & (b < 10)),
        ("filter with OR", lambda a, b, c: (a < 2) | (c > 7)),
        ("chained AND-OR", lambda a, b, c: ((a > 1) & (b < 6)) | (c > 7)),
    ],
)
def test_filter_and_where(engine, spark, description, expr_func):
    F = engine.functions
    schema = _schema(("a", _LONG), ("b", _LONG), ("c", _LONG))
    rows = list(zip([1, 2, 3, 4], [10, 5, 3, 8], [7, 12, 9, 4]))
    sparkle_df = engine.build_df(rows, schema)
    expected = spark.createDataFrame(rows, schema).filter(expr_func(SF.col("a"), SF.col("b"), SF.col("c")))

    filtered = sparkle_df.filter(expr_func(F.col("a"), F.col("b"), F.col("c")))
    assert_matches_spark(filtered, expected, engine)

    where_result = sparkle_df.where(expr_func(F.col("a"), F.col("b"), F.col("c")))
    assert_matches_spark(where_result, expected, engine)


@pytest.mark.feature("dataframe.rlike")
@pytest.mark.parametrize(
    "pattern, expected_matches",
    [
        (".*a.*", ["Alice", "Charlie"]),
        ("^A.*", ["Alice"]),
        (".*b$", ["Bob"]),
        ("^C.*e$", ["Charlie"]),
        ("[aeiou]{2}", []),
    ],
)
def test_rlike(engine, spark, pattern, expected_matches):
    F = engine.functions
    schema = _schema(("name", _STR))
    rows = [("Alice",), ("Bob",), ("Charlie",)]
    actual = engine.build_df(rows, schema).select(F.col("name").rlike(pattern).alias("match"))
    expected = spark.createDataFrame(rows, schema).select(SF.col("name").rlike(pattern).alias("match"))
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.isin")
@pytest.mark.parametrize(
    "column_name, values, use_variadic",
    [
        ("name", ["Alice", "Bob"], False),
        ("name", ["Alice", "Bob"], True),
        ("name", ["Zoe"], False),
        ("name", ["Zoe"], True),
        ("age", [25, 30], False),
        ("age", [25, 30], True),
        ("age", [100], False),
        ("age", [100], True),
        ("salary", [], False),
        ("salary", [], True),
    ],
)
def test_isin(engine, spark, column_name, values, use_variadic):
    F = engine.functions
    schema = _schema(("name", _STR), ("age", _LONG), ("salary", _LONG))
    rows = list(zip(["Alice", "Bob", "Charlie"], [25, 30, 35], [70000, 80000, 90000]))
    sparkle_df = engine.build_df(rows, schema)
    spark_df = spark.createDataFrame(rows, schema)

    if use_variadic:
        actual = sparkle_df.select(F.col(column_name).isin(*values).alias("match"))
    else:
        actual = sparkle_df.select(F.col(column_name).isin(values).alias("match"))
    expected = spark_df.select(SF.col(column_name).isin(values).alias("match"))
    assert_matches_spark(actual, expected, engine)


# ----------------------------------------------------------------------------- #
# groupBy / aggregations
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.groupby.aggregations")
@pytest.mark.parametrize("use_alias", [False, True])
@pytest.mark.parametrize("agg_name", ["count", "sum", "mean", "min", "max"])
def test_groupby_aggregations(engine, spark, use_alias, agg_name):
    F = engine.functions
    sf_agg = getattr(F, agg_name)
    sp_agg = getattr(SF, agg_name)
    schema = _schema(("group", _STR), ("value", _LONG))
    rows = list(zip(["A", "A", "B", "B", "A", "B"], [10, 20, 5, 15, 30, 25]))

    spark_df = spark.createDataFrame(rows, schema)
    expected = (spark_df.groupby("group") if use_alias else spark_df.groupBy("group")).agg(
        sp_agg("value").alias("agg_result")
    )
    expected = expected.withColumn("agg_result", SF.col("agg_result").cast("float"))

    sparkle_df = engine.build_df(rows, schema)
    actual = (sparkle_df.groupby("group") if use_alias else sparkle_df.groupBy("group")).agg(
        sf_agg("value").alias("agg_result")
    )
    actual = actual.withColumn("agg_result", F.col("agg_result").cast(engine.dtype(SparkFloatType())))

    assert_matches_spark(actual.orderBy("group"), expected.orderBy("group"), engine)


@pytest.mark.feature("dataframe.groupby.collect_list")
@pytest.mark.parametrize("use_alias", [False, True])
def test_groupby_collect_list(engine, spark, use_alias):
    """collect_list matches Spark; rows are ordered so list order is deterministic."""
    F = engine.functions
    schema = _schema(("group", _STR), ("value", _LONG))
    rows = list(zip(["A", "A", "A", "B", "B"], [3, None, 1, 2, None]))

    spark_df = spark.createDataFrame(rows, schema).orderBy("group", SF.asc_nulls_last("value"))
    sparkle_df = engine.build_df(rows, schema).sort("group", F.asc_nulls_last("value"))

    expected = (spark_df.groupby("group") if use_alias else spark_df.groupBy("group")).agg(
        SF.collect_list("value").alias("agg_result")
    )
    actual = (sparkle_df.groupby("group") if use_alias else sparkle_df.groupBy("group")).agg(
        F.collect_list("value").alias("agg_result")
    )
    assert_matches_spark(actual.orderBy("group"), expected.orderBy("group"), engine)


@pytest.mark.feature("dataframe.groupby.collect_set")
@pytest.mark.parametrize("use_alias", [False, True])
def test_groupby_collect_set(engine, spark, use_alias):
    """collect_set matches Spark; sort_array makes order comparable (sets are unordered)."""
    F = engine.functions
    schema = _schema(("group", _STR), ("value", _LONG))
    rows = list(zip(["A", "A", "A", "B", "B"], [3, None, 1, 2, None]))

    spark_df = spark.createDataFrame(rows, schema).orderBy("group", SF.asc_nulls_last("value"))
    sparkle_df = engine.build_df(rows, schema).sort("group", F.asc_nulls_last("value"))

    expected = (spark_df.groupby("group") if use_alias else spark_df.groupBy("group")).agg(
        SF.collect_set("value").alias("agg_result")
    )
    expected = expected.select(SF.col("group"), SF.sort_array(SF.col("agg_result")).alias("agg_result")).orderBy(
        "group"
    )

    actual = (sparkle_df.groupby("group") if use_alias else sparkle_df.groupBy("group")).agg(
        F.collect_set("value").alias("agg_result")
    )
    actual = actual.select(F.col("group"), F.sort_array(F.col("agg_result")).alias("agg_result")).orderBy("group")

    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.groupby.collect_list_nested_aliased_struct")
def test_groupby_collect_list_nested_aliased_struct(engine, spark):
    """collect_list(struct(...)) preserves aliased nested struct field names (not col1/col2)."""
    F = engine.functions
    schema = _schema(("group_id", _STR), ("c1", _STR), ("c2", _LONG), ("c3", _STR))
    rows = [("g1", "u", 10, "w"), ("g1", "v", 20, "z")]

    spark_item = SF.struct(
        SF.struct(SF.col("c1"), SF.col("c2")).alias("nested_x"),
        SF.struct(SF.col("c3").alias("leaf")).alias("nested_y"),
    ).alias("item")
    expected = (
        spark.createDataFrame(rows, schema)
        .withColumn("item", spark_item)
        .orderBy("group_id", "c1")
        .groupBy("group_id")
        .agg(SF.collect_list("item").alias("items"))
    )

    sf_item = F.struct(
        F.struct(F.col("c1"), F.col("c2")).alias("nested_x"),
        F.struct(F.col("c3").alias("leaf")).alias("nested_y"),
    ).alias("item")
    actual = (
        engine.build_df(rows, schema)
        .withColumn("item", sf_item)
        .sort("group_id", "c1")
        .groupBy("group_id")
        .agg(F.collect_list("item").alias("items"))
    )
    assert_matches_spark(actual.orderBy("group_id"), expected.orderBy("group_id"), engine)


# ----------------------------------------------------------------------------- #
# join
# ----------------------------------------------------------------------------- #

_JOIN_LEFT_SCHEMA = _schema(("id", _LONG), ("left_val", _STR))
_JOIN_RIGHT_SCHEMA = _schema(("id", _LONG), ("right_val", _STR))
_JOIN_LEFT_ROWS = [(1, "a"), (2, "b"), (3, "c")]
_JOIN_RIGHT_ROWS = [(2, "x"), (3, "y"), (4, "z")]


@pytest.mark.feature("dataframe.join.polars_joins")
@pytest.mark.parametrize(
    "how,on_input,expected_rows,expected_cols",
    [
        ("inner", "id", [(2, "b", "x"), (3, "c", "y")], ["id", "left_val", "right_val"]),
        ("left", "id", [(1, "a", None), (2, "b", "x"), (3, "c", "y")], ["id", "left_val", "right_val"]),
        ("right", "id", [(2, "b", "x"), (3, "c", "y"), (4, None, "z")], ["id", "left_val", "right_val"]),
        (
            "outer",
            "id",
            [(1, "a", None), (2, "b", "x"), (3, "c", "y"), (4, None, "z")],
            ["id", "left_val", "right_val"],
        ),
        ("semi", "id", [(2, "b"), (3, "c")], ["id", "left_val"]),
        ("anti", "id", [(1, "a")], ["id", "left_val"]),
    ],
)
@pytest.mark.parametrize("on_format", ["str", "list_str"])
def test_polars_joins(engine, spark, how, on_input, expected_rows, expected_cols, on_format):
    left = engine.build_df(_JOIN_LEFT_ROWS, _JOIN_LEFT_SCHEMA)
    right = engine.build_df(_JOIN_RIGHT_ROWS, _JOIN_RIGHT_SCHEMA)

    on = on_input if on_format == "str" else [on_input]
    result = left.join(right, on=on, how=how)
    cols = sorted(result.columns)
    result = result.select(cols).orderBy(cols)

    # Build the expected Spark frame from declared rows + a schema (no pandas round-trip).
    expected_fields = [(c, _LONG if c in ("id", "id_right") else _STR) for c in expected_cols]
    spark_expected = spark.createDataFrame(expected_rows, _schema(*expected_fields))
    cols = sorted(spark_expected.columns)
    spark_expected = spark_expected.select(cols).orderBy(cols)

    assert_matches_spark(result, spark_expected, engine)


@pytest.mark.feature("dataframe.join.non_spark_duplicated_keys")
@pytest.mark.parametrize(
    "how,on_keys",
    [
        ("inner", (False, "id")),
        ("left", (False, "id")),
        ("right", (False, "id")),
        ("outer", (False, "id")),
        ("inner", [(False, "id")]),
        ("left", [(False, "id")]),
        ("right", [(False, "id")]),
        ("outer", [(False, "id")]),
    ],
)
def test_joins_non_spark_duplicated_keys(engine, spark, how, on_keys):
    F = engine.functions

    def get_on(funcs, is_col, k):
        return funcs.col(k) if is_col else k

    if isinstance(on_keys, list):
        on_keys = list(on_keys)
        spark_on_keys = [get_on(SF, k[0], k[1]) for k in on_keys]
        sf_on_keys = [get_on(F, k[0], k[1]) for k in on_keys]
    else:
        spark_on_keys = get_on(SF, on_keys[0], on_keys[1])
        sf_on_keys = get_on(F, on_keys[0], on_keys[1])

    spark_left = spark.createDataFrame(_JOIN_LEFT_ROWS, _JOIN_LEFT_SCHEMA)
    spark_right = spark.createDataFrame(_JOIN_RIGHT_ROWS, _JOIN_RIGHT_SCHEMA)
    expected = spark_left.join(spark_right, on=spark_on_keys, how=how)

    sf_left = engine.build_df(_JOIN_LEFT_ROWS, _JOIN_LEFT_SCHEMA)
    sf_right = engine.build_df(_JOIN_RIGHT_ROWS, _JOIN_RIGHT_SCHEMA)
    result = sf_left.join(sf_right, on=sf_on_keys, how=how)

    assert_matches_spark(
        result.select(sorted(result.columns)).orderBy("id"),
        expected.select(sorted(expected.columns)).orderBy("id"),
        engine,
    )


@pytest.mark.feature("dataframe.join.with_duplicated_spark_keys")
@pytest.mark.parametrize("how", ["inner", "left", "right", "outer", "semi", "anti"])
@pytest.mark.parametrize("as_list", [False, True], ids=["col", "list_col"])
def test_joins_with_duplicated_spark_keys(engine, spark, how, as_list):
    """``on=col("id")`` with ``id`` on both sides is an ambiguous self-comparison: Spark rejects it."""
    F = engine.functions
    spark_on = [SF.col("id")] if as_list else SF.col("id")
    sf_on = [F.col("id")] if as_list else F.col("id")

    spark_left = spark.createDataFrame(_JOIN_LEFT_ROWS, _JOIN_LEFT_SCHEMA)
    spark_right = spark.createDataFrame(_JOIN_RIGHT_ROWS, _JOIN_RIGHT_SCHEMA)
    with pytest.raises(Exception, match="AMBIGUOUS_REFERENCE"):
        spark_left.join(spark_right, on=spark_on, how=how).collect()

    sf_left = engine.build_df(_JOIN_LEFT_ROWS, _JOIN_LEFT_SCHEMA)
    sf_right = engine.build_df(_JOIN_RIGHT_ROWS, _JOIN_RIGHT_SCHEMA)
    with pytest.raises(Exception, match="AMBIGUOUS_REFERENCE"):
        engine.to_records(sf_left.join(sf_right, on=sf_on, how=how))


@pytest.mark.feature("dataframe.join.cross_keeps_duplicate_names")
@pytest.mark.parametrize("form", ["join_without_on", "cross_join"])
def test_cross_join_keeps_duplicate_names(engine, spark, form):
    """A cross join keeps both ``id`` columns: a bare ``id`` is ambiguous, ``df["id"]`` picks a side."""
    spark_left = spark.createDataFrame(_JOIN_LEFT_ROWS, _JOIN_LEFT_SCHEMA)
    spark_right = spark.createDataFrame(_JOIN_RIGHT_ROWS, _JOIN_RIGHT_SCHEMA)
    sf_left = engine.build_df(_JOIN_LEFT_ROWS, _JOIN_LEFT_SCHEMA)
    sf_right = engine.build_df(_JOIN_RIGHT_ROWS, _JOIN_RIGHT_SCHEMA)
    if form == "join_without_on":
        spark_joined, sf_joined = spark_left.join(spark_right), sf_left.join(sf_right)
    else:
        spark_joined, sf_joined = spark_left.crossJoin(spark_right), sf_left.crossJoin(sf_right)

    assert sf_joined.columns == spark_joined.columns
    with pytest.raises(Exception, match="AMBIGUOUS_REFERENCE"):
        spark_joined.select("id").collect()
    with pytest.raises(Exception, match="AMBIGUOUS_REFERENCE"):
        engine.to_records(sf_joined.select("id"))

    expected = spark_joined.select(
        spark_left["id"].alias("left_id"), spark_right["id"].alias("right_id"), "left_val", "right_val"
    )
    actual = sf_joined.select(
        sf_left["id"].alias("left_id"), sf_right["id"].alias("right_id"), "left_val", "right_val"
    )
    assert_matches_spark(actual, expected, engine)


# ----------------------------------------------------------------------------- #
# fillna
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.fillna.pyspark_fillna")
@pytest.mark.parametrize(
    "fillna_value, subset",
    [
        (0, None),
        ("unknown", "name"),
        ({"name": "missing", "age": 0}, None),
    ],
)
def test_pyspark_fillna(engine, spark, fillna_value, subset):
    schema = _schema(("name", _STR), ("age", _LONG))
    rows = [("Alice", 25), (None, None), ("Charlie", 35)]
    sparkle_df = engine.build_df(rows, schema)
    spark_df = spark.createDataFrame(rows, schema)

    if isinstance(fillna_value, dict):
        filled_sparkle = sparkle_df.fillna(fillna_value)
        filled_spark = spark_df.fillna(fillna_value)
    else:
        filled_sparkle = sparkle_df.fillna(fillna_value, subset=subset)
        filled_spark = spark_df.fillna(fillna_value, subset=subset)

    filled_sparkle = filled_sparkle.select(sorted(filled_sparkle.columns)).orderBy("name", "age")
    filled_spark = filled_spark.select(sorted(filled_spark.columns)).orderBy("name", "age")
    assert_matches_spark(filled_sparkle, filled_spark, engine)


@pytest.mark.feature("dataframe.fillna.fillna")
@pytest.mark.parametrize(
    "fillna_value, subset, expected_rows",
    [
        (1, None, [("Alice", 25), (None, 1), ("Charlie", 35)]),
        ("unknown", "name", [("Alice", 25), ("unknown", None), ("Charlie", 35)]),
        ({"name": "missing", "age": 0}, None, [("Alice", 25), ("missing", 0), ("Charlie", 35)]),
    ],
)
def test_fillna(engine, spark, fillna_value, subset, expected_rows):
    schema = _schema(("name", _STR), ("age", _LONG))
    rows = [("Alice", 25), (None, None), ("Charlie", 35)]

    sparkle_df = engine.build_df(rows, schema).fillna(fillna_value, subset=subset)
    result = sparkle_df.select(sorted(sparkle_df.columns)).orderBy("name", "age")

    expected = spark.createDataFrame(expected_rows, schema)
    expected = expected.select(sorted(expected.columns)).orderBy("name", "age")
    assert_matches_spark(result, expected, engine)


# ----------------------------------------------------------------------------- #
# __getitem__
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.getitem.column")
def test_getitem_column(engine, spark):
    schema = _schema(("age", _LONG), ("name", _STR))
    rows = [(2, "Alice"), (5, "Bob")]

    sf_df = engine.build_df(rows, schema)
    actual = sf_df[sf_df["age"] > 3]
    spark_df = spark.createDataFrame(rows, schema)
    expected = spark_df[spark_df["age"] > 3]
    assert_matches_spark(actual, expected, engine)


@pytest.mark.feature("dataframe.getitem.list")
def test_getitem_list(engine, spark):
    schema = _schema(("age", _LONG), ("name", _STR))
    rows = [(2, "Alice"), (5, "Bob")]

    sf_df = engine.build_df(rows, schema)
    actual = sf_df[["name", "age"]].orderBy("name")
    spark_df = spark.createDataFrame(rows, schema)
    expected = spark_df[["name", "age"]].orderBy("name")
    assert_matches_spark(actual, expected, engine)


# ----------------------------------------------------------------------------- #
# orderBy
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.order_by.comparison")
@pytest.mark.parametrize("use_col", [False, True])
@pytest.mark.parametrize(
    "columns_order",
    [
        [("id", "asc")],
        [("id", "asc_nulls_first")],
        [("id", "asc_nulls_last")],
        [("name", "asc"), ("id", "asc")],
        [("name", "asc_nulls_first"), ("id", "asc_nulls_first")],
        [("name", "asc_nulls_last"), ("id", "asc_nulls_last")],
        [("id", "desc")],
        [("id", "desc_nulls_first")],
        [("id", "desc_nulls_last")],
        [("name", "desc"), ("id", "desc")],
        [("name", "desc_nulls_first"), ("id", "desc_nulls_first")],
        [("name", "desc_nulls_last"), ("id", "desc_nulls_last")],
        [("name", "asc"), ("id", "desc")],
        [("name", "asc_nulls_first"), ("id", "desc_nulls_first")],
        [("name", "asc_nulls_last"), ("id", "desc_nulls_last")],
        [("name", "desc"), ("id", "asc")],
        [("name", "desc_nulls_first"), ("id", "asc_nulls_first")],
        [("name", "desc_nulls_last"), ("id", "asc_nulls_last")],
    ],
)
def test_order_by_comparison_with_spark(engine, spark, use_col, columns_order):
    F = engine.functions
    schema = _schema(("id", _LONG), ("name", _STR))
    rows = list(zip([3, 1, 2, 0], ["Charlie", "Alice", "Bob", None]))

    sf_order = [getattr(F, direction)(c if not use_col else F.col(c)) for c, direction in columns_order]
    sp_order = [getattr(SF, direction)(c if not use_col else SF.col(c)) for c, direction in columns_order]

    actual = engine.build_df(rows, schema).orderBy(*sf_order)
    expected = spark.createDataFrame(rows, schema).orderBy(*sp_order)
    assert_matches_spark(actual, expected, engine, check_row_order=True)


@pytest.mark.feature("dataframe.order_by.int")
@pytest.mark.parametrize("ordinal", [1, 2, -1, -2])
def test_order_by_int_matches_spark(engine, spark, ordinal):
    schema = _schema(("age", _LONG), ("name", _STR))
    rows = list(zip([2, 5, 1], ["Alice", "Bob", "Carol"]))
    actual = engine.build_df(rows, schema).orderBy(ordinal)
    expected = spark.createDataFrame(rows, schema).orderBy(ordinal)
    assert_matches_spark(actual, expected, engine, check_row_order=True)


@pytest.mark.feature("dataframe.order_by.int_zero_raises")
def test_order_by_int_zero_raises(engine, spark):
    """Ordinals are 1-based: ``orderBy(0)`` is rejected."""
    schema = _schema(("age", _LONG), ("name", _STR))
    rows = [(2, "Alice")]
    with pytest.raises(Exception, match="ZERO_INDEX"):
        spark.createDataFrame(rows, schema).orderBy(0)
    with pytest.raises(Exception, match="ZERO_INDEX"):
        engine.build_df(rows, schema).orderBy(0)


# ----------------------------------------------------------------------------- #
# union / unionByName
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.union.string_promotion")
@pytest.mark.parametrize(
    "numeric_type, numeric_value, text",
    [(_LONG, 10, "20"), (SparkDoubleType(), 1.5, "2.25")],
    ids=["bigint", "double"],
)
@pytest.mark.parametrize("method", ["union", "unionByName"])
def test_union_string_promotion_against_spark(engine, spark, numeric_type, numeric_value, text, method):
    """A string column unioned with a numeric one widens to bigint/double; the strings are cast strictly."""
    left_schema, right_schema = _schema(("v", numeric_type)), _schema(("v", _STR))
    left_rows, right_rows = [(numeric_value,), (None,)], [(text,), (None,)]

    def unioned(left, right):
        return getattr(left, method)(right).select(spark_col("v").cast("string").alias("v"))

    expected = unioned(spark.createDataFrame(left_rows, left_schema), spark.createDataFrame(right_rows, right_schema))
    actual = getattr(engine.build_df(left_rows, left_schema), method)(engine.build_df(right_rows, right_schema))
    actual = actual.select(engine.functions.col("v").cast(engine.dtype(_STR)).alias("v"))
    assert_matches_spark(actual, expected, engine)

    # Malformed: a non-numeric string fails the implicit cast instead of becoming null.
    bad_rows = [("abc",)]
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        getattr(spark.createDataFrame(left_rows, left_schema), method)(
            spark.createDataFrame(bad_rows, right_schema)
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            getattr(engine.build_df(left_rows, left_schema), method)(engine.build_df(bad_rows, right_schema))
        )


@pytest.mark.feature("dataframe.union_by_name.basic")
def test_union_by_name_against_spark(engine, spark):
    left_schema = _schema(("x", _LONG), ("y", _LONG))
    right_schema = _schema(("y", _LONG), ("x", _LONG))
    left_rows = [(1, 10), (2, 20)]
    right_rows = [(30, 3), (40, 4)]

    sf_result = engine.build_df(left_rows, left_schema).unionByName(engine.build_df(right_rows, right_schema))
    sp_result = spark.createDataFrame(left_rows, left_schema).unionByName(
        spark.createDataFrame(right_rows, right_schema)
    )
    assert_matches_spark(sf_result, sp_result, engine)


@pytest.mark.feature("dataframe.union_by_name.allow_missing")
def test_union_by_name_allow_missing_against_spark(engine, spark):
    left_schema = _schema(("x", _LONG), ("y", _LONG))
    right_schema = _schema(("x", _LONG), ("z", _LONG))
    left_rows = [(1, 2)]
    right_rows = [(3, 4)]

    sf_result = engine.build_df(left_rows, left_schema).unionByName(
        engine.build_df(right_rows, right_schema), allowMissingColumns=True
    )
    sp_result = spark.createDataFrame(left_rows, left_schema).unionByName(
        spark.createDataFrame(right_rows, right_schema), allowMissingColumns=True
    )
    assert_matches_spark(sf_result, sp_result, engine)


# ----------------------------------------------------------------------------- #
# createDataFrame
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.create.decimal_rescaled_to_field")
def test_create_decimal_rescaled_to_field_against_spark(engine, spark):
    """A ``Decimal`` is stored at the field's scale (rounded HALF_UP): ``0.065`` in ``decimal(5,4)`` is ``0.0650``."""
    schema = _schema(("d", SparkDecimalType(5, 4)))
    rows = [(Decimal("0.065"),), (Decimal("1"),), (Decimal("0.00005"),), (None,)]
    actual = engine.build_df(rows, schema).select(engine.functions.col("d").cast(engine.dtype(_STR)).alias("d"))
    expected = spark.createDataFrame(rows, schema).select(spark_col("d").cast("string").alias("d"))
    assert_matches_spark(actual, expected, engine)

    # Malformed: a value wider than the field's precision is rejected.
    too_wide = [(Decimal("123.456"),)]
    with pytest.raises(Exception):
        spark.createDataFrame(too_wide, schema).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(too_wide, schema))


@pytest.mark.feature("dataframe.create.empty_with_schema")
def test_create_empty_with_schema_against_spark(engine, spark):
    """An empty frame keeps the declared columns and types, and expressions over it stay typed and empty."""
    F = engine.functions
    schema = _schema(("id", _LONG), ("name", _STR), ("amt", SparkDecimalType(10, 2)))
    actual = engine.build_df([], schema)
    expected = spark.createDataFrame([], schema)

    assert actual.columns == expected.columns
    assert actual.dtypes == expected.dtypes
    assert actual.count() == 0 == expected.count()
    assert_matches_spark(actual, expected, engine)

    actual_projected = actual.select((F.col("id") + 1).alias("next_id"), F.col("name").isNull().alias("no_name"))
    expected_projected = expected.select((SF.col("id") + 1).alias("next_id"), SF.col("name").isNull().alias("no_name"))
    assert actual_projected.dtypes == expected_projected.dtypes
    assert_matches_spark(actual_projected, expected_projected, engine)

    # Malformed: a row narrower than the schema is rejected.
    with pytest.raises(Exception):
        spark.createDataFrame([(1,)], schema).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df([(1,)], schema))


@pytest.mark.feature("dataframe.create.from_dicts")
def test_create_from_dicts_against_spark(engine, spark):
    """Dict rows are matched to the schema by key: key order is irrelevant and a missing key is null."""
    schema = _schema(("id", _LONG), ("name", _STR), ("score", SparkDoubleType()))
    rows = [{"id": 1, "name": "a", "score": 1.5}, {"score": 2.5, "name": "b", "id": 2}, {"id": 3}]
    actual = engine.build_df(rows, schema)
    expected = spark.createDataFrame(rows, schema)
    assert actual.columns == expected.columns
    assert_matches_spark(actual, expected, engine)

    # Malformed: a dict value that does not fit its field's type is rejected.
    bad_rows = [{"id": "not-a-long", "name": "a", "score": 1.0}]
    with pytest.raises(Exception):
        spark.createDataFrame(bad_rows, schema).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(bad_rows, schema))


@pytest.mark.feature("dataframe.create.names_only_schema_infers_types")
def test_create_names_only_schema_infers_types_against_spark(engine, spark):
    """``createDataFrame(rows, ["a", ...])`` names the columns and infers bigint/string/double/boolean/date."""
    names = ["id", "name", "score", "flag", "day"]
    rows = [(1, "a", 1.5, True, date(2024, 1, 1)), (2, None, None, False, None), (3, "c", 3.0, None, date(2024, 3, 1))]
    actual = engine.create_df(rows, names)
    expected = spark.createDataFrame(rows, names)

    assert actual.columns == expected.columns
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)

    # A bigint mixed with a string merges to string (the bigint is stringified).
    mixed = [(1, "a"), ("two", "b")]
    actual_mixed = engine.create_df(mixed, ["id", "name"])
    expected_mixed = spark.createDataFrame(mixed, ["id", "name"])
    assert actual_mixed.dtypes == expected_mixed.dtypes
    assert_matches_spark(actual_mixed, expected_mixed, engine)

    # Malformed: a column whose values have types that do not merge (bigint vs double) cannot be inferred.
    conflicting = [(1, "a"), (1.5, "b")]
    with pytest.raises(Exception, match="CANNOT_MERGE_TYPE"):
        spark.createDataFrame(conflicting, ["id", "name"]).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.create_df(conflicting, ["id", "name"]))


@pytest.mark.feature("dataframe.create.value_type_mismatch_raises")
@pytest.mark.parametrize(
    "dtype, good, bad",
    [
        (_LONG, 1, "1"),
        (_LONG, 2, 1.5),
        (_LONG, 3, True),
        (SparkDoubleType(), 1.5, "1.5"),
        (SparkDoubleType(), 2.5, 1),
        (SparkBooleanType(), True, "true"),
        (SparkDateType(), date(2024, 1, 1), "2024-01-01"),
    ],
    ids=[
        "long_from_str",
        "long_from_float",
        "long_from_bool",
        "double_from_str",
        "double_from_int",
        "bool_from_str",
        "date_from_str",
    ],
)
def test_create_value_type_mismatch_raises(engine, spark, dtype, good, bad):
    """``createDataFrame`` verifies each value against its field type instead of coercing it.

    Only a string field is lenient: Spark 4 stringifies any value placed in it (``True`` becomes ``"true"``).
    """
    schema = _schema(("id", _LONG), ("v", dtype))
    good_rows = [(1, good), (2, None)]
    assert_matches_spark(engine.build_df(good_rows, schema), spark.createDataFrame(good_rows, schema), engine)

    bad_rows = [(1, good), (2, bad)]
    with pytest.raises(Exception, match="FIELD_DATA_TYPE_UNACCEPTABLE"):
        spark.createDataFrame(bad_rows, schema).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(bad_rows, schema))

    # The same values in a string field are stringified rather than rejected.
    str_schema = _schema(("id", _LONG), ("v", _STR))
    assert_matches_spark(engine.build_df(bad_rows, str_schema), spark.createDataFrame(bad_rows, str_schema), engine)


@pytest.mark.feature("dataframe.create.string_field_python_str")
def test_create_string_field_stringifies_like_python_against_spark(engine, spark):
    """A non-string value in a string field is Python's ``str`` of it (bools lowercased), not a SQL cast."""
    values = [
        float("nan"),
        float("inf"),
        1e20,
        1.5e-7,
        -0.0,
        True,
        10**20,
        Decimal("1.50"),
        date(2024, 1, 2),
        datetime(2024, 1, 2, 3, 4, 5, 600),
        b"ab",
        [1, 2],
        {"k": 1},
    ]
    schema = _schema(("i", _LONG), ("s", _STR))
    rows = [(i, v) for i, v in enumerate(values)]
    assert_matches_spark(engine.build_df(rows, schema), spark.createDataFrame(rows, schema), engine)

    # Malformed: the lenient string field does not extend to other types.
    with pytest.raises(Exception, match="FIELD_DATA_TYPE_UNACCEPTABLE"):
        spark.createDataFrame([(1, "x")], _schema(("i", _LONG), ("s", _LONG))).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df([(1, "x")], _schema(("i", _LONG), ("s", _LONG))))


@pytest.mark.feature("dataframe.create.empty_inference_raises")
def test_create_empty_without_types_raises_like_spark(engine, spark):
    """With no rows and only column names there is nothing to infer types from."""
    with pytest.raises(Exception, match="CANNOT_INFER_EMPTY_SCHEMA"):
        spark.createDataFrame([], ["a", "b"])
    with pytest.raises(Exception):
        engine.to_records(engine.create_df([], ["a", "b"]))

    # Valid: the same empty input with a typed schema builds an empty frame.
    schema = _schema(("a", _LONG), ("b", _STR))
    assert_matches_spark(engine.create_df([], schema), spark.createDataFrame([], schema), engine)


def _pandas_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i": [1, 2, 3],
            "f": [1.5, float("nan"), None],
            "s": ["a", None, "c"],
            "b": [True, False, True],
            "t": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-02 00:00:00", None]),
            "n": pd.array([1, None, 3], dtype="Int64"),
            "o": [1, "x", None],
            "d": [date(2024, 1, 1), None, date(2024, 3, 1)],
            "dec": [Decimal("1.50"), None, Decimal("-2")],
            "l": [[1, 2], None, []],
        }
    )


@pytest.mark.feature("dataframe.create.from_pandas")
def test_create_from_pandas_against_spark(engine, spark):
    """A pandas frame goes through PySpark's ``to_records`` path: names become the schema, types are inferred.

    pandas' own representation leaks through, as in Spark: a missing value in a ``str`` column is the string
    ``"nan"``, nullable ``Int64`` becomes ``double`` and a mixed ``object`` column becomes ``string``.
    """
    frame = _pandas_frame()
    actual = engine.create_df(frame, None)
    expected = spark.createDataFrame(frame)
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)

    subset = frame[["i", "s"]]
    renamed = engine.create_df(subset, ["x", "y"])
    expected_renamed = spark.createDataFrame(subset, ["x", "y"])
    assert renamed.dtypes == expected_renamed.dtypes
    assert_matches_spark(renamed, expected_renamed, engine)

    schema = _schema(("i", _LONG), ("s", _STR))
    assert_matches_spark(engine.create_df(subset, schema), spark.createDataFrame(subset, schema), engine)
    typed = engine.create_df(subset, "i bigint, s string")
    assert typed.dtypes == spark.createDataFrame(subset, "i bigint, s string").dtypes

    zoned = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 12:00:00+02:00", None])})
    assert_matches_spark(engine.create_df(zoned, None), spark.createDataFrame(zoned), engine)

    # Malformed: an all-null column, an empty frame and a value that does not fit its field are rejected.
    all_null = pd.DataFrame({"a": [None, None]})
    with pytest.raises(Exception, match="CANNOT_DETERMINE_TYPE"):
        spark.createDataFrame(all_null)
    with pytest.raises(Exception):
        engine.to_records(engine.create_df(all_null, None))

    empty = pd.DataFrame({"a": pd.Series([], dtype="int64")})
    with pytest.raises(Exception, match="CANNOT_INFER_EMPTY_SCHEMA"):
        spark.createDataFrame(empty)
    with pytest.raises(Exception):
        engine.to_records(engine.create_df(empty, None))

    mismatched = pd.DataFrame({"i": ["one"]})
    with pytest.raises(Exception, match="FIELD_DATA_TYPE_UNACCEPTABLE"):
        spark.createDataFrame(mismatched, _schema(("i", _LONG))).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.create_df(mismatched, _schema(("i", _LONG))))


def _arrow_table() -> pa.Table:
    return pa.table(
        {
            "i": pa.array([1, None], pa.int32()),
            "l": pa.array([10, 20], pa.int64()),
            "f": pa.array([1.5, None], pa.float32()),
            "s": ["a", None],
            "b": [True, None],
            "dec": pa.array([Decimal("1.25"), None], pa.decimal128(5, 2)),
            "t": pa.array([datetime(2024, 1, 1, 10), None], pa.timestamp("us")),
            "tz": pa.array([datetime(2024, 1, 1, 12), None], pa.timestamp("us", tz="+02:00")),
            "d": pa.array([date(2024, 1, 1), None]),
            "arr": pa.array([[1, 2], None]),
            "m": pa.array([[("k", 1)], None], pa.map_(pa.string(), pa.int64())),
            "st": pa.array([{"x": 1, "y": "a"}, None]),
        }
    )


@pytest.mark.feature("dataframe.create.from_arrow")
def test_create_from_arrow_against_spark(engine, spark):
    """An Arrow table keeps its Arrow types (``int32`` is ``int``); names or a schema may be given instead."""
    table = _arrow_table()
    actual = engine.create_df(table, None)
    expected = spark.createDataFrame(table)
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)

    subset = table.select(["i", "s"])
    renamed = engine.create_df(subset, ["x", "y"])
    expected_renamed = spark.createDataFrame(subset, ["x", "y"])
    assert renamed.dtypes == expected_renamed.dtypes
    assert_matches_spark(renamed, expected_renamed, engine)

    schema = _schema(("i", _LONG), ("s", _STR))
    widened = engine.create_df(subset, schema)
    expected_widened = spark.createDataFrame(subset, schema)
    assert widened.dtypes == expected_widened.dtypes
    assert_matches_spark(widened, expected_widened, engine)

    # Malformed: a string that is not a number does not fit a bigint field.
    bad = pa.table({"i": ["not-a-number"]})
    with pytest.raises(Exception):
        spark.createDataFrame(bad, _schema(("i", _LONG))).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.create_df(bad, _schema(("i", _LONG))))


@pytest.mark.feature("dataframe.create.from_arrow_stream")
def test_create_from_arrow_stream_frame_against_spark(engine, spark):
    """Any frame exporting the Arrow C stream (here polars) builds like the Arrow table it exports."""
    import polars as pl

    rows = [(1, "a", [1]), (2, None, None), (None, "c", [])]
    frame = pl.DataFrame(rows, schema=["x", "y", "z"], orient="row")
    actual = engine.create_df(frame, None)
    # Spark cannot ingest polars' ``string_view`` export, so its side gets the same rows and Arrow-derived types.
    expected = spark.createDataFrame(rows, "x bigint, y string, z array<bigint>")
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)

    # Malformed: a value that does not fit the requested field type is rejected.
    bad = pl.DataFrame({"x": ["not-a-number"]})
    with pytest.raises(Exception):
        spark.createDataFrame([("not-a-number",)], _schema(("x", _LONG))).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.create_df(bad, _schema(("x", _LONG))))


@pytest.mark.feature("dataframe.create.ddl_schema")
@pytest.mark.parametrize("ddl", ["a int, b string", "a: int, b: string", "a:int,b:string", "struct<a:int,b:string>"])
def test_create_ddl_schema_against_spark(engine, spark, ddl):
    """PySpark's ``_parse_datatype_string`` takes a field list with or without ``name: type`` colons."""
    rows = [(1, "x"), (None, None)]
    actual = engine.create_df(rows, ddl)
    expected = spark.createDataFrame(rows, ddl)
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)

    # Malformed: an unparseable DDL is rejected.
    with pytest.raises(Exception):
        spark.createDataFrame(rows, "a int,")
    with pytest.raises(Exception):
        engine.to_records(engine.create_df(rows, "a int,"))


@pytest.mark.feature("dataframe.create.single_datatype_schema")
def test_create_single_datatype_schema_against_spark(engine, spark):
    """A single ``DataType`` (or a type DDL) schema makes one ``value`` column holding each item."""
    cases = [
        ([1, 2, None], SparkIntegerType()),
        (["a", None], _STR),
        ([[1, 2], None], SparkArrayType(SparkIntegerType())),
    ]
    for data, dtype in cases:
        actual = engine.create_df(data, engine.dtype(dtype))
        expected = spark.createDataFrame(data, dtype)
        assert actual.dtypes == expected.dtypes
        assert_matches_spark(actual, expected, engine)

    actual_ddl = engine.create_df([1, 2], "int")
    expected_ddl = spark.createDataFrame([1, 2], "int")
    assert actual_ddl.dtypes == expected_ddl.dtypes
    assert_matches_spark(actual_ddl, expected_ddl, engine)

    # Malformed: a value of another type, or a tuple row, does not fit the single field.
    for bad in (["a"], [(1,)]):
        with pytest.raises(Exception, match="FIELD_DATA_TYPE_UNACCEPTABLE"):
            spark.createDataFrame(bad, SparkIntegerType()).collect()
        with pytest.raises(Exception):
            engine.to_records(engine.create_df(bad, engine.dtype(SparkIntegerType())))


# ----------------------------------------------------------------------------- #
# distinct / dropDuplicates
# ----------------------------------------------------------------------------- #

_DEDUP_SCHEMA = _schema(("id", _LONG), ("name", _STR), ("score", SparkDoubleType()))
_DEDUP_ROWS = [
    (1, "a", 1.0),
    (1, "a", 1.0),
    (1, "b", 1.0),
    (2, None, None),
    (2, None, None),
    (3, "c", float("nan")),
    (3, "c", float("nan")),
    (None, None, None),
    (None, None, None),
]


def _map_frames(engine, spark):
    """A frame with a map column: Spark rejects set operations (dedup) on map-typed columns."""
    F = engine.functions
    schema = _schema(("k", _STR), ("v", _LONG))
    rows = [("a", 1), ("a", 1)]
    sf = engine.build_df(rows, schema).select(F.create_map(F.col("k"), F.col("v")).alias("m"))
    sp = spark.createDataFrame(rows, schema).select(SF.create_map(SF.col("k"), SF.col("v")).alias("m"))
    return sf, sp


@pytest.mark.feature("dataframe.distinct.basic")
def test_distinct_against_spark(engine, spark):
    """Whole-row dedup: nulls equal nulls and NaN equals NaN."""
    actual = engine.build_df(_DEDUP_ROWS, _DEDUP_SCHEMA).distinct()
    expected = spark.createDataFrame(_DEDUP_ROWS, _DEDUP_SCHEMA).distinct()
    assert actual.count() == expected.count()
    assert_matches_spark(actual, expected, engine)

    # Malformed: a map column cannot be deduplicated.
    sf_map, sp_map = _map_frames(engine, spark)
    with pytest.raises(Exception, match="SET_OPERATION_ON_MAP_TYPE"):
        sp_map.distinct().collect()
    with pytest.raises(Exception):
        engine.to_records(sf_map.distinct())


@pytest.mark.feature("dataframe.drop_duplicates.all_columns")
@pytest.mark.parametrize("method", ["dropDuplicates", "drop_duplicates"])
def test_drop_duplicates_all_columns_against_spark(engine, spark, method):
    actual = getattr(engine.build_df(_DEDUP_ROWS, _DEDUP_SCHEMA), method)()
    expected = getattr(spark.createDataFrame(_DEDUP_ROWS, _DEDUP_SCHEMA), method)()
    assert actual.count() == expected.count()
    assert_matches_spark(actual, expected, engine)

    # Malformed: a map column cannot be deduplicated.
    sf_map, sp_map = _map_frames(engine, spark)
    with pytest.raises(Exception):
        getattr(sp_map, method)().collect()
    with pytest.raises(Exception):
        engine.to_records(getattr(sf_map, method)())


@pytest.mark.feature("dataframe.drop_duplicates.subset")
@pytest.mark.parametrize("subset", [["id"], ["id", "name"], ["name"]])
def test_drop_duplicates_subset_against_spark(engine, spark, subset):
    """One row survives per distinct subset key (null keys group together); which row is unspecified."""
    sf = engine.build_df(_DEDUP_ROWS, _DEDUP_SCHEMA)
    sp = spark.createDataFrame(_DEDUP_ROWS, _DEDUP_SCHEMA)
    actual = sf.dropDuplicates(subset)
    expected = sp.dropDuplicates(subset)
    assert actual.columns == expected.columns
    assert actual.count() == expected.count()
    assert_matches_spark(actual.select(subset), expected.select(subset), engine)

    # Malformed: a subset column that does not exist is rejected.
    with pytest.raises(Exception, match='Cannot resolve column name "nope"'):
        sp.dropDuplicates(["nope"]).collect()
    with pytest.raises(Exception):
        engine.to_records(sf.dropDuplicates(["nope"]))


# ----------------------------------------------------------------------------- #
# filter with a SQL predicate string
# ----------------------------------------------------------------------------- #

_SQL_FILTER_SCHEMA = _schema(("rn", _LONG), ("name", _STR), ("other", _STR))
_SQL_FILTER_ROWS = [(1, "a", "x"), (2, "b", None), (1, None, None), (3, "c", "z"), (None, "d", "w")]


@pytest.mark.feature("dataframe.filter.sql_string_comparison")
@pytest.mark.parametrize(
    "predicate",
    [
        "rn = 1",
        "rn <> 1",
        "rn != 2",
        "rn >= 2",
        "rn < 3",
        "name = 'a'",
        "name IS NULL",
        "name is not null",
        "rn = 1 AND name IS NOT NULL",
        "rn = 1 OR name IS NULL",
        "name is null and other is null",
        "(rn > 1 AND rn < 3) OR name = 'c'",
    ],
)
def test_filter_sql_string_comparison_against_spark(engine, spark, predicate):
    """``df.filter("rn = 1")``-style SQL predicates; a null comparison drops the row."""
    actual = engine.build_df(_SQL_FILTER_ROWS, _SQL_FILTER_SCHEMA).filter(predicate)
    expected = spark.createDataFrame(_SQL_FILTER_ROWS, _SQL_FILTER_SCHEMA).filter(predicate)
    assert_matches_spark(actual, expected, engine)

    where_result = engine.build_df(_SQL_FILTER_ROWS, _SQL_FILTER_SCHEMA).where(predicate)
    assert_matches_spark(where_result, expected, engine)


@pytest.mark.feature("dataframe.filter.sql_string_comparison")
@pytest.mark.parametrize(
    "predicate, spark_error",
    [
        ("rn = = 1", "PARSE_SYNTAX_ERROR"),
        ("rn =", "PARSE_SYNTAX_ERROR"),
        ("nope = 1", "UNRESOLVED_COLUMN"),
        ("name = 1", "CAST_INVALID_INPUT"),
    ],
    ids=["double_operator", "missing_operand", "unknown_column", "ansi_string_to_int"],
)
def test_filter_sql_string_malformed_raises(engine, spark, predicate, spark_error):
    """A malformed predicate, unknown column, or (ANSI) non-numeric string compared to an int raises."""
    with pytest.raises(Exception, match=spark_error):
        spark.createDataFrame(_SQL_FILTER_ROWS, _SQL_FILTER_SCHEMA).filter(predicate).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(_SQL_FILTER_ROWS, _SQL_FILTER_SCHEMA).filter(predicate))


# ----------------------------------------------------------------------------- #
# __getattr__
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.getattr.column")
def test_getattr_column_against_spark(engine, spark):
    """``df.age`` is the column ``age``: usable in select, filter and expressions."""
    schema = _schema(("age", _LONG), ("name", _STR))
    rows = [(2, "Alice"), (5, "Bob"), (None, "Carol")]
    sf = engine.build_df(rows, schema)
    sp = spark.createDataFrame(rows, schema)

    actual = sf.filter(sf.age > 1).select(sf.name, (sf.age + 1).alias("next_age"))
    expected = sp.filter(sp.age > 1).select(sp.name, (sp.age + 1).alias("next_age"))
    assert_matches_spark(actual, expected, engine)

    # Malformed: a missing column is an AttributeError (so ``hasattr`` is False), not a lazy column.
    with pytest.raises(AttributeError):
        sp.nope
    with pytest.raises(AttributeError):
        sf.nope
    assert hasattr(sf, "nope") is hasattr(sp, "nope") is False


# ----------------------------------------------------------------------------- #
# groupBy edge cases
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.groupby.max_strings_and_nulls")
def test_groupby_max_strings_and_nulls_against_spark(engine, spark):
    """max/min over strings are lexicographic and skip nulls; an all-null group yields null; null keys group."""
    F = engine.functions
    schema = _schema(("g", _STR), ("s", _STR))
    rows = [
        ("A", "apple"),
        ("A", "Banana"),
        ("A", None),
        ("A", "banana"),
        ("B", None),
        ("B", None),
        (None, "zeta"),
        (None, "alpha"),
        ("C", ""),
        ("C", "a"),
    ]
    actual = engine.build_df(rows, schema).groupBy("g").agg(F.max("s").alias("mx"), F.min("s").alias("mn"))
    expected = spark.createDataFrame(rows, schema).groupBy("g").agg(SF.max("s").alias("mx"), SF.min("s").alias("mn"))
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)

    # Malformed: aggregating a column that does not exist is rejected.
    with pytest.raises(Exception, match="UNRESOLVED_COLUMN"):
        spark.createDataFrame(rows, schema).groupBy("g").agg(SF.max("nope")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(rows, schema).groupBy("g").agg(F.max("nope")))


@pytest.mark.feature("dataframe.groupby.sum_decimal_result_type")
def test_groupby_sum_decimal_result_type_against_spark(engine, spark):
    """sum(decimal(p,s)) is decimal(p+10,s) at the input scale; sum(bigint) stays bigint."""
    F = engine.functions
    schema = _schema(("g", _STR), ("d", SparkDecimalType(10, 2)), ("n", _LONG))
    rows = [
        ("A", Decimal("1.10"), 1),
        ("A", Decimal("2.25"), 2),
        ("A", None, None),
        ("B", Decimal("99999999.99"), 3),
        ("B", Decimal("0.01"), 4),
        ("C", None, None),
    ]
    actual = engine.build_df(rows, schema).groupBy("g").agg(F.sum("d").alias("sd"), F.sum("n").alias("sn"))
    expected = spark.createDataFrame(rows, schema).groupBy("g").agg(SF.sum("d").alias("sd"), SF.sum("n").alias("sn"))
    assert actual.dtypes == expected.dtypes
    assert_matches_spark(actual, expected, engine)

    # The oracle compares decimals as floats: compare the exact text (scale included) too.
    str_type = engine.dtype(_STR)
    actual_text = actual.select("g", F.col("sd").cast(str_type).alias("sd"))
    expected_text = expected.select("g", SF.col("sd").cast("string").alias("sd"))
    assert_matches_spark(actual_text, expected_text, engine)

    # Malformed: a decimal(38,0) sum that exceeds 38 digits overflows (ANSI raises instead of returning null).
    wide_schema = _schema(("g", _STR), ("d", SparkDecimalType(38, 0)))
    wide_rows = [("A", Decimal("9" * 38)), ("A", Decimal("9" * 38))]
    with pytest.raises(Exception):
        spark.createDataFrame(wide_rows, wide_schema).groupBy("g").agg(SF.sum("d")).collect()
    with pytest.raises(Exception):
        engine.to_records(engine.build_df(wide_rows, wide_schema).groupBy("g").agg(F.sum("d")))


# ----------------------------------------------------------------------------- #
# isEmpty
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.is_empty")
def test_is_empty_against_spark(engine, spark):
    F = engine.functions
    schema = _schema(("id", _LONG), ("name", _STR))
    rows = [(1, "a"), (2, "b")]
    sf = engine.build_df(rows, schema)
    sp = spark.createDataFrame(rows, schema)
    cases = [
        (sf, sp),
        (engine.build_df([], schema), spark.createDataFrame([], schema)),
        (sf.filter(F.col("id") > 5), sp.filter(SF.col("id") > 5)),
        (engine.build_df([(None, None)], schema), spark.createDataFrame([(None, None)], schema)),
    ]
    assert [a.isEmpty() for a, _ in cases] == [e.isEmpty() for _, e in cases] == [False, True, True, False]

    # Malformed: isEmpty evaluates the plan, so an ANSI cast failure in a filter surfaces.
    bad = [(1, "abc")]
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        spark.createDataFrame(bad, schema).filter(SF.col("name") == 1).isEmpty()
    with pytest.raises(Exception):
        engine.build_df(bad, schema).filter(F.col("name") == 1).isEmpty()


# ----------------------------------------------------------------------------- #
# join on a Column condition / left_anti
# ----------------------------------------------------------------------------- #

_QUAL_LEFT_SCHEMA = _schema(("k1", _LONG), ("x", _STR))
_QUAL_RIGHT_SCHEMA = _schema(("k2", _LONG), ("x", _STR))
_QUAL_LEFT_ROWS = [(1, "l1"), (2, "l2"), (3, "l3"), (None, "lnull")]
_QUAL_RIGHT_ROWS = [(2, "r2"), (3, "r3"), (3, "r3b"), (4, "r4"), (None, "rnull")]


@pytest.mark.feature("dataframe.join.column_condition_qualified_refs")
@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
@pytest.mark.parametrize("form", ["getitem", "getattr"])
def test_join_column_condition_qualified_refs_against_spark(engine, spark, how, form):
    """``a.join(b, a["k1"] == b["k2"])`` keeps both ``x`` columns; ``a["x"]`` / ``b["x"]`` pick a side."""
    sa, sb = spark.createDataFrame(_QUAL_LEFT_ROWS, _QUAL_LEFT_SCHEMA), spark.createDataFrame(
        _QUAL_RIGHT_ROWS, _QUAL_RIGHT_SCHEMA
    )
    fa, fb = engine.build_df(_QUAL_LEFT_ROWS, _QUAL_LEFT_SCHEMA), engine.build_df(_QUAL_RIGHT_ROWS, _QUAL_RIGHT_SCHEMA)

    def condition(a, b):
        return a["k1"] == b["k2"] if form == "getitem" else a.k1 == b.k2

    spark_joined = sa.join(sb, condition(sa, sb), how)
    sf_joined = fa.join(fb, condition(fa, fb), how)
    assert sf_joined.columns == spark_joined.columns == ["k1", "x", "k2", "x"]

    expected = spark_joined.select("k1", "k2", sa["x"].alias("ax"), sb["x"].alias("bx"))
    actual = sf_joined.select("k1", "k2", fa["x"].alias("ax"), fb["x"].alias("bx"))
    assert_matches_spark(actual, expected, engine)

    # Malformed: a non-boolean join condition is rejected.
    with pytest.raises(Exception, match="JOIN_CONDITION_IS_NOT_BOOLEAN_TYPE"):
        sa.join(sb, sa["k1"] + sb["k2"], how).collect()
    with pytest.raises(Exception):
        engine.to_records(fa.join(fb, fa["k1"] + fb["k2"], how))


@pytest.mark.feature("dataframe.join.ambiguous_reference_raises")
@pytest.mark.parametrize("op", ["select_str", "select_col", "filter", "with_column"])
def test_join_ambiguous_reference_raises(engine, spark, op):
    """After a Column-condition join with ``x`` on both sides, an unqualified ``x`` is ambiguous."""
    F = engine.functions
    sa, sb = spark.createDataFrame(_QUAL_LEFT_ROWS, _QUAL_LEFT_SCHEMA), spark.createDataFrame(
        _QUAL_RIGHT_ROWS, _QUAL_RIGHT_SCHEMA
    )
    fa, fb = engine.build_df(_QUAL_LEFT_ROWS, _QUAL_LEFT_SCHEMA), engine.build_df(_QUAL_RIGHT_ROWS, _QUAL_RIGHT_SCHEMA)
    spark_joined = sa.join(sb, sa["k1"] == sb["k2"])
    sf_joined = fa.join(fb, fa["k1"] == fb["k2"])

    # Valid: unambiguous names still resolve unqualified.
    assert_matches_spark(sf_joined.select("k1", "k2"), spark_joined.select("k1", "k2"), engine)

    def ambiguous(joined, funcs):
        if op == "select_str":
            return joined.select("x")
        if op == "select_col":
            return joined.select(funcs.col("x"))
        if op == "filter":
            return joined.filter(funcs.col("x") == "l2")
        return joined.withColumn("y", funcs.col("x").isNull())

    with pytest.raises(Exception, match="AMBIGUOUS_REFERENCE"):
        ambiguous(spark_joined, SF).collect()
    with pytest.raises(Exception):
        engine.to_records(ambiguous(sf_joined, F))


@pytest.mark.feature("dataframe.join.left_anti")
@pytest.mark.parametrize("how", ["left_anti", "leftanti", "anti"])
@pytest.mark.parametrize("on_form", ["using", "condition"])
def test_join_left_anti_against_spark(engine, spark, how, on_form):
    """left_anti keeps left rows with no match (a null key never matches) and only the left columns."""
    left_schema = _schema(("id", _LONG), ("left_val", _STR))
    right_schema = _schema(("id", _LONG), ("right_val", _STR))
    left_rows = [(1, "a"), (2, "b"), (2, "b2"), (3, "c"), (None, "n")]
    right_rows = [(2, "x"), (2, "x2"), (4, "z"), (None, "rn")]
    sl, sr = spark.createDataFrame(left_rows, left_schema), spark.createDataFrame(right_rows, right_schema)
    fl, fr = engine.build_df(left_rows, left_schema), engine.build_df(right_rows, right_schema)

    if on_form == "using":
        expected, actual = sl.join(sr, "id", how), fl.join(fr, "id", how)
    else:
        expected, actual = sl.join(sr, sl["id"] == sr["id"], how), fl.join(fr, fl["id"] == fr["id"], how)
    assert actual.columns == expected.columns == ["id", "left_val"]
    assert_matches_spark(actual, expected, engine)

    # Malformed: an unknown join type is rejected.
    with pytest.raises(Exception, match="UNSUPPORTED_JOIN_TYPE"):
        sl.join(sr, "id", "anti_left").collect()
    with pytest.raises(Exception):
        engine.to_records(fl.join(fr, "id", "anti_left"))


# ----------------------------------------------------------------------------- #
# toPandas
# ----------------------------------------------------------------------------- #

_PANDAS_SCHEMA = _schema(
    ("l", _LONG),
    ("d", SparkDoubleType()),
    ("s", _STR),
    ("b", SparkBooleanType()),
    ("ts", SparkTimestampType()),
    ("dt", SparkDateType()),
    ("dec", SparkDecimalType(10, 2)),
)
_PANDAS_ROWS = [
    (1, 1.5, "a", True, datetime(2024, 1, 2, 3, 4, 5), date(2024, 1, 2), Decimal("1.50")),
    (2, -2.0, "b", False, datetime(2023, 12, 31, 23, 59, 59), date(2023, 12, 31), Decimal("-2.25")),
]
_PANDAS_NULL_ROW = (None, None, None, None, None, None, None)


@pytest.mark.feature("dataframe.to_pandas.dtypes")
@pytest.mark.parametrize("with_nulls", [False, True], ids=["no_nulls", "with_nulls"])
def test_to_pandas_dtypes_against_spark(engine, spark, with_nulls):
    """Nulls widen bigint to float64 (NaN) and bool to object; timestamps are datetime64, dates/decimals object."""
    rows = _PANDAS_ROWS + ([_PANDAS_NULL_ROW] if with_nulls else [])
    actual = engine.build_df(rows, _PANDAS_SCHEMA).toPandas()
    expected = spark.createDataFrame(rows, _PANDAS_SCHEMA).toPandas()

    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.dtypes.astype(str).to_dict() == expected.dtypes.astype(str).to_dict()


@pytest.mark.feature("dataframe.to_pandas.dtypes")
def test_to_pandas_empty_and_malformed_against_spark(engine, spark):
    """An empty frame keeps Spark's per-type dtypes; ``toPandas`` evaluates the plan, so ANSI failures raise."""
    actual = engine.build_df([], _PANDAS_SCHEMA).toPandas()
    expected = spark.createDataFrame([], _PANDAS_SCHEMA).toPandas()
    assert list(actual.columns) == list(expected.columns)
    assert actual.dtypes.astype(str).to_dict() == expected.dtypes.astype(str).to_dict()

    F = engine.functions
    schema = _schema(("s", _STR))
    bad = [("1",), ("abc",)]
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        spark.createDataFrame(bad, schema).select(SF.col("s").cast("bigint")).toPandas()
    with pytest.raises(Exception):
        engine.build_df(bad, schema).select(F.col("s").cast(engine.dtype(_LONG))).toPandas()


# ----------------------------------------------------------------------------- #
# union (positional)
# ----------------------------------------------------------------------------- #


@pytest.mark.feature("dataframe.union.positional")
@pytest.mark.parametrize("method", ["union", "unionAll"])
def test_union_positional_against_spark(engine, spark, method):
    """union matches columns by position (not name), keeps the left names and keeps duplicates (UNION ALL)."""
    left_schema = _schema(("a", _LONG), ("b", _LONG), ("c", _STR))
    right_schema = _schema(("b", _LONG), ("a", _LONG), ("z", _STR))
    left_rows = [(1, 10, "x"), (2, 20, None)]
    right_rows = [(100, 1000, "y"), (1, 10, "x")]
    actual = getattr(engine.build_df(left_rows, left_schema), method)(engine.build_df(right_rows, right_schema))
    expected = getattr(spark.createDataFrame(left_rows, left_schema), method)(
        spark.createDataFrame(right_rows, right_schema)
    )
    assert actual.columns == expected.columns == ["a", "b", "c"]
    assert actual.count() == expected.count() == 4
    assert_matches_spark(actual, expected, engine)

    # Malformed: positionally paired columns of incompatible types are rejected.
    bad_schema = _schema(("a", SparkBooleanType()), ("b", _LONG), ("c", _STR))
    bad_rows = [(True, 1, "q")]
    with pytest.raises(Exception, match="INCOMPATIBLE_COLUMN_TYPE"):
        getattr(spark.createDataFrame(left_rows, left_schema), method)(
            spark.createDataFrame(bad_rows, bad_schema)
        ).collect()
    with pytest.raises(Exception):
        engine.to_records(
            getattr(engine.build_df(left_rows, left_schema), method)(engine.build_df(bad_rows, bad_schema))
        )


@pytest.mark.feature("dataframe.union.column_count_mismatch_raises")
@pytest.mark.parametrize("method", ["union", "unionAll"])
def test_union_column_count_mismatch_raises(engine, spark, method):
    """Positional union of inputs with different column counts is rejected."""
    two = _schema(("a", _LONG), ("b", _LONG))
    three = _schema(("a", _LONG), ("b", _LONG), ("c", _LONG))
    rows2, rows3 = [(1, 2)], [(3, 4, 5)]

    # Valid: equal widths union fine.
    actual = getattr(engine.build_df(rows2, two), method)(engine.build_df([(6, 7)], two))
    expected = getattr(spark.createDataFrame(rows2, two), method)(spark.createDataFrame([(6, 7)], two))
    assert_matches_spark(actual, expected, engine)

    for left, right in [((rows2, two), (rows3, three)), ((rows3, three), (rows2, two))]:
        with pytest.raises(Exception, match="NUM_COLUMNS_MISMATCH"):
            getattr(spark.createDataFrame(*left), method)(spark.createDataFrame(*right)).collect()
        with pytest.raises(Exception):
            engine.to_records(getattr(engine.build_df(*left), method)(engine.build_df(*right)))
