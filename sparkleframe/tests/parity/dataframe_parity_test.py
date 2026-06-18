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

import pyspark.sql.functions as SF
import pytest
from pyspark.sql.functions import col as spark_col
from pyspark.sql.types import FloatType as SparkFloatType
from pyspark.sql.types import LongType as SparkLongType
from pyspark.sql.types import StringType as SparkStringType
from pyspark.sql.types import StructField as SparkStructField
from pyspark.sql.types import StructType as SparkStructType

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
    actual = sample.sparkle(engine).select(F.col("name"), F.col("salary") * 1.1)
    expected = sample.spark(spark).select(spark_col("name"), (spark_col("salary") * 1.1).alias("salary"))
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
        (
            "cross",
            None,
            [
                (1, "a", 2, "x"),
                (1, "a", 3, "y"),
                (1, "a", 4, "z"),
                (2, "b", 2, "x"),
                (2, "b", 3, "y"),
                (2, "b", 4, "z"),
                (3, "c", 2, "x"),
                (3, "c", 3, "y"),
                (3, "c", 4, "z"),
            ],
            ["id", "left_val", "id_right", "right_val"],
        ),
    ],
)
@pytest.mark.parametrize("on_format", ["str", "col", "list_str", "list_col"])
def test_polars_joins(engine, spark, how, on_input, expected_rows, expected_cols, on_format):
    F = engine.functions

    left = engine.build_df(_JOIN_LEFT_ROWS, _JOIN_LEFT_SCHEMA)
    right = engine.build_df(_JOIN_RIGHT_ROWS, _JOIN_RIGHT_SCHEMA)

    if on_input is None:
        on = None
    elif on_format == "str":
        on = on_input
    elif on_format == "col":
        on = F.col(on_input)
    elif on_format == "list_str":
        on = [on_input]
    elif on_format == "list_col":
        on = [F.col(on_input)]
    else:
        raise ValueError("Invalid on_format")

    result = left.join(right, on=on, how=how)
    cols = sorted(result.columns)
    result = result.select(cols).orderBy(cols)

    # An outer join keyed on a Column expression (not a string name) does NOT coalesce
    # the join keys, so the right-only row's ``id`` stays null instead of 4.
    if how == "outer" and on_format in ("col", "list_col"):
        expected_rows = [(None,) + r[1:] if r[0] == 4 else r for r in expected_rows]

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
@pytest.mark.parametrize(
    "how,on_keys",
    [
        ("inner", (True, "id")),
        ("left", (True, "id")),
        ("outer", (True, "id")),
        ("inner", [(True, "id")]),
        ("left", [(True, "id")]),
        ("outer", [(True, "id")]),
    ],
)
def test_joins_with_duplicated_spark_keys(engine, spark, how, on_keys):
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

    # Spark rejects a Column-equality join on identically named keys; use the rename workaround.
    with pytest.raises(Exception):
        spark_left.join(spark_right, on=spark_on_keys, how=how)
    spark_right_renamed = spark_right.withColumnRenamed("id", "id_right")
    expected = spark_left.join(spark_right_renamed, SF.col("id") == SF.col("id_right"), how=how).drop("id_right")

    sf_left = engine.build_df(_JOIN_LEFT_ROWS, _JOIN_LEFT_SCHEMA)
    sf_right = engine.build_df(_JOIN_RIGHT_ROWS, _JOIN_RIGHT_SCHEMA)
    result = sf_left.join(sf_right, on=sf_on_keys, how=how)

    assert_matches_spark(
        result.select(sorted(result.columns)).orderBy("id"),
        expected.select(sorted(expected.columns)).orderBy("id"),
        engine,
    )


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


# ----------------------------------------------------------------------------- #
# unionByName
# ----------------------------------------------------------------------------- #


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
