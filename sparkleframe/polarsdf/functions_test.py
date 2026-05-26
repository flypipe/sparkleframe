import builtins
import json
import re
import uuid as std_uuid

import pandas as pd
import pandas.testing as pdt
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pyspark.sql.functions import abs as spark_abs
from pyspark.sql.functions import array_contains as spark_array_contains
from pyspark.sql.functions import asc as spark_asc
from pyspark.sql.functions import asc_nulls_first as spark_asc_nulls_first
from pyspark.sql.functions import asc_nulls_last as spark_asc_nulls_last
from pyspark.sql.functions import coalesce as spark_coalesce
from pyspark.sql.functions import col as spark_col
from pyspark.sql.functions import concat as spark_concat
from pyspark.sql.functions import count as spark_count
from pyspark.sql.functions import create_map as spark_create_map
from pyspark.sql.functions import current_date as spark_current_date
from pyspark.sql.functions import current_timestamp as spark_current_timestamp
from pyspark.sql.functions import date_format as spark_date_format
from pyspark.sql.functions import date_sub as spark_date_sub
from pyspark.sql.functions import datediff as spark_datediff
from pyspark.sql.functions import dense_rank as spark_dense_rank
from pyspark.sql.functions import desc as spark_desc
from pyspark.sql.functions import desc_nulls_first as spark_desc_nulls_first
from pyspark.sql.functions import desc_nulls_last as spark_desc_nulls_last
from pyspark.sql.functions import element_at as spark_element_at
from pyspark.sql.functions import explode_outer as spark_explode_outer
from pyspark.sql.functions import filter as spark_filter
from pyspark.sql.functions import first as spark_first
from pyspark.sql.functions import floor as spark_floor
from pyspark.sql.functions import from_json as spark_from_json
from pyspark.sql.functions import get_json_object as spark_get_json_object
from pyspark.sql.functions import greatest as spark_greatest
from pyspark.sql.functions import initcap as spark_initcap
from pyspark.sql.functions import isnan as spark_isnan
from pyspark.sql.functions import least as spark_least
from pyspark.sql.functions import length as spark_length
from pyspark.sql.functions import lit as spark_lit
from pyspark.sql.functions import lower as spark_lower
from pyspark.sql.functions import map_from_entries as spark_map_from_entries
from pyspark.sql.functions import map_keys as spark_map_keys
from pyspark.sql.functions import md5 as spark_md5
from pyspark.sql.functions import monotonically_increasing_id as spark_monotonically_increasing_id
from pyspark.sql.functions import months_between as spark_months_between
from pyspark.sql.functions import now as spark_now
from pyspark.sql.functions import nullif as spark_nullif
from pyspark.sql.functions import pow as spark_pow
from pyspark.sql.functions import rand as spark_rand
from pyspark.sql.functions import rank as spark_rank
from pyspark.sql.functions import regexp_replace as spark_regexp_replace
from pyspark.sql.functions import round as spark_round
from pyspark.sql.functions import row_number as spark_row_number
from pyspark.sql.functions import size as spark_size
from pyspark.sql.functions import sort_array as spark_sort_array
from pyspark.sql.functions import split as spark_split
from pyspark.sql.functions import struct as spark_struct
from pyspark.sql.functions import substring as spark_substring
from pyspark.sql.functions import to_date as spark_to_date
from pyspark.sql.functions import to_json as spark_to_json
from pyspark.sql.functions import to_timestamp as spark_to_timestamp
from pyspark.sql.functions import transform as spark_transform
from pyspark.sql.functions import trim as spark_trim
from pyspark.sql.functions import try_divide as spark_try_divide
from pyspark.sql.functions import try_element_at as spark_try_element_at
from pyspark.sql.functions import try_to_date as spark_try_to_date
from pyspark.sql.functions import try_to_timestamp as spark_try_to_timestamp
from pyspark.sql.functions import unix_millis as spark_unix_millis
from pyspark.sql.functions import when as spark_when
from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import DoubleType as SparkDoubleType
from pyspark.sql.types import IntegerType as SparkIntegerType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import StringType as SparkStringType
from pyspark.sql.types import StructField as SparkStructField
from pyspark.sql.types import StructType as SparkStructType
from pyspark.sql.window import Window as SparkWindow

from sparkleframe.polarsdf import Window
from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.functions import (
    abs,
    array,
    array_contains,
    asc,
    asc_nulls_first,
    asc_nulls_last,
    broadcast,
    coalesce,
    col,
    concat,
    count,
    create_map,
    current_date,
    current_timestamp,
    date_format,
    date_sub,
    datediff,
    dense_rank,
    desc,
    desc_nulls_first,
    desc_nulls_last,
    element_at,
    explode,
    filter,
    first,
    floor,
    from_json,
    get_json_object,
    greatest,
    initcap,
    isnan,
    least,
    length,
    lit,
    lower,
    map_from_entries,
    map_keys,
    md5,
    monotonically_increasing_id,
    months_between,
    now,
    nullif,
    pow,
    rand,
    rank,
    regexp_replace,
    round,
    row_number,
    size,
    sort_array,
    split,
    struct,
    substring,
    to_date,
    to_json,
    to_timestamp,
    transform,
    trim,
    try_divide,
    try_element_at,
    try_to_date,
    try_to_timestamp,
    uuid,
    when,
)
from sparkleframe.polarsdf.types import IntegerType, StringType, StructField, StructType
from sparkleframe.tests.pyspark_test import assert_pyspark_df_equal
from sparkleframe.tests.utils import assert_sparkle_spark_frame_are_equal, create_spark_df, spark_rows_from_dict

sample_data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}


@pytest.fixture
def sparkle_df():
    return DataFrame(pl.DataFrame(sample_data))


@pytest.fixture
def spark_df(spark):
    return spark.createDataFrame(pd.DataFrame(sample_data))


class TestFunctions:
    def test_when(self, spark, sparkle_df, spark_df):
        expr = when(col("a") > 2, "yes").otherwise("no")

        # Add the result column to the full Polars DataFrame
        result_spark_df = spark.createDataFrame(sparkle_df.withColumn("result", expr).toPandas())

        # Add result column to full Spark DataFrame
        expected_spark_df = spark_df.withColumn("result", spark_when(spark_col("a") > 2, "yes").otherwise("no"))

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_chained_when_boolean_output(self, spark):
        data = {"b": ["A", "B", "C", "D"], "c": ["b", "e", "g", "z"]}
        polars_df = DataFrame(pl.DataFrame(data))
        expr = (
            when((col("b") == "A") & (col("c").isin("A", "b", "c")), True)
            .when((col("b") == "B") & (col("c").isin("d", "e")), True)
            .when((col("b") == "C") & (col("c").isin("f", "g", "h", "i")), True)
            .otherwise(False)
        )

        result_df = polars_df.withColumn("result", expr)
        result_spark_df = spark.createDataFrame(result_df.df.to_dicts())

        # Expected result using PySpark chained when()
        expected_df = spark.createDataFrame(
            spark_rows_from_dict(data),
            list(data.keys()),
        ).withColumn(
            "result",
            spark_when((spark_col("b") == "A") & (spark_col("c").isin("A", "b", "c")), True)
            .when((spark_col("b") == "B") & (spark_col("c").isin("d", "e")), True)
            .when((spark_col("b") == "C") & (spark_col("c").isin("f", "g", "h", "i")), True)
            .otherwise(False),
        )

        # Compare results
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "json_data, path, expected_values",
        [
            ([json.dumps({"a": 1}), json.dumps({"a": 2})], "$.a", ["1", "2"]),
            ([json.dumps({"a": {"b": 3}}), json.dumps({"a": {"b": 4}})], "$.a.b", ["3", "4"]),
            ([json.dumps({"arr": [10, 20]}), json.dumps({"arr": [30, 40]})], "$.arr[1]", ["20", "40"]),
            ([json.dumps({"a": {"b": [5, 6]}}), json.dumps({"a": {"b": [7, 8]}})], "$.a.b[0]", ["5", "7"]),
            (
                [json.dumps({"items": [{"id": 1}, {"id": 2}]}), json.dumps({"items": [{"id": 3}, {"id": 4}]})],
                "$.items[1].id",
                ["2", "4"],
            ),
        ],
    )
    def test_get_json_object(self, spark, json_data, path, expected_values):
        data = {"json_col": json_data}
        spark_df = spark.createDataFrame(
            spark_rows_from_dict(data),
            list(data.keys()),
        )
        expected_df = spark_df.select(spark_get_json_object("json_col", path).alias("result"))

        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(get_json_object("json_col", path).alias("result"))
        result_spark_df = spark.createDataFrame(result_df.toPandas())

        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "literal_value",
        [
            42,  # int
            3.14,  # float
            "hello",  # string
            True,  # boolean
            None,  # null
        ],
    )
    def test_lit_against_spark(self, spark, literal_value):
        data = {"x": [1, 2, 3]}
        sparkle_df = DataFrame(pl.DataFrame(data))
        result_df = sparkle_df.select(lit(literal_value).alias("value")).toPandas()

        # Result using Spark
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_lit(literal_value).alias("value")).toPandas()

        # Compare using pandas
        pdt.assert_frame_equal(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_dtype=False,  # Important: ignores schema/type mismatches
        )

    @pytest.mark.parametrize(
        "a_vals, b_vals, expected_vals",
        [
            ([None, 2, None], [1, None, 3], [1, 2, 3]),
            ([None, None, None], [None, None, None], [None, None, None]),
            ([None, 5, 6], ["x", "y", None], ["x", 5, 6]),
            (["", None, "z"], ["a", "b", None], ["", "b", "z"]),
        ],
    )
    def test_coalesce_against_spark(self, spark, a_vals, b_vals, expected_vals):
        data = {"a": a_vals, "b": b_vals}
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(coalesce(col("a"), col("b")).alias("result"))

        if expected_vals == [None, None, None]:
            null_schema = SparkStructType(
                [
                    SparkStructField("a", SparkStringType(), True),
                    SparkStructField("b", SparkStringType(), True),
                ]
            )
            spark_df = spark.createDataFrame(spark_rows_from_dict(data), null_schema)
        else:
            spark_df = spark.createDataFrame(
                spark_rows_from_dict(data),
                list(data.keys()),
            )

        # Spark requires a common type for coalesce; string covers int+str mixes (Spark 4 strict).
        ca = spark_col("a").cast(SparkStringType())
        cb = spark_col("b").cast(SparkStringType())
        expected_spark_df = spark_df.select(spark_coalesce(ca, cb).alias("result"))

        if expected_vals == [None, None, None]:
            out_schema = SparkStructType([SparkStructField("result", SparkStringType(), True)])
            result_spark_df = create_spark_df(spark, result_df, schema=out_schema)
        else:
            result_spark_df = create_spark_df(spark, result_df)

        assert_pyspark_df_equal(
            result_spark_df.select(spark_col("result").cast(SparkStringType()).alias("result")),
            expected_spark_df.select(spark_col("result").cast(SparkStringType()).alias("result")),
            ignore_nullable=True,
        )

    @pytest.mark.parametrize(
        "values, scale",
        [
            ([1.234, 2.345, 3.456], 0),  # round to integer
            ([1.234, 2.345, 3.456], 1),  # round to 1 decimal
            ([1.234, 2.345, 3.456], 2),  # round to 2 decimals
            ([None, 2.555, 3.666], 1),  # include None
        ],
    )
    def test_round_against_spark(self, spark, values, scale):
        data = {"x": values}

        # Sparkleframe / Polars
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(round(col("x"), scale).alias("rounded")).toPandas()

        # PySpark
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_round(spark_col("x"), scale).alias("rounded")).toPandas()

        # Compare using pandas
        pdt.assert_frame_equal(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-5,
        )

    @pytest.mark.parametrize(
        "data, column",
        [
            ({"x": [3, 1, 2]}, "x"),
            ({"x": ["b", "c", "a"]}, "x"),
            ({"x": [3.3, 1.1, 2.2]}, "x"),
        ],
    )
    def test_asc_against_spark(self, spark, data, column):
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))

        # SparkleFrame: order by asc
        result_df = polars_df.df.sort(asc(col(column)).to_native())
        result_spark_df = create_spark_df(spark, result_df)

        # PySpark: order by asc
        expected_df = spark_df.orderBy(spark_asc(column))

        # Compare using PySpark equality
        assert_pyspark_df_equal(result_spark_df.orderBy("x"), expected_df.orderBy("x"), ignore_nullable=True)

    @pytest.mark.parametrize(
        "data, column",
        [
            ({"x": [3, 1, 2]}, "x"),
            ({"x": ["b", "c", "a"]}, "x"),
            ({"x": [3.3, 1.1, 2.2]}, "x"),
        ],
    )
    def test_desc_against_spark(self, spark, data, column):
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))

        # SparkleFrame: order by desc
        result_df = polars_df.df.sort(desc(col(column)).to_native())
        result_spark_df = create_spark_df(spark, result_df)

        # PySpark: order by desc
        expected_df = spark_df.orderBy(spark_desc(column))

        # Compare using PySpark equality
        assert_pyspark_df_equal(result_spark_df.orderBy("x"), expected_df.orderBy("x"), ignore_nullable=True)

    @pytest.mark.parametrize(
        "col_input",
        [
            "txt",
            col("txt"),
        ],
    )
    @pytest.mark.parametrize(
        "input_values, pattern, replacement, expected_values",
        [
            (["abc123", "xyz456"], r"\d+", "", ["abc", "xyz"]),  # Remove digits
            (["hello world", "world hello"], "world", "earth", ["hello earth", "earth hello"]),  # Replace word
            (["aaa", "aba", "aca"], "a", "x", ["xxx", "xbx", "xcx"]),  # Replace all a's
            (["test123", "123test"], r"^\d+", "NUM", ["test123", "NUMtest"]),  # Match digits at start
            (["test123", "123test"], r"\d+$", "END", ["testEND", "123test"]),  # Match digits at end
        ],
    )
    def test_regexp_replace_str_vs_column(self, spark, col_input, pattern, replacement, input_values, expected_values):
        data = {"txt": input_values}
        spark_input_df = spark.createDataFrame(
            spark_rows_from_dict(data),
            list(data.keys()),
        )

        # Expected Spark result
        expected_df = spark_input_df.select(spark_regexp_replace("txt", pattern, replacement).alias("replaced"))

        # SparkleFrame Polars-based result
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(regexp_replace(col_input, pattern, replacement).alias("replaced"))
        result_spark_df = spark.createDataFrame(result_df.df.to_dicts())

        # Validate against PySpark
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "input_values",
        [
            ["abc", "de", ""],  # basic strings
            ["你好", "世界", ""],  # unicode characters
            [None, "x", "longer string"],  # includes None
            ["😊", "👍🏽", "💯"],  # emojis with multiple bytes
        ],
    )
    @pytest.mark.parametrize("col_input", ["txt", col("txt")])
    def test_length_str_vs_column(self, spark, input_values, col_input):
        data = {"txt": input_values}
        polars_df = DataFrame(pl.DataFrame(data))

        # Row tuples keep None as Spark null (vs pandas object columns → NaN).
        spark_df = spark.createDataFrame(
            spark_rows_from_dict(data),
            list(data.keys()),
        )
        expected_df = spark_df.select(spark_length("txt").alias("result"))

        # SparkleFrame/Polars result
        result_df = polars_df.select(length(col_input).alias("result"))
        result_spark_df = spark.createDataFrame(result_df.df.to_dicts()).withColumn(
            "result", spark_col("result").cast(SparkIntegerType())
        )

        # Assert equality
        assert_pyspark_df_equal(result_spark_df, expected_df)

    @pytest.mark.parametrize(
        "col_input",
        [
            "ts",
            col("ts"),
        ],
    )
    @pytest.mark.parametrize(
        "datetime_strs, fmt",
        [
            # Standard format
            (["2023-01-01 12:34:56", "2024-02-02 23:45:01"], "yyyy-MM-dd HH:mm:ss"),
            # Day-first format
            (["01-03-2023 09:15:00", "31-12-2022 23:59:59"], "dd-MM-yyyy HH:mm:ss"),
            # Compact format
            (["20230101 120000", "20240101 130000"], "yyyyMMdd HHmmss"),
            # Millisecond precision (3 digits)
            (["2024-05-31 20:14:19.993", "2023-12-12 11:11:11.123"], "yyyy-MM-dd HH:mm:ss.SSS"),
            # Microsecond precision (6 digits)
            (["2024-05-31 23:58:32.880000", "2023-12-12 11:11:11.123456"], "yyyy-MM-dd HH:mm:ss.SSSSSS"),
            # Single-digit millisecond
            (["2024-05-31 20:14:19.9", "2023-12-12 11:11:11.1"], "yyyy-MM-dd HH:mm:ss.S"),
            # Two-digit millisecond
            (["2024-05-31 20:14:19.99", "2023-12-12 11:11:11.12"], "yyyy-MM-dd HH:mm:ss.SS"),
            # Five-digit fractional seconds (partial microseconds)
            (["2024-05-31 20:14:19.12345", "2023-12-12 11:11:11.99999"], "yyyy-MM-dd HH:mm:ss.SSSSS"),
        ],
    )
    def test_to_timestamp_against_spark(self, spark, col_input, datetime_strs, fmt):
        data = {"ts": datetime_strs}
        polars_df = DataFrame(pl.DataFrame(data))

        spark_df = spark.createDataFrame(
            spark_rows_from_dict(data),
            list(data.keys()),
        )
        expected_df = spark_df.select(spark_to_timestamp("ts", fmt).alias("result"))

        # Sparkleframe / Polars output
        result_df = create_spark_df(spark, polars_df.select(to_timestamp(col_input, fmt).alias("result")))

        assert_pyspark_df_equal(result_df, expected_df)

    @pytest.mark.parametrize(
        "datetime_strs",
        [
            ["2023-01-01 12:34:56", "2024-02-02 23:45:01"],
            ["2024-05-31T20:14:19", "2023-12-12T11:11:11"],
        ],
    )
    def test_to_timestamp_no_format_against_spark(self, spark, datetime_strs):
        data = {"ts": datetime_strs}
        polars_df = DataFrame(pl.DataFrame(data))

        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_to_timestamp("ts").alias("result"))
        result_df = polars_df.select(to_timestamp("ts").alias("result"))

        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_to_timestamp_no_format_iso8601_z_against_spark(self, spark) -> None:
        data = {"createdOn": ["2026-04-26T00:00:00Z"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_to_timestamp("createdOn").alias("t"))
        for fn in (to_timestamp, try_to_timestamp):
            result_df = polars_df.select(fn("createdOn").alias("t"))
            assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_to_timestamp_no_format_malformed_raises(self, spark):
        """Spark 4 to_timestamp(col) without format is ANSI-strict and raises on malformed input."""
        polars_df = DataFrame(pl.DataFrame({"ts": ["2023-01-01 12:34:56", "not-a-date"]}))
        with pytest.raises(Exception):
            polars_df.select(to_timestamp("ts").alias("result")).to_native_df()

        spark_df = spark.createDataFrame([("not-a-date",)], ["ts"])
        with pytest.raises(Exception):
            spark_df.select(spark_to_timestamp("ts").alias("result")).collect()

    @pytest.mark.parametrize(
        "bad_value, fmt",
        [
            ("not-a-date", "yyyy-MM-dd HH:mm:ss"),
            ("01-03-2023 09:15:00", "yyyy-MM-dd HH:mm:ss"),
        ],
    )
    def test_to_timestamp_with_format_malformed_raises(self, spark, bad_value, fmt):
        """Spark 4 to_timestamp(col, fmt) is ANSI-strict and raises on malformed or pattern-mismatched input."""
        polars_df = DataFrame(pl.DataFrame({"ts": [bad_value]}))
        with pytest.raises(Exception):
            polars_df.select(to_timestamp("ts", fmt).alias("result")).to_native_df()

        spark_df = spark.createDataFrame([(bad_value,)], ["ts"])
        with pytest.raises(Exception):
            spark_df.select(spark_to_timestamp("ts", fmt).alias("result")).collect()

    @pytest.mark.parametrize(
        "sparkle_col_type, spark_col_type",
        [
            (str, str),
            (col, spark_col),
        ],
    )
    @pytest.mark.parametrize(
        "sparkle_func, spark_func",
        [
            (rank, spark_rank),
            (dense_rank, spark_dense_rank),
            (row_number, spark_row_number),
        ],
    )
    @pytest.mark.parametrize(
        "partition_cols, order_cols",
        [
            (["group"], [("category", "asc")]),
            (["group"], [("category", "asc_nulls_first")]),
            (["group"], [("category", "asc_nulls_last")]),
            (["group"], [("value", "desc")]),
            (["group", "subcategory"], [("value", "asc")]),
            (["group", "subcategory"], [("value", "desc"), ("category", "asc")]),
            (["group", "category"], [("subcategory", "asc"), ("value", "desc")]),
        ],
    )
    def test_ranks_with_subcategory(
        self, spark, sparkle_col_type, spark_col_type, sparkle_func, spark_func, partition_cols, order_cols
    ):
        # Extended sample data with `subcategory`
        # Expanded dataset for more exhaustive testing
        row_dicts = [
            {"group": "A", "category": "A", "subcategory": "alpha", "value": 100},
            {"group": "A", "category": "A", "subcategory": "alpha", "value": 100},
            {"group": "A", "category": "A", "subcategory": "alpha", "value": 200},
            {"group": "A", "category": "A", "subcategory": "beta", "value": 50},
            {"group": "A", "category": "B", "subcategory": "alpha", "value": 120},
            {"group": "A", "category": "B", "subcategory": "beta", "value": 180},
            {"group": "B", "category": "A", "subcategory": "alpha", "value": 300},
            {"group": "B", "category": "A", "subcategory": "beta", "value": 310},
            {"group": "B", "category": "B", "subcategory": "alpha", "value": 150},
            {"group": "B", "category": "B", "subcategory": "beta", "value": 160},
            {"group": "B", "category": "B", "subcategory": "beta", "value": 170},
            {"group": "C", "category": "A", "subcategory": "alpha", "value": 80},
            {"group": "C", "category": "A", "subcategory": "alpha", "value": 85},
            {"group": "C", "category": "A", "subcategory": "beta", "value": 70},
            {"group": "C", "category": "B", "subcategory": "beta", "value": 60},
            {"group": "C", "category": "B", "subcategory": "beta", "value": 100},
        ]
        data = {key: [row[key] for row in row_dicts] for key in row_dicts[0]}

        order_func = {
            "asc": (asc, spark_asc),
            "asc_nulls_first": (asc_nulls_first, spark_asc_nulls_first),
            "asc_nulls_last": (asc_nulls_last, spark_asc_nulls_last),
            "desc": (desc, spark_desc),
            "desc_nulls_first": (desc_nulls_first, spark_desc_nulls_first),
            "desc_nulls_last": (desc_nulls_last, spark_desc_nulls_last),
        }

        pl_df = pl.DataFrame(data)

        # Build Sparkle DataFrame
        sparkle_df = DataFrame(pl_df)

        # Convert to order expressions
        order_exprs = [order_func[direction][0](col) for col, direction in order_cols]

        # Apply rank over window
        sparkle_df = sparkle_df.withColumn(
            "rank",
            sparkle_func().over(
                Window.partitionBy(*[sparkle_col_type(col) for col in partition_cols]).orderBy(*order_exprs)
            ),
        )

        # Cast and sort for stable comparison
        sparkle_df = (
            create_spark_df(spark, sparkle_df)
            .withColumn("rank", spark_col("rank").cast(SparkIntegerType()))
            .orderBy("group", "category", "subcategory", "value")
        )

        # Build Spark reference DataFrame
        spark_df = create_spark_df(spark, pl_df)

        spark_order_exprs = [order_func[direction][1](col) for col, direction in order_cols]

        spark_df = spark_df.withColumn(
            "rank",
            spark_func().over(
                SparkWindow.partitionBy(*[spark_col_type(col) for col in partition_cols]).orderBy(*spark_order_exprs)
            ),
        ).orderBy("group", "category", "subcategory", "value")

        assert_pyspark_df_equal(sparkle_df, spark_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "values",
        [
            [-5, -1, 0, 1, 5],  # integers
            [-3.5, -0.1, 0.0, 0.1, 3.5],  # floats
            [None, -2, 2, None],  # include None
        ],
    )
    def test_abs_against_spark(self, spark, values):
        data = {"x": values}

        # Sparkleframe Polars (native null ints align with Spark row inference)
        sf_df = DataFrame(pl.DataFrame(data))
        result_sf = sf_df.select(abs(col("x")).alias("abs_x"))
        result_spark_df = create_spark_df(spark, result_sf)

        # PySpark
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_abs("x").alias("abs_x"))

        # Compare
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True, allow_nan_equality=True)

    @pytest.mark.parametrize("col_input", ["txt", col("txt")])
    @pytest.mark.parametrize(
        "input_values",
        [
            ["ABC", "abc", "AbC"],  # mixed case
            ["", None, "Already lower"],  # empty and None
            ["MiXeD 123!@$", "CamelCase", "UPPER lower"],  # with numbers/symbols
        ],
    )
    def test_lower_str_vs_column(self, spark, col_input, input_values):
        data = {"txt": input_values}

        spark_input_df = spark.createDataFrame(
            spark_rows_from_dict(data),
            list(data.keys()),
        )
        expected_df = spark_input_df.select(spark_lower("txt").alias("lowered"))

        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(lower(col_input).alias("lowered"))
        result_spark_df = create_spark_df(spark, result_df)

        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    def test_struct_oracle(self, spark, sparkle_df, spark_df):
        """PySpark parity: field names follow CreateStruct (plain cols vs colN for lit / expr)."""
        exprs = [
            struct("a", "b").alias("s1"),
            struct(col("a"), col("b")).alias("s2"),
            struct([col("a"), col("b")]).alias("s3"),
            struct(col("a"), lit(1)).alias("s4"),
            struct(lit(1), lit(2)).alias("s5"),
            struct(col("a") + 1, col("b")).alias("s6"),
            struct(col("a"), struct(col("b"), lit(3))).alias("s7"),
            struct(col("a").alias("z")).alias("s8"),
        ]
        expected_spark_df = spark_df.select(
            spark_struct("a", "b").alias("s1"),
            spark_struct(spark_col("a"), spark_col("b")).alias("s2"),
            spark_struct([spark_col("a"), spark_col("b")]).alias("s3"),
            spark_struct(spark_col("a"), spark_lit(1)).alias("s4"),
            spark_struct(spark_lit(1), spark_lit(2)).alias("s5"),
            spark_struct(spark_col("a") + 1, spark_col("b")).alias("s6"),
            spark_struct(spark_col("a"), spark_struct(spark_col("b"), spark_lit(3))).alias("s7"),
            spark_struct(spark_col("a").alias("z")).alias("s8"),
        )
        pdf = sparkle_df.select(*exprs).toPandas()
        result_spark_df = spark.createDataFrame(pdf, schema=expected_spark_df.schema)
        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_struct_nested_aliased_inner_structs_match_spark(self, spark, sparkle_df, spark_df):
        """Outer struct must use .alias() names for nested struct children (not col1/col2)."""
        inner_left = struct(col("a").alias("field_a"))
        inner_right = struct(col("b").alias("field_b"))
        composite = struct(inner_left.alias("nested_x"), inner_right.alias("nested_y"))
        pdf = sparkle_df.select(composite.alias("composite")).toPandas()
        expected_spark_df = spark_df.select(
            spark_struct(
                spark_struct(spark_col("a").alias("field_a")).alias("nested_x"),
                spark_struct(spark_col("b").alias("field_b")).alias("nested_y"),
            ).alias("composite")
        )
        result_spark_df = spark.createDataFrame(pdf, schema=expected_spark_df.schema)
        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_struct_nested_with_array_and_map(self, spark):
        """Parity with Spark struct(array, map); compare via Arrow (avoids pandas map/dict mismatch)."""
        schema = SparkStructType(
            [
                SparkStructField("id", SparkIntegerType(), True),
                SparkStructField("nums", SparkArrayType(SparkIntegerType()), True),
                SparkStructField("kv", SparkMapType(SparkStringType(), SparkDoubleType()), True),
            ]
        )
        spark_row = (1, [10, 20, 30], {"x": 1.0, "y": 2.0})
        spark_input = spark.createDataFrame([spark_row], schema)
        pl_df = pl.DataFrame(
            {
                "id": [1],
                "nums": [[10, 20, 30]],
                "kv": [[{"key": "x", "value": 1.0}, {"key": "y", "value": 2.0}]],
            }
        )
        sparkle_df = DataFrame(pl_df)
        exprs = [
            struct(col("id"), col("nums"), col("kv")).alias("s1"),
            struct(col("id"), struct(col("nums"), col("kv"))).alias("s2"),
        ]
        expected_spark_df = spark_input.select(
            spark_struct(spark_col("id"), spark_col("nums"), spark_col("kv")).alias("s1"),
            spark_struct(spark_col("id"), spark_struct(spark_col("nums"), spark_col("kv"))).alias("s2"),
        )
        got = sparkle_df.select(*exprs).to_native_df()
        expected_pl = pl.from_arrow(expected_spark_df.toArrow())
        assert_frame_equal(got, expected_pl, check_dtypes=False)

    def test_struct_requires_at_least_one_column(self):
        with pytest.raises(ValueError, match="struct requires at least one column"):
            struct()


class TestConcat:
    """Behaviour tests for :func:`~sparkleframe.polarsdf.functions.concat` (not copied from prior commits)."""

    def test_concat_without_inputs_raises(self) -> None:
        with pytest.raises(ValueError, match="concat requires at least one column"):
            concat()

    def test_concat_single_column_is_identity_on_strings(self, spark) -> None:
        data = {"token": ["zig", None, ""]}
        pl_df = pl.DataFrame(data)
        polars_df = DataFrame(pl_df)
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        out_schema = SparkStructType([SparkStructField("out", SparkStringType(), True)])
        result_spark_df = create_spark_df(spark, polars_df.select(concat("token").alias("out")), schema=out_schema)
        expected_df = spark_df.select(spark_concat(spark_col("token")).alias("out"))
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    def test_concat_two_parts_any_null_yields_null(self, spark) -> None:
        data = {
            "prefix": ["aa", None, "cc"],
            "suffix": ["bb", "bb", None],
        }
        pl_df = pl.DataFrame(data)
        polars_df = DataFrame(pl_df)
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        out_schema = SparkStructType([SparkStructField("out", SparkStringType(), True)])
        result_spark_df = create_spark_df(
            spark,
            polars_df.select(concat(col("prefix"), col("suffix")).alias("out")),
            schema=out_schema,
        )
        expected_df = spark_df.select(spark_concat(spark_col("prefix"), spark_col("suffix")).alias("out"))
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    def test_concat_accepts_string_name_or_column_object(self, spark) -> None:
        pl_df = pl.DataFrame({"segment": ["north", "south"]})
        polars_df = DataFrame(pl_df)
        spark_df = spark.createDataFrame(pl_df.to_pandas())
        by_name_spark = create_spark_df(
            spark, polars_df.select(concat("segment", lit(":"), col("segment")).alias("out"))
        )
        by_col_spark = create_spark_df(
            spark, polars_df.select(concat(col("segment"), lit(":"), "segment").alias("out"))
        )
        expected_df = spark_df.select(
            spark_concat(spark_col("segment"), spark_lit(":"), spark_col("segment")).alias("out")
        )
        assert_pyspark_df_equal(by_name_spark, expected_df, ignore_nullable=True)
        assert_pyspark_df_equal(by_col_spark, expected_df, ignore_nullable=True)

    def test_concat_coerces_integer_columns_like_strings(self, spark) -> None:
        pl_df = pl.DataFrame({"lane": [7, 0], "slot": [13, 42]})
        polars_df = DataFrame(pl_df)
        result_spark_df = create_spark_df(
            spark,
            polars_df.select(concat(col("lane"), lit("-"), col("slot")).alias("merged")),
        )
        spark_df = spark.createDataFrame(pl_df.to_pandas())
        expected_df = spark_df.select(
            spark_concat(spark_col("lane"), spark_lit("-"), spark_col("slot")).alias("merged"),
        )
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)


_RE_UUID_V4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


class TestUuid:
    def test_uuid_yields_version4_string_format(self) -> None:
        """``uuid()`` uses ``uuid4()``; unseeded values are not equal to PySpark, only shape is asserted."""
        df = DataFrame(pl.DataFrame({"a": list(range(20))}))
        out = df.select(uuid().alias("u")).to_native_df()["u"].to_list()
        assert len(out) == 20
        assert len(set(out)) == 20
        for s in out:
            assert _RE_UUID_V4.match(s) is not None
            assert std_uuid.UUID(s).version == 4


class TestInitcap:
    """Parity with PySpark for initcap."""

    @pytest.mark.parametrize(
        "values",
        [
            ["hello world", "FOO BAR", "already Title"],
            [None, "", "café latte"],
            ["UPPER", "lower", "mIxEd CaSe"],
        ],
    )
    def test_initcap_against_spark(self, spark, values) -> None:
        data = {"s": values}
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(initcap("s").alias("out"))
        expected_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            spark_initcap(spark_col("s")).alias("out")
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestMd5:
    def test_md5_string_against_spark(self, spark) -> None:
        data = {"s": ["abc", "", None, "café"]}
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(md5("s").alias("h"))
        expected_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            spark_md5(spark_col("s")).alias("h")
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_md5_binary_against_spark(self, spark) -> None:
        data = {"b": [b"abc", None, b"", b"\x00\xff"]}
        polars_df = DataFrame(pl.DataFrame(data, schema={"b": pl.Binary}))
        result_df = polars_df.select(md5("b").alias("h"))
        spark_in = spark.createDataFrame(spark_rows_from_dict(data), ["b"])
        expected_df = spark_in.select(spark_md5(spark_col("b")).alias("h"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestTrimAndSplit:
    """Parity with PySpark for trim and split."""

    @pytest.mark.parametrize(
        "values",
        [
            ["  a  ", "b\t", " c \n"],
            [None, "  x  ", ""],
        ],
    )
    def test_trim_against_spark(self, spark, values) -> None:
        data = {"s": values}
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(trim("s").alias("out"))
        expected_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            spark_trim(spark_col("s")).alias("out")
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    @pytest.mark.parametrize(
        "values, pattern, limit",
        [
            (["a-b-c", "x-y-z", None], r"-", -1),
            (["a1b1c", "nope"], r"\d", -1),
            (["a-b-c-d", "p.q"], r"-", 2),
        ],
    )
    def test_split_against_spark(self, spark, values: list, pattern: str, limit: int) -> None:
        data = {"s": values}
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(split("s", pattern, limit).alias("parts"))
        expected_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            spark_split(spark_col("s"), spark_lit(pattern), limit).alias("parts")
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestNowAndMonotonicallyIncreasingId:
    """Parity with PySpark for now and monotonically_increasing_id."""

    def test_now_all_rows_equal_and_close_to_spark(self, spark) -> None:
        data = {"x": [1, 2, 3]}
        polars_df = DataFrame(pl.DataFrame(data))
        result_spark_df = create_spark_df(spark, polars_df.select(now().alias("t")))
        expected_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            spark_now().alias("t")
        )
        ms1 = [r[0] for r in result_spark_df.select(spark_unix_millis("t").alias("m")).collect()]
        ms2 = [r[0] for r in expected_df.select(spark_unix_millis("t").alias("m")).collect()]
        assert len(set(ms1)) == 1
        assert len(set(ms2)) == 1
        assert builtins.abs(ms1[0] - ms2[0]) < 3_000

    def test_current_timestamp_all_rows_equal_and_close_to_spark(self, spark) -> None:
        data = {"x": [1, 2, 3]}
        polars_df = DataFrame(pl.DataFrame(data))
        result_spark_df = create_spark_df(spark, polars_df.select(current_timestamp().alias("t")))
        expected_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            spark_current_timestamp().alias("t")
        )
        ms1 = [r[0] for r in result_spark_df.select(spark_unix_millis("t").alias("m")).collect()]
        ms2 = [r[0] for r in expected_df.select(spark_unix_millis("t").alias("m")).collect()]
        assert len(set(ms1)) == 1
        assert len(set(ms2)) == 1
        assert builtins.abs(ms1[0] - ms2[0]) < 3_000

    def test_monotonically_increasing_id_against_spark(self, spark) -> None:
        data = {"k": ["a", "b", "c", "d"]}
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(monotonically_increasing_id().alias("id"))
        expected_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            spark_monotonically_increasing_id().alias("id")
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestToDate:
    """Tests for to_date — strict date parsing (Spark 4 ANSI default)."""

    @pytest.mark.parametrize(
        "date_strs, fmt",
        [
            (["1997-02-28", "2024-12-31"], "yyyy-MM-dd"),
            (["28-02-1997", "31-12-2024"], "dd-MM-yyyy"),
        ],
    )
    def test_to_date_against_spark(self, spark, date_strs, fmt):
        data = {"d": date_strs}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_to_date("d", fmt).alias("result"))
        result_df = polars_df.select(to_date("d", fmt).alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    @pytest.mark.parametrize(
        "bad_value, fmt",
        [
            ("bad", "yyyy-MM-dd"),
            ("1997-02-28", "dd-MM-yyyy"),
        ],
    )
    def test_to_date_malformed_raises(self, spark, bad_value, fmt):
        """Spark 4 to_date is ANSI-strict and raises on malformed or pattern-mismatched input."""
        polars_df = DataFrame(pl.DataFrame({"d": [bad_value]}))
        with pytest.raises(Exception):
            polars_df.select(to_date("d", fmt).alias("result")).to_native_df()

        spark_df = spark.createDataFrame([(bad_value,)], ["d"])
        with pytest.raises(Exception):
            spark_df.select(spark_to_date("d", fmt).alias("result")).collect()


class TestRand:
    def test_rand_seeded_is_repeatable_and_in_range(self) -> None:
        polars_df = DataFrame(pl.DataFrame({"x": [1, 2, 3, 4, 5]}))
        a = polars_df.select(rand(42).alias("r")).to_native_df()["r"].to_list()
        b = polars_df.select(rand(42).alias("r")).to_native_df()["r"].to_list()
        assert a == b
        assert all(isinstance(v, float) and 0.0 <= v < 1.0 for v in a)

    def test_rand_unseeded_values_in_unit_interval(self) -> None:
        polars_df = DataFrame(pl.DataFrame({"x": list(range(50))}))
        vals = polars_df.select(rand().alias("r")).to_native_df()["r"].to_list()
        assert all(isinstance(v, float) and 0.0 <= v < 1.0 for v in vals)


class TestArray:
    def test_array_empty_broadcasts_per_row(self) -> None:
        polars_df = DataFrame(pl.DataFrame({"x": [1, 2]}))
        col_vals = polars_df.withColumn("a", array()).to_native_df()["a"].to_list()
        assert col_vals == [[], []]

    def test_array_from_column_names(self) -> None:
        polars_df = DataFrame(pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
        col_vals = polars_df.select(array("a", "b").alias("m")).to_native_df()["m"].to_list()
        assert col_vals == [[1, 3], [2, 4]]


class TestSizeObjectList:
    """``size`` on Polars ``Object`` list cells (nested JSON-like structs)."""

    def test_size_object_list_select(self) -> None:
        offers = pl.Series("offers", [[1, 2], [], None, [3]], dtype=pl.Object)
        df = DataFrame(pl.DataFrame([offers]))
        assert df.select(size("offers").alias("sz")).to_native_df()["sz"].to_list() == [2, 0, None, 1]

    def test_size_typed_list_column(self) -> None:
        df = DataFrame(pl.DataFrame({"arr": [[1, 2], [], [3]]}))
        assert df.select(size("arr").alias("sz")).to_native_df()["sz"].to_list() == [2, 0, 1]

    def test_filter_or_on_object_list_columns_like_heloc(self) -> None:
        """Spark ``size`` + ``filter`` on list fields that Polars stores as ``Object``."""
        offers = pl.Series("replacements_offers", [[1, 2], None, [], [3]], dtype=pl.Object)
        loans = pl.Series("replacements_loans", [[], [3], [], []], dtype=pl.Object)
        df = DataFrame(pl.DataFrame([offers, loans]))
        replaceable_offer_has_items = col("replacements_offers").isNotNull() & (size(col("replacements_offers")) > 0)
        replaceable_loans_has_items = col("replacements_loans").isNotNull() & (size(col("replacements_loans")) > 0)
        out = df.filter(replaceable_offer_has_items | replaceable_loans_has_items)
        assert out.count() == 3


class TestToJson:
    def test_options_argument_rejected(self) -> None:
        polars_df = DataFrame(pl.DataFrame({"a": [1]}))
        with pytest.raises(ValueError):
            polars_df.select(to_json(struct("a"), {"timestampFormat": "yyyy"}).alias("j"))


class TestTryToTimestamp:
    """Tests for try_to_timestamp — verifies null-safe parsing behaviour."""

    @pytest.mark.parametrize(
        "datetime_strs, fmt",
        [
            (["2023-01-01 12:34:56", "2024-02-02 23:45:01"], "yyyy-MM-dd HH:mm:ss"),
            (["01-03-2023 09:15:00", "31-12-2022 23:59:59"], "dd-MM-yyyy HH:mm:ss"),
            (["2024-05-31 20:14:19.993", "2023-12-12 11:11:11.123"], "yyyy-MM-dd HH:mm:ss.SSS"),
        ],
    )
    def test_try_to_timestamp_valid_matches_to_timestamp(self, spark, datetime_strs, fmt):
        df = pd.DataFrame({"ts": datetime_strs})
        polars_df = DataFrame(pl.DataFrame(df))

        result_strict = polars_df.select(to_timestamp("ts", fmt).alias("result")).to_native_df()
        result_try = polars_df.select(try_to_timestamp("ts", fmt).alias("result")).to_native_df()

        assert result_strict["result"].to_list() == result_try["result"].to_list()

        spark_df = spark.createDataFrame(df)
        expected_df = spark_df.select(spark_try_to_timestamp(spark_col("ts"), spark_lit(fmt)).alias("result"))
        result_df = polars_df.select(try_to_timestamp("ts", fmt).alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_try_to_timestamp_malformed_returns_null(self, spark):
        data = {"ts": ["2023-01-01 12:34:56", "not-a-date", None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_try_to_timestamp("ts").alias("result"))
        result_df = polars_df.select(try_to_timestamp("ts").alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    @pytest.mark.parametrize(
        "datetime_strs, fmt",
        [
            (
                ["2023-01-01 12:34:56", "not-a-date", None],
                "yyyy-MM-dd HH:mm:ss",
            ),
            (
                ["2023-01-01 12:34:56", "01-03-2023 09:15:00", None],
                "yyyy-MM-dd HH:mm:ss",
            ),
        ],
    )
    def test_try_to_timestamp_with_format_malformed_returns_null(self, spark, datetime_strs, fmt):
        data = {"ts": datetime_strs}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_try_to_timestamp(spark_col("ts"), spark_lit(fmt)).alias("result"))
        result_df = polars_df.select(try_to_timestamp("ts", fmt).alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_try_to_timestamp_format_iso_t_separator_returns_null(self, spark) -> None:
        data = {"ts": ["2024-03-15T10:20:30", "2024-03-15 10:20:30"]}
        fmt = "yyyy-MM-dd HH:mm:ss"
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.selectExpr(f"try_to_timestamp(ts, '{fmt}') as t")
        result_df = polars_df.select(try_to_timestamp("ts", fmt).alias("t"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_to_timestamp_with_format_iso_t_separator_raises(self, spark) -> None:
        fmt = "yyyy-MM-dd HH:mm:ss"
        polars_df = DataFrame(pl.DataFrame({"ts": ["2024-03-15T10:20:30"]}))
        with pytest.raises(Exception):
            polars_df.select(to_timestamp("ts", fmt).alias("t")).to_native_df()
        spark_df = spark.createDataFrame([("2024-03-15T10:20:30",)], ["ts"])
        with pytest.raises(Exception):
            spark_df.selectExpr(f"to_timestamp(ts, '{fmt}') as t").collect()

    def test_try_to_timestamp_accepts_column_input(self, spark):
        data = {"ts": ["2023-01-01 12:34:56"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_try_to_timestamp("ts").alias("result"))
        result_df = polars_df.select(try_to_timestamp(col("ts")).alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestTryToDate:
    """Tests for try_to_date — verifies null-safe date parsing."""

    @pytest.mark.parametrize(
        "date_strs, fmt",
        [
            (["1997-02-28", "2024-12-31"], "yyyy-MM-dd"),
            (["28-02-1997", "31-12-2024"], "dd-MM-yyyy"),
        ],
    )
    def test_try_to_date_valid_matches_to_date(self, spark, date_strs, fmt):
        data = {"d": date_strs}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))

        result_strict = polars_df.select(to_date("d", fmt).alias("result")).to_native_df()
        result_try = polars_df.select(try_to_date("d", fmt).alias("result")).to_native_df()
        assert result_strict["result"].to_list() == result_try["result"].to_list()

        expected_df = spark_df.select(spark_try_to_date("d", fmt).alias("result"))
        result_df = polars_df.select(try_to_date("d", fmt).alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_try_to_date_default_format_malformed_returns_null(self, spark):
        data = {"d": ["1997-02-28", "2024-12-31", "bad", None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_try_to_date("d").alias("result"))
        result_df = polars_df.select(try_to_date("d").alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_try_to_date_custom_format_against_spark(self, spark):
        data = {"d": ["28-02-1997", "31-12-2024"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_try_to_date("d", "dd-MM-yyyy").alias("result"))
        result_df = polars_df.select(try_to_date("d", "dd-MM-yyyy").alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_try_to_date_custom_format_malformed_returns_null(self, spark):
        data = {
            "d": ["28-02-1997", "31-12-2024", "not-a-date", "1997-02-28", None],
        }
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_try_to_date("d", "dd-MM-yyyy").alias("result"))
        result_df = polars_df.select(try_to_date("d", "dd-MM-yyyy").alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_try_to_date_accepts_column_input(self, spark):
        data = {"d": ["2024-01-01"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_try_to_date("d").alias("result"))
        result_df = polars_df.select(try_to_date(col("d")).alias("result"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestTryElementAt:
    """Tests for try_element_at — arrays (1-based) and maps."""

    @staticmethod
    def _spark_df_string_array(spark, values: list):
        return spark.createDataFrame(
            [(values,)],
            schema=SparkStructType([SparkStructField("arr", SparkArrayType(SparkStringType()), True)]),
        )

    @staticmethod
    def _spark_df_from_polars_expr(spark, polars_df: DataFrame, expr, value_type):
        pdf = polars_df.select(expr).to_native_df().to_pandas()
        return spark.createDataFrame(pdf, schema=SparkStructType([SparkStructField("v", value_type, True)]))

    def test_array_positive_index(self, spark):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", 1).alias("v")).to_native_df()
        assert result["v"][0] == "a"
        spark_df = self._spark_df_string_array(spark, ["a", "b", "c"])
        expected = spark_df.select(spark_try_element_at(spark_col("arr"), spark_lit(1)).alias("v"))
        result_spark_df = create_spark_df(spark, polars_df.select(try_element_at("arr", 1).alias("v")))
        assert_pyspark_df_equal(result_spark_df, expected, ignore_nullable=True)

    def test_array_last_element(self, spark):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", 3).alias("v")).to_native_df()
        assert result["v"][0] == "c"
        spark_df = self._spark_df_string_array(spark, ["a", "b", "c"])
        expected = spark_df.select(spark_try_element_at(spark_col("arr"), spark_lit(3)).alias("v"))
        result_spark_df = create_spark_df(spark, polars_df.select(try_element_at("arr", 3).alias("v")))
        assert_pyspark_df_equal(result_spark_df, expected, ignore_nullable=True)

    def test_array_negative_index(self, spark):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", -1).alias("v")).to_native_df()
        assert result["v"][0] == "c"
        spark_df = self._spark_df_string_array(spark, ["a", "b", "c"])
        expected = spark_df.select(spark_try_element_at(spark_col("arr"), spark_lit(-1)).alias("v"))
        result_spark_df = create_spark_df(spark, polars_df.select(try_element_at("arr", -1).alias("v")))
        assert_pyspark_df_equal(result_spark_df, expected, ignore_nullable=True)

    def test_array_oob_returns_null(self, spark):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", 4).alias("v")).to_native_df()
        assert result["v"][0] is None
        spark_df = self._spark_df_string_array(spark, ["a", "b", "c"])
        expected = spark_df.select(spark_try_element_at(spark_col("arr"), spark_lit(4)).alias("v"))
        null_string_schema = SparkStructType([SparkStructField("v", SparkStringType(), True)])
        result_spark_df = create_spark_df(
            spark,
            polars_df.select(try_element_at("arr", 4).alias("v")),
            schema=null_string_schema,
        )
        assert_pyspark_df_equal(result_spark_df, expected, ignore_nullable=True)

    def test_array_zero_index_returns_null(self, spark):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", 0).alias("v")).to_native_df()
        assert result["v"][0] is None
        # Spark 4 try_element_at(col, 0) fails at runtime (invalid index); we return null instead.
        result_spark_df = self._spark_df_from_polars_expr(
            spark, polars_df, try_element_at("arr", 0).alias("v"), SparkStringType()
        )
        assert result_spark_df.collect()[0]["v"] is None

    def test_map_literal_key_via_lit_column(self, spark) -> None:
        """Literal map keys use ``lit`` in PySpark; ``try_element_at`` does not treat bare strings as literals."""
        df = pl.DataFrame({"m": [[{"key": "a", "value": 1.0}, {"key": "b", "value": 2.0}]]})
        polars_df = DataFrame(df)
        spark_df = spark.createDataFrame(
            [({"a": 1.0, "b": 2.0},)],
            schema=SparkStructType([SparkStructField("m", SparkMapType(SparkStringType(), SparkDoubleType()), True)]),
        )
        sf_result = polars_df.select(try_element_at("m", lit("a")).alias("v"))
        spark_result = spark_df.select(spark_try_element_at(spark_col("m"), spark_lit("a")).alias("v"))
        assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

    def test_map_string_extraction_is_column_name(self, spark) -> None:
        """``try_element_at(map, 'col')`` uses column ``col`` as the key (SPARK-48766), not literal ``'col'``."""
        df = pl.DataFrame(
            {
                "m": [[{"key": "a", "value": 1.0}, {"key": "b", "value": 2.0}]],
                "lookup": ["a"],
            }
        )
        polars_df = DataFrame(df)
        spark_df = spark.createDataFrame(
            [({"a": 1.0, "b": 2.0}, "a")],
            schema=SparkStructType(
                [
                    SparkStructField("m", SparkMapType(SparkStringType(), SparkDoubleType()), True),
                    SparkStructField("lookup", SparkStringType(), True),
                ]
            ),
        )
        sf_result = polars_df.select(try_element_at("m", "lookup").alias("v"))
        spark_result = spark_df.select(spark_try_element_at(spark_col("m"), spark_col("lookup")).alias("v"))
        assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

    def test_accepts_column_input(self, spark):
        df = pl.DataFrame({"arr": [["x", "y"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at(col("arr"), 1).alias("v")).to_native_df()
        assert result["v"][0] == "x"
        spark_df = self._spark_df_string_array(spark, ["x", "y"])
        expected = spark_df.select(spark_try_element_at(spark_col("arr"), spark_lit(1)).alias("v"))
        result_spark_df = create_spark_df(spark, polars_df.select(try_element_at(col("arr"), 1).alias("v")))
        assert_pyspark_df_equal(result_spark_df, expected, ignore_nullable=True)


class TestElementAt:
    """``element_at`` is ANSI-strict on invalid array indices (Spark 4); not an alias of ``try_element_at``."""

    @staticmethod
    def _spark_df_string_array(spark, values: list):
        return spark.createDataFrame(
            [(values,)],
            schema=SparkStructType([SparkStructField("arr", SparkArrayType(SparkStringType()), True)]),
        )

    def test_array_valid_index_matches_spark(self, spark) -> None:
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        spark_df = self._spark_df_string_array(spark, ["a", "b", "c"])
        sf_result = polars_df.select(element_at("arr", 2).alias("v"))
        spark_result = spark_df.select(spark_element_at(spark_col("arr"), spark_lit(2)).alias("v"))
        assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

    def test_array_oob_raises_like_spark(self, spark) -> None:
        """Spark 4 ANSI: out-of-bounds ``element_at`` raises; ``try_element_at`` returns null."""
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        spark_df = self._spark_df_string_array(spark, ["a", "b", "c"])
        with pytest.raises(Exception):
            spark_df.select(spark_element_at(spark_col("arr"), spark_lit(4))).collect()
        with pytest.raises(Exception):
            polars_df.select(element_at("arr", 4)).to_native_df()
        # Lenient counterpart must not raise.
        assert polars_df.select(try_element_at("arr", 4).alias("v")).to_native_df()["v"][0] is None

    def test_array_zero_index_raises_like_spark(self, spark) -> None:
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        spark_df = self._spark_df_string_array(spark, ["a", "b", "c"])
        with pytest.raises(Exception):
            spark_df.select(spark_element_at(spark_col("arr"), spark_lit(0))).collect()
        with pytest.raises(Exception):
            polars_df.select(element_at("arr", 0)).to_native_df()

    def test_map_literal_string_key_matches_spark(self, spark) -> None:
        """``element_at(map, 'key')`` uses a literal key (unlike ``try_element_at`` string = column name)."""
        df = pl.DataFrame({"m": [[{"key": "a", "value": 1.0}, {"key": "b", "value": 2.0}]]})
        polars_df = DataFrame(df)
        spark_df = spark.createDataFrame(
            [({"a": 1.0, "b": 2.0},)],
            schema=SparkStructType([SparkStructField("m", SparkMapType(SparkStringType(), SparkDoubleType()), True)]),
        )
        sf_result = polars_df.select(element_at("m", "a").alias("v"))
        spark_result = spark_df.select(spark_element_at(spark_col("m"), spark_lit("a")).alias("v"))
        assert_sparkle_spark_frame_are_equal(sf_result, spark_result)

    def test_map_missing_literal_key_returns_null(self, spark) -> None:
        df = pl.DataFrame({"m": [[{"key": "a", "value": 1.0}, {"key": "b", "value": 2.0}]]})
        polars_df = DataFrame(df)
        spark_df = spark.createDataFrame(
            [({"a": 1.0, "b": 2.0},)],
            schema=SparkStructType([SparkStructField("m", SparkMapType(SparkStringType(), SparkDoubleType()), True)]),
        )
        sf_result = polars_df.select(element_at("m", "c").alias("v"))
        spark_result = spark_df.select(spark_element_at(spark_col("m"), spark_lit("c")).alias("v"))
        assert_sparkle_spark_frame_are_equal(sf_result, spark_result)


class TestSubstring:
    def test_substring_against_spark(self, spark) -> None:
        data = {"s": ["hELlo woRLd", None, ""]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_substring("s", spark_lit(2), spark_lit(3)).alias("sub"))
        result_df = polars_df.select(substring("s", 2, 3).alias("sub"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestArrayFunctions:
    def test_array_contains_and_size_against_spark(self, spark) -> None:
        data = {"arr": [[1, 2, 3], [10]]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(
            spark_array_contains("arr", spark_lit(2)).alias("has2"),
            spark_size("arr").alias("sz"),
        )
        result_df = polars_df.select(
            array_contains("arr", 2).alias("has2"),
            size("arr").alias("sz"),
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_array_filter_and_transform_against_spark(self, spark) -> None:
        data = {"arr": [[1, 2, 3], [10]]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(
            spark_filter("arr", lambda x: x > 1).alias("flt"),
            spark_transform("arr", lambda x: x * 2).alias("dbl"),
        )
        result_df = polars_df.select(
            filter("arr", lambda c: c > 1).alias("flt"),
            transform("arr", lambda c: c * 2).alias("dbl"),
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_explode_against_spark(self, spark) -> None:
        data = {"a": [[1, 2], None, []]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_explode_outer("a").alias("e"))
        result_df = polars_df.select(explode("a").alias("e"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestDateFunctions:
    def test_date_sub_and_datediff_against_spark(self, spark) -> None:
        data = {"d": [None, "2024-01-10", "2024-01-01"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(
            spark_date_sub(spark_col("d"), spark_lit(2)).alias("sub"),
            spark_datediff(spark_lit("2024-01-20"), spark_col("d")).alias("dd"),
        )
        result_df = polars_df.select(
            date_sub("d", 2).alias("sub"),
            datediff(lit("2024-01-20"), "d").alias("dd"),
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_datediff_iso8601_string_against_spark(self, spark) -> None:
        data = {"a": ["2026-04-26T00:00:00Z"]}
        end = "2026-04-27"
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(spark_datediff(spark_lit(end), spark_col("a")).alias("d"))
        result_df = polars_df.select(datediff(lit(end), "a").alias("d"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_datetime_ge_date_sub_against_spark(self, spark) -> None:
        data = {"created": ["2026-04-26T00:00:00Z", "2026-01-01T00:00:00Z", None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        expected_df = spark_df.select(
            (spark_col("created") >= spark_date_sub(spark_current_date(), spark_lit(30))).alias("passes_30d")
        )
        result_df = polars_df.select((col("created") >= date_sub(current_date(), 30)).alias("passes_30d"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_months_between_against_spark(self, spark) -> None:
        data = {"start": ["2024-01-15"], "end": ["2024-03-20"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_spark_df = create_spark_df(spark, polars_df.select(months_between("end", "start").alias("mb")))
        expected_df = spark_df.select(spark_months_between("end", "start").alias("mb"))
        assert_pyspark_df_equal(result_spark_df, expected_df, precision=8)


class TestMapFunctions:
    def test_map_keys_against_spark(self, spark) -> None:
        data = {"m": [{"a": 1.0, "b": 2.0}]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(
            [({"a": 1.0, "b": 2.0},)],
            schema=SparkStructType([SparkStructField("m", SparkMapType(SparkStringType(), SparkDoubleType()), True)]),
        )
        expected_df = spark_df.select(spark_map_keys("m").alias("keys"))
        result_df = polars_df.select(map_keys("m").alias("keys"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_map_from_entries_against_spark(self, spark) -> None:
        """map_from_entries produces a map that getItem can look up by key,
        matching Spark's MapType semantics."""
        from pyspark.sql import Row

        entries = [Row(key="a", value=1), Row(key="b", value=2)]
        spark_df = spark.createDataFrame([Row(entries=entries)])
        polars_df = DataFrame(pl.DataFrame({"entries": [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]]}))
        expected_df = spark_df.select(spark_map_from_entries("entries").alias("m")).select(
            spark_col("m").getItem("a").alias("val_a")
        )
        result_df = polars_df.select(map_from_entries("entries").alias("m")).select(
            col("m").getItem("a").alias("val_a")
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestFromJson:
    def test_from_json_struct_against_spark(self, spark) -> None:
        payload = '{"field1": "hello", "field2": 999}'
        data = {"j": [payload]}
        schema = StructType([StructField("field1", StringType()), StructField("field2", IntegerType())])
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        spark_schema = SparkStructType(
            [
                SparkStructField("field1", SparkStringType()),
                SparkStructField("field2", SparkIntegerType()),
            ]
        )
        expected_df = spark_df.select(spark_from_json("j", spark_schema).alias("parsed"))
        result_df = polars_df.select(from_json("j", schema).alias("parsed"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_from_json_malformed_returns_null(self, spark) -> None:
        schema = StructType([StructField("field1", StringType()), StructField("field2", IntegerType())])
        spark_schema = SparkStructType(
            [
                SparkStructField("field1", SparkStringType()),
                SparkStructField("field2", SparkIntegerType()),
            ]
        )
        valid_data = {"j": ['{"field1": "ok", "field2": 1}']}
        polars_valid = DataFrame(pl.DataFrame(valid_data))
        spark_valid = spark.createDataFrame(spark_rows_from_dict(valid_data), list(valid_data.keys()))
        assert_sparkle_spark_frame_are_equal(
            polars_valid.select(from_json("j", schema).alias("parsed")),
            spark_valid.select(spark_from_json("j", spark_schema).alias("parsed")),
        )

        bad_data = {"j": ["not-json"]}
        polars_bad = DataFrame(pl.DataFrame(bad_data))
        result = polars_bad.select(from_json("j", schema).alias("parsed")).to_native_df()
        assert result["parsed"][0] is None

        spark_bad = spark.createDataFrame(spark_rows_from_dict(bad_data), list(bad_data.keys()))
        spark_row = spark_bad.select(spark_from_json("j", spark_schema).alias("parsed")).collect()[0][0]
        assert spark_row is None or (spark_row.field1 is None and spark_row.field2 is None)


class TestFirstAgg:
    def test_first_agg_against_spark(self, spark) -> None:
        data = {"g": [1, 1, 2, 2], "v": [10, 20, 30, 40]}
        polars_df = DataFrame(pl.DataFrame(data)).sort("g", "v")
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).orderBy("g", "v")
        expected_df = spark_df.groupBy("g").agg(spark_first("v").alias("fv"))
        result_df = polars_df.groupBy("g").agg(first("v").alias("fv"))
        assert_sparkle_spark_frame_are_equal(result_df.orderBy("g"), expected_df.orderBy("g"))


class TestBroadcast:
    def test_broadcast_returns_same_dataframe(self) -> None:
        d = DataFrame(pl.DataFrame({"x": [1]}))
        assert broadcast(d) is d


class TestWhenLitNonePreservesType:
    def test_when_lit_none_preserves_double_type(self, spark) -> None:
        """lit(None) inside when/then must not force the result to String.
        PySpark infers the type from the non-null branch; sparkleframe must match."""
        data = {"price": [10.0, -1.0, 5.0]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.withColumn("price", when(col("price") <= 0, lit(None)).otherwise(col("price")))
        expected_df = spark_df.withColumn(
            "price", spark_when(spark_col("price") <= 0, spark_lit(None)).otherwise(spark_col("price"))
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_when_lit_none_preserves_integer_type(self, spark) -> None:
        data = {"v": [1, -2, 3]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.withColumn("v", when(col("v") < 0, lit(None)).otherwise(col("v")))
        expected_df = spark_df.withColumn(
            "v", spark_when(spark_col("v") < 0, spark_lit(None)).otherwise(spark_col("v"))
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_when_lit_none_preserves_string_type(self, spark) -> None:
        data = {"s": ["hello", "", "world"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.withColumn("s", when(col("s") == "", lit(None)).otherwise(col("s")))
        expected_df = spark_df.withColumn(
            "s", spark_when(spark_col("s") == "", spark_lit(None)).otherwise(spark_col("s"))
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestToJsonParity:
    def test_to_json_struct_against_spark(self, spark) -> None:
        data = {"a": [1, 2], "b": ["x", "y"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(to_json(struct("a", "b")).alias("j"))
        expected_df = spark_df.select(spark_to_json(spark_struct("a", "b")).alias("j"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_to_json_array_against_spark(self, spark) -> None:
        from pyspark.sql.functions import array as spark_array

        data = {"a": [1, 2], "b": [3, 4]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(to_json(array("a", "b")).alias("j"))
        expected_df = spark_df.select(spark_to_json(spark_array("a", "b")).alias("j"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestLeastGreatest:
    def test_least_against_spark(self, spark) -> None:
        data = {"a": [10, 1, None], "b": [5, None, 3], "c": [8, 2, 7]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(least("a", "b", "c").alias("min"))
        expected_df = spark_df.select(spark_least("a", "b", "c").alias("min"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_greatest_against_spark(self, spark) -> None:
        data = {"a": [10, 1, None], "b": [5, None, 3], "c": [8, 2, 7]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(greatest("a", "b", "c").alias("max"))
        expected_df = spark_df.select(spark_greatest("a", "b", "c").alias("max"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_least_all_nulls_returns_null(self, spark) -> None:
        polars_df = DataFrame(
            pl.DataFrame({"a": pl.Series([None, None], dtype=pl.Int64), "b": pl.Series([None, None], dtype=pl.Int64)})
        )
        from pyspark.sql.types import LongType as SparkLongType

        schema = SparkStructType(
            [SparkStructField("a", SparkLongType(), True), SparkStructField("b", SparkLongType(), True)]
        )
        spark_df = spark.createDataFrame([(None, None), (None, None)], schema)
        result_df = polars_df.select(least("a", "b").alias("min"))
        expected_df = spark_df.select(spark_least("a", "b").alias("min"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestCreateMap:
    def test_create_map_against_spark(self, spark) -> None:
        data = {"k": ["a", "b"], "v": [1, 2]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(to_json(create_map("k", "v")).alias("j"))
        expected_df = spark_df.select(spark_to_json(spark_create_map("k", "v")).alias("j"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestArrayParity:
    def test_array_against_spark(self, spark) -> None:
        from pyspark.sql.functions import array as spark_array

        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(array("a", "b").alias("arr"))
        expected_df = spark_df.select(spark_array("a", "b").alias("arr"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestDateFormat:
    def test_date_format_against_spark(self, spark) -> None:
        data = {"ts": ["2023-01-15 10:30:45", "2024-12-25 00:00:00"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))

        sf_ts = polars_df.select(date_format(to_timestamp("ts"), "yyyy-MM-dd").alias("d"))
        sp_ts = spark_df.select(spark_date_format(spark_to_timestamp("ts"), "yyyy-MM-dd").alias("d"))
        assert_sparkle_spark_frame_are_equal(sf_ts, sp_ts)

    def test_date_format_time_parts_against_spark(self, spark) -> None:
        data = {"ts": ["2023-06-15 14:05:09"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))

        sf = polars_df.select(date_format(to_timestamp("ts"), "HH:mm:ss").alias("t"))
        sp = spark_df.select(spark_date_format(spark_to_timestamp("ts"), "HH:mm:ss").alias("t"))
        assert_sparkle_spark_frame_are_equal(sf, sp)


class TestFloor:
    def test_floor_against_spark(self, spark) -> None:
        data = {"v": [1.9, 2.1, -0.5, 0.0, None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(floor("v").alias("f"))
        expected_df = spark_df.select(spark_floor("v").alias("f"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestPow:
    def test_pow_against_spark(self, spark) -> None:
        data = {"base": [2.0, 3.0, 10.0], "exp": [3.0, 2.0, 0.0]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(pow("base", "exp").alias("p"))
        expected_df = spark_df.select(spark_pow("base", "exp").alias("p"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_pow_with_literal_exponent_against_spark(self, spark) -> None:
        data = {"base": [2.0, 3.0, 4.0]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(pow(col("base"), lit(2)).alias("p"))
        expected_df = spark_df.select(spark_pow(spark_col("base"), spark_lit(2)).alias("p"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestIsnan:
    def test_isnan_against_spark(self, spark) -> None:
        data = {"v": [1.0, float("nan"), None, 0.0]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(isnan("v").alias("n"))
        expected_df = spark_df.select(spark_isnan("v").alias("n"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestTryDivide:
    def test_try_divide_against_spark(self, spark) -> None:
        data = {"a": [10.0, 9.0, None, 5.0], "b": [2.0, 0.0, 3.0, None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(try_divide("a", "b").alias("d"))
        expected_df = spark_df.select(spark_try_divide("a", "b").alias("d"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestNullif:
    def test_nullif_against_spark(self, spark) -> None:
        data = {"a": [1, 2, 3, None], "b": [1, 3, 3, None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(nullif("a", "b").alias("n"))
        expected_df = spark_df.select(spark_nullif("a", "b").alias("n"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_nullif_with_nulls_preserves_e1(self, spark) -> None:
        data = {"a": [5, None, 3], "b": [None, 2, None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(nullif("a", "b").alias("n"))
        expected_df = spark_df.select(spark_nullif("a", "b").alias("n"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestRandParity:
    def test_rand_output_shape_and_type_matches_spark(self, spark) -> None:
        data = {"x": [1, 2, 3, 4, 5]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        sf_result = polars_df.select(rand(42).alias("r"))
        sp_result = spark_df.select(spark_rand(42).alias("r"))
        sf_vals = sf_result.to_native_df()["r"].to_list()
        sp_vals = [row[0] for row in sp_result.collect()]
        assert len(sf_vals) == len(sp_vals)
        assert all(isinstance(v, float) and 0.0 <= v < 1.0 for v in sf_vals)
        assert all(isinstance(v, float) and 0.0 <= v < 1.0 for v in sp_vals)


class TestSortArray:
    def test_sort_array_asc_against_spark(self, spark) -> None:
        data = {"arr": [[3, 1, 2], [6, 4, 5], None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(sort_array("arr").alias("s"))
        expected_df = spark_df.select(spark_sort_array("arr").alias("s"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_sort_array_desc_against_spark(self, spark) -> None:
        data = {"arr": [[3, 1, 2], [6, 4, 5]]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(sort_array("arr", asc=False).alias("s"))
        expected_df = spark_df.select(spark_sort_array("arr", asc=False).alias("s"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestCount:
    def test_count_star_against_spark(self, spark) -> None:
        data = {"g": ["a", "a", "b", "b", "b"], "v": [1, None, 3, 4, None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.groupBy("g").agg(count("*").alias("cnt")).orderBy("g")
        expected_df = spark_df.groupBy("g").agg(spark_count("*").alias("cnt")).orderBy("g")
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_count_column_against_spark(self, spark) -> None:
        data = {"g": ["a", "a", "b", "b", "b"], "v": [1, None, 3, 4, None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.groupBy("g").agg(count("v").alias("cnt")).orderBy("g")
        expected_df = spark_df.groupBy("g").agg(spark_count("v").alias("cnt")).orderBy("g")
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_count_star_no_groupby_against_spark(self, spark) -> None:
        data = {"v": [1, None, 3]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(count("*").alias("cnt"))
        expected_df = spark_df.select(spark_count("*").alias("cnt"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestAbs:
    def test_abs_against_spark(self, spark) -> None:
        data = {"v": [-3.5, 0.0, 2.1, -7.0, None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(abs(col("v")).alias("a"))
        expected_df = spark_df.select(spark_abs(spark_col("v")).alias("a"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_abs_integer_against_spark(self, spark) -> None:
        data = {"v": [-10, 0, 5, -1, None]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(abs("v").alias("a"))
        expected_df = spark_df.select(spark_abs("v").alias("a"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_abs_expression_against_spark(self, spark) -> None:
        data = {"a": [1.0, 5.0, 3.0], "b": [4.0, 2.0, 3.0]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.withColumn("diff", abs(col("a") - col("b")))
        expected_df = spark_df.withColumn("diff", spark_abs(spark_col("a") - spark_col("b")))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_abs_of_subtraction_withcolumn_against_spark(self, spark) -> None:
        """Regression: arithmetic with unresolved dtypes chained with abs() via withColumn
        must not produce an untyped UDF (Polars rejects map_batches without return_dtype)."""
        data = {"x": [1.0, 5.0, 3.0], "y": [4.0, 2.0, 3.0]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.withColumn("d", abs(col("x") - col("y")))
        expected_df = spark_df.withColumn("d", spark_abs(spark_col("x") - spark_col("y")))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestStructFieldNaming:
    def test_struct_child_field_name_with_alias_on_element_expr(self) -> None:
        """Regression: _struct_child_field_name must not crash on expressions derived
        from pl.element() (no root column name). The alias set on the Column wrapper
        should be used as the field name."""
        from sparkleframe.polarsdf.functions import _struct_child_field_name

        elem_col = col("x").alias("key")
        expr = elem_col.to_native()
        name = _struct_child_field_name(elem_col, expr, 0)
        assert name == "key"

    def test_struct_child_field_name_aliased_literal(self) -> None:
        """lit(...).alias('name') must use the alias, not col{N+1}."""
        from sparkleframe.polarsdf.functions import _struct_child_field_name

        aliased_lit = lit("hello").alias("greeting")
        expr = aliased_lit.to_native()
        name = _struct_child_field_name(aliased_lit, expr, 0)
        assert name == "greeting"

    def test_struct_child_field_name_fallback_without_alias(self) -> None:
        """When output_name() fails and no alias is set, fall back to col{N+1}."""
        from sparkleframe.polarsdf.functions import _struct_child_field_name

        class _FakeExprMeta:
            def output_name(self):
                raise Exception("no root column")

            def undo_aliases(self):
                raise Exception("no root column")

            def serialize(self):
                raise Exception("no root column")

        class _FakeExpr:
            meta = _FakeExprMeta()

        plain_col = col("x")
        plain_col._output_alias = None
        name = _struct_child_field_name(plain_col, _FakeExpr(), 2)
        assert name == "col3"

    def test_struct_from_aliased_columns_against_spark(self, spark) -> None:
        from pyspark.sql.functions import struct as spark_struct

        data = {"a": [1, 2], "b": ["x", "y"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(struct(col("a").alias("k"), col("b").alias("v")).alias("s"))
        expected_df = spark_df.select(spark_struct(spark_col("a").alias("k"), spark_col("b").alias("v")).alias("s"))
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_struct_from_aliased_literals_against_spark(self, spark) -> None:
        """lit(...).alias('name') inside struct must use the alias as the field name."""
        data = {"x": [1, 2]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(
            struct(
                lit("aaa").alias("f1"),
                lit("bbb").alias("f2"),
                lit("ccc").alias("f3"),
            ).alias("s")
        )
        expected_df = spark_df.select(
            spark_struct(
                spark_lit("aaa").alias("f1"),
                spark_lit("bbb").alias("f2"),
                spark_lit("ccc").alias("f3"),
            ).alias("s")
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_to_json_struct_aliased_literals_against_spark(self, spark) -> None:
        """to_json(struct(lit(...).alias(...), ...)) must use alias names as JSON keys."""
        data = {"x": [1, 2]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(
            to_json(
                struct(
                    lit("aaa").alias("f1"),
                    lit("bbb").alias("f2"),
                    lit("ccc").alias("f3"),
                )
            ).alias("j")
        )
        expected_df = spark_df.select(
            spark_to_json(
                spark_struct(
                    spark_lit("aaa").alias("f1"),
                    spark_lit("bbb").alias("f2"),
                    spark_lit("ccc").alias("f3"),
                )
            ).alias("j")
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)

    def test_struct_mixed_aliased_lit_and_plain_col_against_spark(self, spark) -> None:
        """Struct with a mix of aliased literals and plain column references."""
        data = {"a": [1, 2], "b": ["x", "y"]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        result_df = polars_df.select(
            struct(
                col("b"),
                lit("fixed").alias("tag"),
            ).alias("s")
        )
        expected_df = spark_df.select(
            spark_struct(
                spark_col("b"),
                spark_lit("fixed").alias("tag"),
            ).alias("s")
        )
        assert_sparkle_spark_frame_are_equal(result_df, expected_df)


class TestTransformStruct:
    def test_transform_struct_and_explode_against_spark(self, spark) -> None:
        """Regression: transform with a struct-producing lambda must not produce
        FixedSizeBinary. The result must be explodable and match PySpark."""
        data = {"items": [["a=1", "b=2"], ["c=3"]]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))

        sf = polars_df.withColumn("items", transform(col("items"), lambda x: split(x, "=", 2)))
        sp = spark_df.withColumn("items", spark_transform(spark_col("items"), lambda x: spark_split(x, "=", 2)))

        sf = sf.withColumn(
            "items",
            transform(
                col("items"),
                lambda x: struct(element_at(x, 1).alias("key"), element_at(x, 2).alias("value")),
            ),
        )
        sp = sp.withColumn(
            "items",
            spark_transform(
                spark_col("items"),
                lambda x: spark_struct(spark_element_at(x, 1).alias("key"), spark_element_at(x, 2).alias("value")),
            ),
        )

        sf_exploded = sf.withColumn("item", explode("items")).select("item")
        sp_exploded = sp.withColumn("item", spark_explode_outer(spark_col("items"))).select("item")
        assert_sparkle_spark_frame_are_equal(sf_exploded, sp_exploded)

    def test_transform_struct_inside_when_otherwise_against_spark(self, spark) -> None:
        """Regression: transform+struct wrapped in when/otherwise must preserve
        List(Struct) type — lit(None) must not collapse the column to Null."""
        data = {"items": [["a=1", "b=2"], None, ["c=3"]]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))

        def struct_lambda_sf(x):
            return struct(element_at(x, 1).alias("key"), element_at(x, 2).alias("value"))

        def struct_lambda_sp(x):
            return spark_struct(spark_element_at(x, 1).alias("key"), spark_element_at(x, 2).alias("value"))

        sf = polars_df.withColumn("items", transform(col("items"), lambda x: split(x, "=", 2)))
        sp = spark_df.withColumn("items", spark_transform(spark_col("items"), lambda x: spark_split(x, "=", 2)))

        sf = sf.withColumn(
            "items",
            when(col("items").isNotNull(), transform(col("items"), struct_lambda_sf)).otherwise(lit(None)),
        )
        sp = sp.withColumn(
            "items",
            spark_when(
                spark_col("items").isNotNull(), spark_transform(spark_col("items"), struct_lambda_sp)
            ).otherwise(spark_lit(None)),
        )

        sf = sf.withColumn("items", filter(col("items"), lambda x: x.isNotNull()))
        sp = sp.withColumn("items", spark_filter(spark_col("items"), lambda x: x.isNotNull()))
        assert_sparkle_spark_frame_are_equal(sf, sp)


class TestFilterStructField:
    def test_filter_by_struct_field_equality_against_spark(self, spark) -> None:
        """Regression: filter lambda using getItem + == on a List(Struct) column must
        produce native expressions (not Object-typed UDFs) so list.eval works."""
        data = {
            "pairs": [
                [{"key": "utm_source", "value": "google"}, {"key": "utm_medium", "value": "cpc"}],
                [{"key": "other", "value": "x"}],
            ]
        }
        from pyspark.sql.types import ArrayType as SparkArrayType
        from pyspark.sql.types import StringType as SparkStringType
        from pyspark.sql.types import StructField as SparkStructField
        from pyspark.sql.types import StructType as SparkStructType

        spark_schema = SparkStructType(
            [
                SparkStructField(
                    "pairs",
                    SparkArrayType(
                        SparkStructType(
                            [SparkStructField("key", SparkStringType()), SparkStructField("value", SparkStringType())]
                        )
                    ),
                )
            ]
        )
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), schema=spark_schema)

        sf = polars_df.withColumn("pairs", filter(col("pairs"), lambda x: x.getItem("key") == lit("utm_source")))
        sp = spark_df.withColumn(
            "pairs",
            spark_filter(spark_col("pairs"), lambda x: x.getItem("key") == spark_lit("utm_source")),
        )
        assert_sparkle_spark_frame_are_equal(sf, sp)

    def test_getitem_on_struct_inside_transform_against_spark(self, spark) -> None:
        """getItem on a struct element inside transform must return typed values,
        not Object-typed UDF results."""
        data = {
            "pairs": [
                [{"key": "a", "value": "1"}, {"key": "b", "value": "2"}],
            ]
        }
        from pyspark.sql.types import ArrayType as SparkArrayType
        from pyspark.sql.types import StringType as SparkStringType
        from pyspark.sql.types import StructField as SparkStructField
        from pyspark.sql.types import StructType as SparkStructType

        spark_schema = SparkStructType(
            [
                SparkStructField(
                    "pairs",
                    SparkArrayType(
                        SparkStructType(
                            [SparkStructField("key", SparkStringType()), SparkStructField("value", SparkStringType())]
                        )
                    ),
                )
            ]
        )
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), schema=spark_schema)

        sf = polars_df.withColumn("keys", transform(col("pairs"), lambda x: x.getItem("key")))
        sp = spark_df.withColumn("keys", spark_transform(spark_col("pairs"), lambda x: x.getItem("key")))
        assert_sparkle_spark_frame_are_equal(sf, sp)


class TestMapFromEntries:
    """Parity tests for map_from_entries and related map operations."""

    def test_map_from_entries_against_spark(self, spark) -> None:
        """map_from_entries on List(Struct(key, value)) should produce a map
        that getItem can look up by key."""
        data = {
            "entries": [
                [{"key": "utm_source", "value": "google"}, {"key": "utm_medium", "value": "cpc"}],
                [{"key": "plan", "value": "premium"}],
            ]
        }
        spark_schema = SparkStructType(
            [
                SparkStructField(
                    "entries",
                    SparkArrayType(
                        SparkStructType(
                            [SparkStructField("key", SparkStringType()), SparkStructField("value", SparkStringType())]
                        )
                    ),
                )
            ]
        )
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), schema=spark_schema)

        sf = polars_df.withColumn("m", map_from_entries(col("entries"))).withColumn(
            "src", col("m").getItem("utm_source")
        )
        sp = spark_df.withColumn("m", spark_map_from_entries(spark_col("entries"))).withColumn(
            "src", spark_col("m").getItem("utm_source")
        )
        assert_sparkle_spark_frame_are_equal(
            sf.select("src"),
            sp.select("src"),
        )

    def test_create_map_getitem_against_spark(self, spark) -> None:
        """create_map + getItem should look up a value by key."""
        polars_df = DataFrame(pl.DataFrame({"x": ["hello"], "y": ["world"]}))
        spark_df = spark.createDataFrame(pd.DataFrame({"x": ["hello"], "y": ["world"]}))

        sf = polars_df.withColumn("m", create_map(lit("x_val"), col("x"), lit("y_val"), col("y"))).withColumn(
            "got", col("m").getItem("x_val")
        )
        sp = spark_df.withColumn(
            "m", spark_create_map(spark_lit("x_val"), spark_col("x"), spark_lit("y_val"), spark_col("y"))
        ).withColumn("got", spark_col("m").getItem("x_val"))
        assert_sparkle_spark_frame_are_equal(
            sf.select("got"),
            sp.select("got"),
        )


class TestElementAtWithLitIndex:
    """Parity tests for element_at / try_element_at with F.lit(int) index."""

    def test_try_element_at_with_lit_int_against_spark(self, spark) -> None:
        """try_element_at(array, F.lit(1)) should extract the first element."""
        spark_schema = SparkStructType([SparkStructField("arr", SparkArrayType(SparkStringType()))])
        polars_df = DataFrame(pl.DataFrame({"arr": [["a", "b", "c"]]}))
        spark_df = spark.createDataFrame([(["a", "b", "c"],)], schema=spark_schema)

        sf = polars_df.withColumn("first", try_element_at(col("arr"), lit(1)))
        sp = spark_df.withColumn("first", spark_try_element_at(spark_col("arr"), spark_lit(1)))
        assert_sparkle_spark_frame_are_equal(sf.select("first"), sp.select("first"))

    def test_element_at_with_lit_int_against_spark(self, spark) -> None:
        """element_at(array, F.lit(2)) should extract the second element."""
        spark_schema = SparkStructType([SparkStructField("arr", SparkArrayType(SparkStringType()))])
        polars_df = DataFrame(pl.DataFrame({"arr": [["x", "y", "z"]]}))
        spark_df = spark.createDataFrame([(["x", "y", "z"],)], schema=spark_schema)

        sf = polars_df.withColumn("second", element_at(col("arr"), lit(2)))
        sp = spark_df.withColumn("second", spark_element_at(spark_col("arr"), spark_lit(2)))
        assert_sparkle_spark_frame_are_equal(sf.select("second"), sp.select("second"))


class TestGetItemOnListInFilter:
    """Parity tests for getItem on a list column used inside filter."""

    def test_getitem_int_in_filter_against_spark(self, spark) -> None:
        """filter(col('arr').getItem(0).isNotNull()) should keep non-null first elements."""
        spark_schema = SparkStructType([SparkStructField("arr", SparkArrayType(SparkStringType()))])
        polars_df = DataFrame(pl.DataFrame({"arr": [["a", "b"], None, ["c"]]}))
        spark_df = spark.createDataFrame(
            [(["a", "b"],), (None,), (["c"],)],
            schema=spark_schema,
        )

        sf = polars_df.filter(col("arr").getItem(0).isNotNull())
        sp = spark_df.filter(spark_col("arr").getItem(0).isNotNull())
        assert_sparkle_spark_frame_are_equal(sf, sp)

    def test_getitem_string_key_on_map_in_filter_against_spark(self, spark) -> None:
        """getItem on a map-as-struct column inside filter must resolve values."""
        data = {
            "entries": [
                [{"key": "a", "value": "1"}],
                [{"key": "b", "value": "2"}],
            ]
        }
        spark_schema = SparkStructType(
            [
                SparkStructField(
                    "entries",
                    SparkArrayType(
                        SparkStructType(
                            [SparkStructField("key", SparkStringType()), SparkStructField("value", SparkStringType())]
                        )
                    ),
                )
            ]
        )
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(spark_rows_from_dict(data), schema=spark_schema)

        sf = (
            polars_df.withColumn("m", map_from_entries(col("entries")))
            .withColumn("got", col("m").getItem("a"))
            .filter(col("got").isNotNull())
        )
        sp = (
            spark_df.withColumn("m", spark_map_from_entries(spark_col("entries")))
            .withColumn("got", spark_col("m").getItem("a"))
            .filter(spark_col("got").isNotNull())
        )
        assert_sparkle_spark_frame_are_equal(
            sf.select("got"),
            sp.select("got"),
        )
