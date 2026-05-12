"""
PySpark 4.x behaviour parity for recently added DataFrame / Column / functions APIs.

These tests treat Spark 4.1 (see requirements-dev) as the reference implementation.
"""

from __future__ import annotations

import polars as pl
import pytest
from pyspark.sql import functions as F
from pyspark.sql.functions import col as spark_col
from pyspark.sql.types import IntegerType as SparkIntegerType
from pyspark.sql.types import LongType as SparkLongType
from pyspark.sql.types import StructField, StructType

import sparkleframe.polarsdf.functions as PF
from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.tests.pyspark_test import assert_pyspark_df_equal
from sparkleframe.tests.utils import create_spark_df, spark_rows_from_dict

_EMPTY_ID_SCHEMA = StructType([StructField("id", SparkLongType(), True)])


class TestToTimestampFormatParity:
    """``_to_datetime_column``: format-based parse aligned with Spark (no extra ISO fallback)."""

    @pytest.mark.parametrize(
        "fmt",
        ["yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd H:m:s"],
    )
    def test_valid_string_matches_spark(self, spark, fmt: str) -> None:
        data = {"ts": ["2024-03-15 10:20:30", None]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        for fn in (PF.to_timestamp, PF.try_to_timestamp):
            got = create_spark_df(
                spark,
                pl_df.select(fn("ts", fmt).alias("t")),
            )
            # Use SQL expr — PySpark 4 Python API for 2-arg to_timestamp can differ by build.
            exp = sdf.selectExpr(f"to_timestamp(ts, '{fmt}') as t")
            assert_pyspark_df_equal(got, exp, ignore_nullable=True)
            try_exp = sdf.selectExpr(f"try_to_timestamp(ts, '{fmt}') as t")
            assert_pyspark_df_equal(got, try_exp, ignore_nullable=True)

    def test_iso_t_separator_does_not_match_space_format(self, spark) -> None:
        """Null for T-separated value; Spark 4 ANSI ``to_timestamp`` would fail the stage — align with try_to."""
        data = {"ts": ["2024-03-15T10:20:30", "2024-03-15 10:20:30"]}
        fmt = "yyyy-MM-dd HH:mm:ss"
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(spark, pl_df.select(PF.to_timestamp("ts", fmt).alias("t")))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        exp = sdf.selectExpr(f"try_to_timestamp(ts, '{fmt}') as t")
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)


class TestToTimestampOneArgParity:
    """Omitted ``fmt``: PySpark 4 one-arg :func:`to_timestamp` matches ``cast("timestamp")`` on strings."""

    def test_parses_iso8601_z_like_cast(self, spark) -> None:
        s = "2026-04-26T00:00:00Z"
        data = {"createdOn": [s]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        exp = sdf.select(F.to_timestamp("createdOn").alias("t"))
        for fn in (PF.to_timestamp, PF.try_to_timestamp):
            got = create_spark_df(spark, pl_df.select(fn("createdOn").alias("t")))
            assert_pyspark_df_equal(got, exp, ignore_nullable=True)
        got_cast = create_spark_df(spark, pl_df.select(PF.col("createdOn").cast(PF.TimestampType()).alias("t")))
        assert_pyspark_df_equal(got_cast, exp, ignore_nullable=True)


class TestColumnCastAnsiNullsSpark4:
    """
    Per-row null on invalid cast (Polars ``cast(..., strict=False)``), like ``try_cast`` in Spark 4.

    A plain ``cast`` in Spark 4 with ANSI can **fail the query** on the first bad value; use
    ``try_cast`` in tests as the comparable reference for null-on-invalid behaviour.
    """

    def test_invalid_string_to_int_null_against_spark_try_cast(self, spark) -> None:
        from sparkleframe.polarsdf.types import IntegerType

        data = {"s": ["42", "not_int", None]}
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(PF.col("s").cast(IntegerType()).alias("n")),
        )
        # Spark: ``try_cast`` lives on ``Column`` (not ``functions``) in PySpark 4.1+.
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            spark_col("s").try_cast(SparkIntegerType()).alias("n")
        )
        assert_pyspark_df_equal(
            got.select(spark_col("n").cast(SparkIntegerType()).alias("n")),
            exp,
            ignore_nullable=True,
        )


class TestStringAndJsonFunctions:
    def test_initcap_substring_against_spark(self, spark) -> None:
        data = {"s": ["hELlo woRLd", None, ""]}
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(
                PF.initcap("s").alias("ic"),
                PF.substring("s", 2, 3).alias("sub"),
            ),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            F.initcap("s").alias("ic"),
            F.substring("s", F.lit(2), F.lit(3)).alias("sub"),
        )
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)


class TestArrayFunctions:
    def test_array_contains_and_size_against_spark(self, spark) -> None:
        # Omit null array rows: Spark createDataFrame inference fails on (bool, null) mixes.
        data = {
            "arr": [
                [1, 2, 3],
                [10],
            ],
        }
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(
                PF.array_contains("arr", 2).alias("has2"),
                PF.size("arr").alias("sz"),
            ),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            F.array_contains("arr", F.lit(2)).alias("has2"),
            F.size("arr").alias("sz"),
        )
        assert_pyspark_df_equal(
            got.select(spark_col("has2"), spark_col("sz").cast(SparkIntegerType()).alias("sz")),
            exp,
            ignore_nullable=True,
        )

    @pytest.mark.parametrize("asc", [True, False])
    def test_sort_array_int_and_empty_matches_spark(self, spark, asc: bool) -> None:
        data = {"arr": [[3, 1, 2], None, []]}
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(spark, pl_df.select(PF.sort_array("arr", asc).alias("s")))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        exp = sdf.select(F.sort_array(spark_col("arr"), asc).alias("s"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_sort_array_strings_matches_spark(self, spark) -> None:
        data = {"arr": [["b", "a"], None, []]}
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(spark, pl_df.select(PF.sort_array("arr").alias("s")))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        exp = sdf.select(F.sort_array(spark_col("arr")).alias("s"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_array_filter_and_transform_against_spark(self, spark) -> None:
        data = {
            "arr": [
                [1, 2, 3],
                [10],
            ],
        }
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(
                PF.filter("arr", lambda c: c > 1).alias("flt"),
                PF.transform("arr", lambda c: c * 2).alias("dbl"),
            ),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            F.filter("arr", lambda x: x > 1).alias("flt"),
            F.transform("arr", lambda x: x * 2).alias("dbl"),
        )
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_explode_against_spark(self, spark) -> None:
        data = {"a": [[1, 2], None, []]}
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(PF.explode("a").alias("e")),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            F.explode_outer("a").alias("e")
        )
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)


class TestDateFunctions:
    def test_date_sub_datediff_against_spark(self, spark) -> None:
        data = {
            "d": [None, "2024-01-10", "2024-01-01"],
        }
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(
                PF.date_sub("d", 2).alias("sub"),
                PF.datediff(PF.lit("2024-01-20"), "d").alias("dd"),
            ),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            F.date_sub(spark_col("d"), F.lit(2)).alias("sub"),
            F.datediff(F.lit("2024-01-20"), spark_col("d")).alias("dd"),
        )
        assert_pyspark_df_equal(
            got.withColumn("dd", spark_col("dd").cast("int")),
            exp,
            ignore_nullable=True,
        )

    def test_datediff_iso8601_string_fixed_end_vs_spark(self, spark) -> None:
        """``datediff`` with string ``…T…Z`` must match Spark (not null from plain str→date cast)."""
        data = {"a": ["2026-04-26T00:00:00Z"]}
        pl_df = DataFrame(pl.DataFrame(data))
        # Fixed end date for deterministic test (same in Spark + PF).
        end = "2026-04-27"
        got = create_spark_df(
            spark,
            pl_df.select(PF.datediff(PF.lit(end), "a").alias("d")),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            F.datediff(F.lit(end), spark_col("a")).alias("d")
        )
        assert_pyspark_df_equal(
            got.withColumn("d", spark_col("d").cast("int")),
            exp,
            ignore_nullable=True,
        )

    def test_datetime_ge_date_sub_matches_spark(self, spark) -> None:
        """Offer-age style filter: ``datetime_created >= date_sub(current_date(), n)`` must not be all-null."""
        data = {"created": ["2026-04-26T00:00:00Z", "2026-01-01T00:00:00Z", None]}
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(
                (PF.col("created") >= PF.date_sub(PF.current_date(), 30)).alias("passes_30d"),
            ),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            (spark_col("created") >= F.date_sub(F.current_date(), F.lit(30))).alias("passes_30d")
        )
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)


class TestDateFormatParity:
    """``date_format``: Spark pattern letters → string, aligned with PySpark 4."""

    def test_date_and_timestamp_columns_match_spark(self, spark) -> None:
        from datetime import date, datetime

        data = {
            "d": [date(2024, 3, 15), None, date(2000, 1, 2)],
            "ts": [datetime(2024, 3, 15, 10, 20, 30), None, datetime(2000, 1, 2, 1, 2, 3)],
        }
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        fmt_date = "yyyy-MM-dd"
        fmt_ts = "yyyy-MM-dd HH:mm:ss"
        got = create_spark_df(
            spark,
            pl_df.select(
                PF.date_format("d", fmt_date).alias("ds"),
                PF.date_format("d", fmt_ts).alias("dts"),
                PF.date_format("ts", fmt_ts).alias("tss"),
            ),
        )
        exp = sdf.select(
            F.date_format(spark_col("d"), fmt_date).alias("ds"),
            F.date_format(spark_col("d"), fmt_ts).alias("dts"),
            F.date_format(spark_col("ts"), fmt_ts).alias("tss"),
        )
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    @pytest.mark.parametrize(
        "pattern",
        ["yyyy/MM/dd", "dd-MM-yyyy HH:mm:ss"],
    )
    def test_parametrized_patterns_match_spark(self, spark, pattern: str) -> None:
        from datetime import datetime

        data = {"ts": [datetime(2024, 12, 1, 8, 9, 10), None]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.date_format("ts", pattern).alias("s")))
        exp = sdf.select(F.date_format(spark_col("ts"), pattern).alias("s"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)


class TestLeastParity:
    """``least``: row-wise minimum with nulls skipped, aligned with PySpark 4."""

    def test_int_columns_nulls_match_spark(self, spark) -> None:
        data = {"a": [1, 10, None, 3], "b": [2, 5, None, None], "c": [3, 1, 1, None]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.least("a", "b", "c").alias("m")))
        exp = sdf.select(F.least(spark_col("a"), spark_col("b"), spark_col("c")).alias("m"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_with_lit_matches_spark(self, spark) -> None:
        data = {"a": [10, 3, None]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.least("a", PF.lit(5)).alias("m")))
        exp = sdf.select(F.least(spark_col("a"), F.lit(5)).alias("m"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_strings_match_spark(self, spark) -> None:
        data = {"x": ["b", "z", "m"], "y": ["a", None, "n"]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.least("x", "y").alias("m")))
        exp = sdf.select(F.least(spark_col("x"), spark_col("y")).alias("m"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_mixed_int_and_double_match_spark(self, spark) -> None:
        data = {"a": [1, 10], "b": [2.5, 2.0]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.least("a", "b").alias("m")))
        exp = sdf.select(F.least(spark_col("a"), spark_col("b")).alias("m"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)


class TestExtendedFunctionsParity:
    """Parity for ``floor``, ``pow``, ``isnan``, ``try_divide``, ``greatest``, ``create_map``, ``months_between``."""

    def test_floor_float_matches_spark(self, spark) -> None:
        data = {"x": [1.7, -2.3, None]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.floor("x").alias("f")))
        exp = sdf.select(F.floor(spark_col("x")).alias("f"))
        assert_pyspark_df_equal(
            got.withColumn("f", spark_col("f").cast("double")),
            exp.withColumn("f", spark_col("f").cast("double")),
            ignore_nullable=True,
        )

    def test_pow_matches_spark(self, spark) -> None:
        data = {"a": [2, 2, 3], "b": [3.0, None, 2.0]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.pow("a", "b").alias("p")))
        exp = sdf.select(F.pow(spark_col("a"), spark_col("b")).alias("p"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_isnan_float_matches_spark(self, spark) -> None:
        data = {"x": [1.0, float("nan"), None, 2.5]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.isnan("x").alias("n")))
        exp = sdf.select(F.isnan(spark_col("x")).alias("n"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_try_divide_matches_spark(self, spark) -> None:
        data = {"a": [10, 10, None, 10], "b": [2, 0, 2, None]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.try_divide("a", "b").alias("q")))
        exp = sdf.select(F.try_divide(spark_col("a"), spark_col("b")).alias("q"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_greatest_matches_spark(self, spark) -> None:
        data = {"a": [1, 10, None, 3], "b": [2, 5, None, None], "c": [3, 1, 1, None]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.greatest("a", "b", "c").alias("m")))
        exp = sdf.select(F.greatest(spark_col("a"), spark_col("b"), spark_col("c")).alias("m"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_create_map_lookup_matches_spark(self, spark) -> None:
        data = {"c1": ["v1", "a"], "c2": ["v2", "b"]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        m_pf = PF.create_map(PF.lit("k1"), "c1", PF.lit("k2"), "c2")
        got = create_spark_df(
            spark,
            pl_df.select(
                PF.try_element_at(m_pf, "k1").alias("v1"),
                PF.try_element_at(m_pf, "k2").alias("v2"),
            ),
        )
        m_sp = F.create_map(F.lit("k1"), spark_col("c1"), F.lit("k2"), spark_col("c2"))
        exp = sdf.select(m_sp.alias("m")).select(
            spark_col("m").getItem("k1").alias("v1"), spark_col("m").getItem("k2").alias("v2")
        )
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_months_between_dates_matches_spark(self, spark) -> None:
        from datetime import date

        data = {
            "e": [date(2024, 3, 15), date(2024, 1, 31), None, date(2024, 2, 28)],
            "s": [date(2024, 1, 15), date(2024, 1, 1), date(2024, 1, 1), date(2024, 3, 30)],
        }
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.months_between("e", "s").alias("mb")))
        exp = sdf.select(F.months_between(spark_col("e"), spark_col("s")).alias("mb"))
        assert_pyspark_df_equal(
            got.withColumn("mb", spark_col("mb").cast("double")),
            exp.withColumn("mb", spark_col("mb").cast("double")),
            ignore_nullable=True,
            allow_nan_equality=True,
            precision=6,
        )


class TestNullifAndTrim:
    def test_nullif_two_columns_against_spark(self, spark) -> None:
        data = {"a": [1, 2, 2, None, 1], "b": [1, 2, 3, 1, None]}
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(PF.nullif("a", "b").alias("n")),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            F.nullif(spark_col("a"), spark_col("b")).alias("n")
        )
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_nullif_with_lit_against_spark(self, spark) -> None:
        data = {"x": [0, 0, 1, None]}
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(PF.nullif("x", PF.lit(0)).alias("n")),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            F.nullif(spark_col("x"), F.lit(0)).alias("n")
        )
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_trim_against_spark(self, spark) -> None:
        data = {"s": ["  a  ", "b\t", None, " c \n"]}
        pl_df = DataFrame(pl.DataFrame(data))
        got = create_spark_df(
            spark,
            pl_df.select(PF.trim("s").alias("t")),
        )
        exp = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).select(
            F.trim(spark_col("s")).alias("t")
        )
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)


class TestGroupingFirst:
    def test_first_agg_against_spark(self, spark) -> None:
        data = {"g": [1, 1, 2, 2], "v": [10, 20, 30, 40]}
        pl_df = DataFrame(pl.DataFrame(data)).sort("g", "v")
        s_df = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys())).orderBy("g", "v")
        got = create_spark_df(
            spark,
            pl_df.groupBy("g").agg(PF.first("v").alias("fv")),
        )
        exp = s_df.groupBy("g").agg(F.first("v").alias("fv"))
        assert_pyspark_df_equal(
            got.orderBy("g"),
            exp.orderBy("g"),
            ignore_nullable=True,
        )


class TestBroadcast:
    def test_broadcast_returns_same_dataframe(self) -> None:
        d = DataFrame(pl.DataFrame({"x": [1]}))
        assert PF.broadcast(d) is d


class TestWithColumnLitOnEmptyFrame:
    """``F.lit`` + ``withColumn`` on 0-row frames: Spark broadcasts; Polars must accept the expr."""

    def test_with_column_string_lit_empty_rows_matches_spark(self, spark) -> None:
        pl_df = DataFrame(pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)}))
        tag = "general.raw_clutch_lending.co_applicants"
        tagged = pl_df.withColumn("data_origin", PF.lit(tag))
        exp = spark.createDataFrame([], _EMPTY_ID_SCHEMA).withColumn("data_origin", F.lit(tag))
        got = create_spark_df(spark, tagged)
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_with_column_null_lit_empty_rows_matches_spark(self, spark) -> None:
        pl_df = DataFrame(pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)}))
        tagged = pl_df.withColumn("n", PF.lit(None))
        # Spark ``lit(None)`` is void; sparkleframe ``lit(None)`` matches Spark string nulls.
        exp = spark.createDataFrame([], _EMPTY_ID_SCHEMA).withColumn("n", F.lit(None).cast("string"))
        got = create_spark_df(spark, tagged)
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)


class TestArrayParity:
    def test_array_columns_and_literals(self, spark) -> None:
        data = {"a": [1, None, 3], "b": [10, 20, 30]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(
            spark,
            pl_df.select(
                PF.array("a", "b").alias("arr"),
                PF.array(PF.lit(1), PF.lit(2)).alias("lit_arr"),
            ),
        )
        exp = sdf.select(
            F.array(spark_col("a"), spark_col("b")).alias("arr"),
            F.array(F.lit(1), F.lit(2)).alias("lit_arr"),
        )
        assert got.collect() == exp.collect()


class TestToJsonParity:
    def test_struct_matches_spark(self, spark) -> None:
        data = {"a": [1, 2], "b": ["x", "y"]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.to_json(PF.struct("a", "b")).alias("j")))
        exp = sdf.select(F.to_json(F.struct(spark_col("a"), spark_col("b"))).alias("j"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_struct_ignore_null_fields_false_matches_spark(self, spark) -> None:
        data = {"a": [1, 2], "b": [None, "y"]}
        pl_df = DataFrame(pl.DataFrame(data, schema={"a": pl.Int64, "b": pl.String}))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        options = {"ignoreNullFields": False}
        got = create_spark_df(spark, pl_df.select(PF.to_json(PF.struct("a", "b"), options=options).alias("j")))
        exp = sdf.select(F.to_json(F.struct(spark_col("a"), spark_col("b")), options=options).alias("j"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_create_map_matches_spark(self, spark) -> None:
        data = {"v": [100, 200]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        m_pf = PF.create_map(PF.lit("k"), "v")
        got = create_spark_df(spark, pl_df.select(PF.to_json(m_pf).alias("j")))
        m_sp = F.create_map(F.lit("k"), spark_col("v"))
        exp = sdf.select(F.to_json(m_sp).alias("j"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)

    def test_array_primitives_matches_spark(self, spark) -> None:
        data = {"arr": [[1, 2], None, []]}
        pl_df = DataFrame(pl.DataFrame(data))
        sdf = spark.createDataFrame(spark_rows_from_dict(data), list(data.keys()))
        got = create_spark_df(spark, pl_df.select(PF.to_json("arr").alias("j")))
        exp = sdf.select(F.to_json(spark_col("arr")).alias("j"))
        assert_pyspark_df_equal(got, exp, ignore_nullable=True)
