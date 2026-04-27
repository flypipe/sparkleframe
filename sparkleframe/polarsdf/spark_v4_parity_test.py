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
