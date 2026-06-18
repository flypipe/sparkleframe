import builtins
import re
import uuid as std_uuid

import pandas as pd
import pandas.testing as pdt
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pyspark.sql.functions import asc as spark_asc
from pyspark.sql.functions import asc_nulls_first as spark_asc_nulls_first
from pyspark.sql.functions import asc_nulls_last as spark_asc_nulls_last
from pyspark.sql.functions import col as spark_col
from pyspark.sql.functions import current_timestamp as spark_current_timestamp
from pyspark.sql.functions import dense_rank as spark_dense_rank
from pyspark.sql.functions import desc as spark_desc
from pyspark.sql.functions import desc_nulls_first as spark_desc_nulls_first
from pyspark.sql.functions import desc_nulls_last as spark_desc_nulls_last
from pyspark.sql.functions import from_json as spark_from_json
from pyspark.sql.functions import lit as spark_lit
from pyspark.sql.functions import map_from_entries as spark_map_from_entries
from pyspark.sql.functions import now as spark_now
from pyspark.sql.functions import rand as spark_rand
from pyspark.sql.functions import rank as spark_rank
from pyspark.sql.functions import round as spark_round
from pyspark.sql.functions import row_number as spark_row_number
from pyspark.sql.functions import struct as spark_struct
from pyspark.sql.functions import unix_millis as spark_unix_millis
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
    array,
    asc,
    asc_nulls_first,
    asc_nulls_last,
    broadcast,
    col,
    concat,
    current_timestamp,
    dense_rank,
    desc,
    desc_nulls_first,
    desc_nulls_last,
    from_json,
    lit,
    map_from_entries,
    now,
    rand,
    rank,
    round,
    row_number,
    size,
    struct,
    to_json,
    try_element_at,
    uuid,
)
from sparkleframe.polarsdf.types import IntegerType, StringType, StructField, StructType
from sparkleframe.engine import Engine
from sparkleframe.tests.parity.engines import ENGINES
from sparkleframe.tests.parity.oracle import assert_matches_spark
from sparkleframe.tests.utils import create_spark_df

sample_data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}


@pytest.fixture
def sparkle_df():
    return DataFrame(pl.DataFrame(sample_data))


@pytest.fixture
def spark_df(spark):
    return spark.createDataFrame(pd.DataFrame(sample_data))


class TestFunctions:

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
        spark_df = create_spark_df(spark, sparkle_df)
        expected_df = spark_df.select(spark_lit(literal_value).alias("value")).toPandas()

        # Compare using pandas
        pdt.assert_frame_equal(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_dtype=False,  # Important: ignores schema/type mismatches
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
        spark_df = create_spark_df(spark, polars_df)
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
        sparkle_df = sparkle_df.withColumn("rank", col("rank").cast(IntegerType())).orderBy(
            "group", "category", "subcategory", "value"
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

        assert_matches_spark(sparkle_df, spark_df, ENGINES[Engine.POLARS])

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


class TestNowAndMonotonicallyIncreasingId:
    """Parity with PySpark for now and monotonically_increasing_id."""

    def test_now_all_rows_equal_and_close_to_spark(self, spark) -> None:
        data = {"x": [1, 2, 3]}
        polars_df = DataFrame(pl.DataFrame(data))
        result_spark_df = create_spark_df(spark, polars_df.select(now().alias("t")))
        expected_df = create_spark_df(spark, polars_df).select(spark_now().alias("t"))
        ms1 = [r[0] for r in result_spark_df.select(spark_unix_millis("t").alias("m")).collect()]
        ms2 = [r[0] for r in expected_df.select(spark_unix_millis("t").alias("m")).collect()]
        assert len(set(ms1)) == 1
        assert len(set(ms2)) == 1
        assert builtins.abs(ms1[0] - ms2[0]) < 3_000

    def test_current_timestamp_all_rows_equal_and_close_to_spark(self, spark) -> None:
        data = {"x": [1, 2, 3]}
        polars_df = DataFrame(pl.DataFrame(data))
        result_spark_df = create_spark_df(spark, polars_df.select(current_timestamp().alias("t")))
        expected_df = create_spark_df(spark, polars_df).select(spark_current_timestamp().alias("t"))
        ms1 = [r[0] for r in result_spark_df.select(spark_unix_millis("t").alias("m")).collect()]
        ms2 = [r[0] for r in expected_df.select(spark_unix_millis("t").alias("m")).collect()]
        assert len(set(ms1)) == 1
        assert len(set(ms2)) == 1
        assert builtins.abs(ms1[0] - ms2[0]) < 3_000


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


class TestElementAt:
    """``element_at`` is ANSI-strict on invalid array indices (Spark 4); not an alias of ``try_element_at``."""

    @staticmethod
    def _spark_df_string_array(spark, values: list):
        return spark.createDataFrame(
            [(values,)],
            schema=SparkStructType([SparkStructField("arr", SparkArrayType(SparkStringType()), True)]),
        )


class TestMapFunctions:

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
        assert_matches_spark(result_df, expected_df, ENGINES[Engine.POLARS])


class TestFromJson:

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
        spark_valid = create_spark_df(spark, polars_valid)
        assert_matches_spark(
            polars_valid.select(from_json("j", schema).alias("parsed")),
            spark_valid.select(spark_from_json("j", spark_schema).alias("parsed")),
            ENGINES[Engine.POLARS],
        )

        bad_data = {"j": ["not-json"]}
        polars_bad = DataFrame(pl.DataFrame(bad_data))
        result = polars_bad.select(from_json("j", schema).alias("parsed")).to_native_df()
        assert result["parsed"][0] is None

        spark_bad = create_spark_df(spark, polars_bad)
        spark_row = spark_bad.select(spark_from_json("j", spark_schema).alias("parsed")).collect()[0][0]
        assert spark_row is None or (spark_row.field1 is None and spark_row.field2 is None)


class TestBroadcast:
    def test_broadcast_returns_same_dataframe(self) -> None:
        d = DataFrame(pl.DataFrame({"x": [1]}))
        assert broadcast(d) is d


class TestRandParity:
    def test_rand_output_shape_and_type_matches_spark(self, spark) -> None:
        data = {"x": [1, 2, 3, 4, 5]}
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = create_spark_df(spark, polars_df)
        sf_result = polars_df.select(rand(42).alias("r"))
        sp_result = spark_df.select(spark_rand(42).alias("r"))
        sf_vals = sf_result.to_native_df()["r"].to_list()
        sp_vals = [row[0] for row in sp_result.collect()]
        assert len(sf_vals) == len(sp_vals)
        assert all(isinstance(v, float) and 0.0 <= v < 1.0 for v in sf_vals)
        assert all(isinstance(v, float) and 0.0 <= v < 1.0 for v in sp_vals)


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
