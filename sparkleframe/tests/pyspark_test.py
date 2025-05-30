# sparkleframe/tests/pyspark_test.py
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.functions import round as spark_round

# utils.py or inside your test helpers
from pyspark.sql.types import FloatType, DoubleType


def round_numeric_columns(df, precision=5):
    for field in df.schema.fields:
        if isinstance(field.dataType, (FloatType, DoubleType)):
            df = df.withColumn(field.name, spark_round(field.name, precision))
    return df


def assert_pyspark_df_equal(df1, df2, precision=None, **kwargs):
    """
    Compares two PySpark DataFrames for equality.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        precision: Optional rounding precision for float columns.
        **kwargs: Extra arguments to pass to chispa.assert_df_equality.
    """
    if precision is not None:
        df1 = round_numeric_columns(df1, precision)
        df2 = round_numeric_columns(df2, precision)

    assert_df_equality(df1, df2, **kwargs)
