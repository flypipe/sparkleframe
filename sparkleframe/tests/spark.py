# noinspection PyUnresolvedReferences
import os

from pyspark.sql import SparkSession

# Avoid WARNING:root:'PYARROW_IGNORE_TIMEZONE' environment variable was not set
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


def get_spark():
    configs = (
        SparkSession.builder.appName("sparkleframe")
        .master("local[1]")
        .config("spark.sql.session.timeZone", os.environ["TIMEZONE"])
    )

    spark = configs.getOrCreate()
    return spark


spark = get_spark()
