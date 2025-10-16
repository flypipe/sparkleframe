# noinspection PyUnresolvedReferences
import os

from sparkleframe.polarsdf.session import SparkSession

# Avoid WARNING:root:'PYARROW_IGNORE_TIMEZONE' environment variable was not set
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


def get_sparkle():
    return SparkSession()


sparkle = get_sparkle()
