import sys

import pytest

# Skip writing pyc files on a readonly filesystem.
sys.dont_write_bytecode = True


@pytest.fixture(scope="function", autouse=False)
def spark():
    from sparkleframe.tests.spark import spark as spark_session

    return spark_session


@pytest.fixture(scope="function", autouse=False)
def sparkle():
    from sparkleframe.tests.sparkle import sparkle as sparkle_session

    return sparkle_session
