import os
import sys
from pathlib import Path

import pytest

# Skip writing pyc files on a readonly filesystem.
sys.dont_write_bytecode = True

# Local-dev defaults so direct `pytest` invocations work without a wrapper.
# Docker/CI flows set these explicitly via .env and are unaffected.
os.environ.setdefault("TIMEZONE", "UTC")
if "JAVA_HOME" not in os.environ:
    for candidate in (
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
        "/usr/lib/jvm/java-17-openjdk-amd64",
    ):
        if Path(candidate).exists():
            os.environ["JAVA_HOME"] = candidate
            os.environ["PATH"] = f"{candidate}/bin:" + os.environ.get("PATH", "")
            break


@pytest.fixture(scope="function", autouse=False)
def spark():
    from sparkleframe.tests.spark import spark as spark_session

    return spark_session


@pytest.fixture(scope="function", autouse=False)
def sparkle():
    from sparkleframe.tests.sparkle import sparkle as sparkle_session

    return sparkle_session
