import sys

import pytest

from sparkleframe.activate import activate, activate_context, deactivate
from sparkleframe.engine import Engine


def _pyspark_module_keys():
    return [k for k in sys.modules if k == "pyspark" or k.startswith("pyspark.")]


@pytest.fixture(autouse=True)
def restore_pyspark():
    """activate() rebinds the global ``pyspark`` modules in ``sys.modules``.

    Snapshot the real modules before each test and restore those exact objects
    afterwards. Restoring the original module objects (rather than re-importing, as
    ``deactivate`` does) preserves pyspark's class identities, so the shared real-Spark
    fixtures used by the rest of the suite keep working.
    """
    saved = {k: sys.modules[k] for k in _pyspark_module_keys()}
    try:
        yield
    finally:
        for k in _pyspark_module_keys():
            del sys.modules[k]
        sys.modules.update(saved)


class TestActivate:
    def test_activate_default_binds_polars_engine(self):
        activate()

        import pyspark.sql as pysql
        import pyspark.sql.functions as F

        assert pysql.__name__ == "sparkleframe.polarsdf"
        assert F.__name__ == "sparkleframe.polarsdf.functions"
        # Public PySpark classes are exposed under the rebound namespace.
        assert hasattr(pysql, "DataFrame")
        assert hasattr(pysql, "Column")
        assert hasattr(pysql, "SparkSession")

    def test_activate_explicit_polars_engine(self):
        activate(Engine.POLARS)

        import pyspark.sql as pysql

        assert pysql.__name__ == "sparkleframe.polarsdf"

    def test_activate_python_engine_raises_not_implemented(self):
        # The python engine package does not exist yet, so selecting it must fail clearly.
        with pytest.raises(NotImplementedError):
            activate(Engine.PYTHON)

    def test_activate_context_binds_then_restores(self):
        with activate_context():
            import pyspark.sql as pysql

            assert pysql.__name__ == "sparkleframe.polarsdf"

    def test_deactivate_restores_real_pyspark(self):
        activate()
        import pyspark.sql as mocked

        assert mocked.__name__ == "sparkleframe.polarsdf"

        deactivate()
        import pyspark.sql as restored

        # The real pyspark (installed in the test environment) is importable again.
        assert restored.__name__ == "pyspark.sql"
