from sparkleframe.engine import Engine


class TestEngine:
    def test_polars_module_and_prefix(self):
        assert Engine.POLARS.module == "polarsdf"
        assert Engine.POLARS.class_prefix == "Polars"

    def test_python_module_and_prefix(self):
        assert Engine.PYTHON.module == "python"
        assert Engine.PYTHON.class_prefix == "Python"

    def test_clean_class_name_strips_matching_prefix(self):
        assert Engine.POLARS.clean_class_name("PolarsColumn") == "Column"
        assert Engine.POLARS.clean_class_name("PolarsDataFrame") == "DataFrame"
        assert Engine.PYTHON.clean_class_name("PythonColumn") == "Column"

    def test_clean_class_name_leaves_non_prefixed_unchanged(self):
        assert Engine.POLARS.clean_class_name("Column") == "Column"
        assert Engine.POLARS.clean_class_name("functions") == "functions"
        # A name carrying the other engine's prefix is not stripped.
        assert Engine.PYTHON.clean_class_name("PolarsColumn") == "PolarsColumn"

    def test_engine_is_re_exported_from_package_root(self):
        from sparkleframe import Engine as PackageEngine

        assert PackageEngine is Engine
