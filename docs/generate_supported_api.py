
import importlib
import inspect
import os
import pathlib
import urllib

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Backends to report coverage for, in display order. Each entry is the tab label
# shown in the docs and the `sparkleframe.<module>` package that implements it.
BACKENDS = [
    {"label": "Polars", "module": "polarsdf"},
    {"label": "Python", "module": "python"},
]


def get_data_types(T):
    # Get all data type classes from the module
    return sorted([
        getattr(T, attr).__name__ for attr in dir(T)
        if attr.endswith('Type') and isinstance(getattr(T, attr), type)
    ])


def get_functions(obj):
    """Extract all public functions from pyspark.sql.functions"""
    all_names = dir(obj)
    return sorted([
        name for name in all_names
        if not name.startswith("_") and inspect.isfunction(getattr(obj, name))
    ])

def check_doc_url(func_name, url_key):
    """Check if the function's API URL exists"""
    url = f"https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.{url_key}.{func_name}.html"
    try:
        response = requests.head(url, timeout=5)
        exists = response.status_code == 200
    except requests.RequestException as e:
        print(f"Error checking {url}: {e}")
        exists = False
    return url, exists, func_name


def load_backend_implementations(module_name):
    """Return the SparkleFrame implementation object for each compared module.

    Maps the comparison key (Column, DataFrame, functions, ...) to the object in
    `sparkleframe.<module_name>` whose public members we reflect over. If the backend
    package is not available yet (e.g. the Python backend before it is implemented),
    every entry is None so that the coverage report renders the whole API as missing.
    """
    try:
        column = importlib.import_module(f"sparkleframe.{module_name}.column").Column
        dataframe = importlib.import_module(f"sparkleframe.{module_name}.dataframe").DataFrame
        functions = importlib.import_module(f"sparkleframe.{module_name}.functions")
        grouped_data = importlib.import_module(f"sparkleframe.{module_name}.group").GroupedData
        session = importlib.import_module(f"sparkleframe.{module_name}.session").SparkSession
        types = importlib.import_module(f"sparkleframe.{module_name}.types")
        window = importlib.import_module(f"sparkleframe.{module_name}.window").Window
        return {
            "Column": column,
            "DataFrame": dataframe,
            "functions": functions,
            "GroupedData": grouped_data,
            "SparkSession": session,
            "types": types,
            "Window": window,
        }
    except ModuleNotFoundError:
        return {}


def build_pyspark_reference(modules, num_threads):
    """Enumerate the PySpark API once, keeping only entries that resolve to a real doc URL.

    Returns an ordered list of (module_meta, [(name, url), ...]) so the (network-heavy)
    URL existence check is shared across every backend tab instead of repeated per tab.
    """
    reference = []
    for module in sorted(modules, key=lambda m: m["module_url_key"]):
        pyspark_names = module["lambda"](module["pyspark"])
        members = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(check_doc_url, name, module["url_key"]) for name in sorted(pyspark_names)]
            for future in as_completed(futures):
                url, exists, name = future.result()
                if exists:
                    members.append((name, url))
        members.sort(key=lambda item: item[0])
        reference.append((module, members))
    return reference


def render_backend_coverage(reference, implementations, reflectors):
    """Render the ✅/❌ coverage bullet list for a single backend, indented for a content tab."""
    lines = []
    for module, members in reference:
        url_key = module["url_key"]
        module_url_key = module["module_url_key"]
        impl_obj = implementations.get(url_key)
        implemented = set(reflectors[url_key](impl_obj)) if impl_obj is not None else set()

        lines.append(
            f'\n\n    ## <a href="https://spark.apache.org/docs/latest/api/python/reference/'
            f'pyspark.sql/{module_url_key}.html" target="_blank">pyspark.sql.{url_key}</a>'
        )
        for name, url in members:
            link = f'<a href="{url}" target="_blank">{name}</a>'
            if name in implemented:
                lines.append(f'    * ✅ {link}')
            else:
                params = {
                    "title": f"[PYSPARK_API_REQUEST] add support to pyspark.sql.{url_key}.{name}",
                    "body": f"Please implement feature [pyspark.sql.{url_key}.{name}]({url})",
                }
                request_link = (
                    f"(<a href='https://github.com/flypipe/sparkleframe/issues/new?"
                    f"{urllib.parse.urlencode(params)}' target='_blank'>request feature :simple-github:</a>)"
                )
                lines.append(f'    * ❌ {link} {request_link}')
    return lines


if __name__ == "__main__":
    from pyspark.sql.catalog import Catalog as PYSPARK_CATALOG  # noqa: F401

    from pyspark.sql.column import Column as PYSPARK_COLUMN
    from pyspark.sql.dataframe import DataFrame as PYSPARK_DATAFRAME
    import pyspark.sql.functions as PYSPARK_FUNCTIONS
    from pyspark.sql.group import GroupedData as PYSPARK_GROUPED_DATA
    from pyspark.sql.window import Window as PYSPARK_WINDOW
    from pyspark.sql.session import SparkSession as PYSPARK_SESSION
    import pyspark.sql.types as PYSPARK_TYPES

    MODULES = [
        {"url_key": "Column", "module_url_key": "column", "pyspark": PYSPARK_COLUMN, "lambda": get_functions},
        {"url_key": "DataFrame", "module_url_key": "dataframe", "pyspark": PYSPARK_DATAFRAME, "lambda": get_functions},
        {"url_key": "functions", "module_url_key": "functions", "pyspark": PYSPARK_FUNCTIONS, "lambda": get_functions},
        {"url_key": "GroupedData", "module_url_key": "grouping", "pyspark": PYSPARK_GROUPED_DATA, "lambda": get_functions},
        {"url_key": "SparkSession", "module_url_key": "spark_session", "pyspark": PYSPARK_SESSION, "lambda": get_functions},
        {"url_key": "types", "module_url_key": "data_types", "pyspark": PYSPARK_TYPES, "lambda": get_data_types},
        {"url_key": "Window", "module_url_key": "window", "pyspark": PYSPARK_WINDOW, "lambda": get_functions},
    ]

    # Reflector used to list the implemented members of each backend module, keyed by url_key.
    REFLECTORS = {module["url_key"]: module["lambda"] for module in MODULES}

    # Number of threads: CPU cores minus 1, minimum 1
    num_threads = max(os.cpu_count() - 1, 1)

    # Enumerate the PySpark API (and its doc URLs) once, then reuse it for every backend tab.
    reference = build_pyspark_reference(MODULES, num_threads)

    output = []
    for backend in BACKENDS:
        implementations = load_backend_implementations(backend["module"])
        output.append(f'=== "{backend["label"]}"')
        output.extend(render_backend_coverage(reference, implementations, REFLECTORS))
        output.append("")

    path = os.path.join(pathlib.Path(__file__).resolve().parent, "supported_api_doc.md")
    with open(path, "w") as file:
        file.write("\n".join(output))
