
import inspect
import os
import pathlib
import urllib

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

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

if __name__ == "__main__":
    from pyspark.sql.catalog import Catalog as PYSPARK_CATALOG
    SPARKLE_CATALOG = None

    from pyspark.sql.column import Column as PYSPARK_COLUMN
    from sparkleframe.polarsdf.column import Column as SPARKLE_COLUMN

    from pyspark.sql.dataframe import DataFrame as PYSPARK_DATAFRAME
    from sparkleframe.polarsdf.dataframe import DataFrame as SPARKLE_DATAFRAME


    import pyspark.sql.functions as PYSPARK_FUNCTIONS
    import sparkleframe.polarsdf.functions as SPARKLE_FUNCTIONS

    from pyspark.sql.group import GroupedData as PYSPARK_GROUPED_DATA
    from sparkleframe.polarsdf.group import GroupedData as SPARKLE_GROUPED_DATA

    from pyspark.sql.window import Window as PYSPARK_WINDOW
    from sparkleframe.polarsdf.window import Window as SPARKLE_WINDOW

    from pyspark.sql.session import SparkSession as PYSPARK_SESSION
    from sparkleframe.polarsdf.session import SparkSession as SPARKLE_SESSION

    import pyspark.sql.types as PYSPARK_TYPES
    import sparkleframe.polarsdf.types as SPARKLE_TYPES

    MODULES = [

        {"url_key": "Column", "module_url_key": "column", "pyspark": PYSPARK_COLUMN, "sparkleframe": SPARKLE_COLUMN, "lambda": get_functions},
        {"url_key": "DataFrame", "module_url_key": "dataframe", "pyspark": PYSPARK_DATAFRAME, "sparkleframe": SPARKLE_DATAFRAME, "lambda": get_functions},
        {"url_key": "functions", "module_url_key": "functions", "pyspark": PYSPARK_FUNCTIONS, "sparkleframe": SPARKLE_FUNCTIONS, "lambda": get_functions},
        {"url_key": "GroupedData", "module_url_key": "grouping", "pyspark": PYSPARK_GROUPED_DATA, "sparkleframe": SPARKLE_GROUPED_DATA, "lambda": get_functions},
        {"url_key": "SparkSession", "module_url_key": "spark_session", "pyspark": PYSPARK_SESSION, "sparkleframe": SPARKLE_SESSION, "lambda": get_functions},
        {"url_key": "types", "module_url_key": "data_types", "pyspark": PYSPARK_TYPES, "sparkleframe": SPARKLE_TYPES, "lambda": get_data_types},
        {"url_key": "Window", "module_url_key": "window", "pyspark": PYSPARK_WINDOW, "sparkleframe": SPARKLE_WINDOW, "lambda": get_functions},
    ]

    pyspark_functions = []
    # 3️⃣ Number of threads: CPU cores minus 1, minimum 1
    num_threads = max(os.cpu_count() - 1, 1)
    for modules in sorted(MODULES, key=lambda module: module["module_url_key"]):

        pyspark_functions_list = modules['lambda'](modules["pyspark"])
        sparkleframe_functions = modules['lambda'](modules["sparkleframe"])
        pyspark_functions.append(f'\n\n## <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/{modules["module_url_key"]}.html" target="_blank">pyspark.sql.{modules["url_key"]}</a>')
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(check_doc_url, pyspark_func_name, modules['url_key']) for pyspark_func_name in sorted(pyspark_functions_list)]

            for future in as_completed(futures):
                url, exists, pyspark_func_name = future.result()

                if not exists:
                    # not found in pyspark documentation
                    continue

                line = f'<a href="{url}" target="_blank">{pyspark_func_name}</a>'
                if pyspark_func_name in sparkleframe_functions:
                    line = f'* ✅ ' + line
                else:
                    params = {
                        "title": f"[PYSPARK_API_REQUEST] add support to pyspark.sql.{modules['url_key']}.{pyspark_func_name}",
                        "body": f"Please implement feature [pyspark.sql.{modules['url_key']}.{pyspark_func_name}]({url})"
                    }

                    line = f'* ❌ ' + line + f" (<a href='https://github.com/flypipe/sparkleframe/issues/new?{urllib.parse.urlencode(params)}' target='_blank'>request feature :simple-github:</a>)"


                print(line)
                pyspark_functions.append(line)



    path = os.path.join(pathlib.Path(__file__).resolve().parent, "supported_api_doc.md")
    with open(path, "w") as file:
        file.write("\n".join(pyspark_functions))
