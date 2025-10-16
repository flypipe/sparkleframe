"""
this is my <file_name.py>

<content of the file>

implement the pyspark <function> function like as defined in <function_url> into my <file_name.py>

"""

import os
import pathlib


def generate_pyspark_function_instruction(file_path: str, function_name: str, function_url: str) -> str:
    """
    Generates a formatted message including the contents of a Python file and an instruction
    to implement a specific PySpark function based on its official documentation.

    Args:
        file_path (str): Path to the Python file.
        function_name (str): Name of the PySpark function to implement.
        function_url (str): URL to the official PySpark documentation for the function.

    Returns:
        str: Formatted message.
    """
    file_path = str(pathlib.Path(__file__).parent.parent.resolve()) + file_path
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        file_contents = file.read()

    file_name = os.path.basename(file_path)

    message = f"""\
this is the content of my {file_name}:

{file_contents}

implement the pyspark {function_name} function like as defined in {function_url} into my {file_name}
"""
    return message


def generate_test_instruction(file_path: str, function_name: str, compare_output_spark=False) -> str:
    """
    Generates a formatted message including the contents of a Python file and an instruction
    to implement a specific PySpark function based on its official documentation.

    Args:
        file_path (str): Path to the Python file.
        function_name (str): Name of the PySpark function to implement.
        function_url (str): URL to the official PySpark documentation for the function.

    Returns:
        str: Formatted message.
    """
    file_path = str(pathlib.Path(__file__).parent.parent.resolve()) + file_path
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        file_contents = file.read()

    file_name = os.path.basename(file_path)

    spark_msg_output = (
        ", create a spark dataframe build from PolarDataframe and use another spark dataframe to compare them "
        if compare_output_spark
        else ""
    )

    message = f"""\
this is the content of my test file {file_name}:

{file_contents}

generate a pytest test case, using pytest.parametrize to test the logic `{function_name}` that has been just implemented{spark_msg_output}
"""
    return message


file_path = "/sparkleframe/polarsdf/dataframe.py"
function_name = "count"
pyspark_function_url = (
    "https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.count.html"
)

msg = generate_pyspark_function_instruction(file_path, function_name, pyspark_function_url)

# msg = generate_test_instruction(file_path, function_name, compare_output_spark=True)

print(msg)
