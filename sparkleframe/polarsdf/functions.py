from typing import Union

from sparkleframe.polarsdf.column import Column, _to_expr
import polars as pl

def col(name: str) -> Column:
    """
    Mimics pyspark.sql.functions.col by returning a Column object.

    Args:
        name (str): Name of the column.

    Returns:
        Column: A Column object for building expressions.
    """
    return Column(name)

# Entry point function that mimics pyspark.sql.functions.when
def when(condition: Column, value) -> Column:
    """
    Starts a conditional column expression.

    Returns a Column containing a partial when().then() expression,
    which must be completed with .otherwise().
    """
    expr = pl.when(condition.to_native()).then(_to_expr(value))
    return Column(expr)

from sparkleframe.polarsdf.column import Column, _to_expr
import polars as pl


def col(name: str) -> Column:
    return Column(name)


def when(condition: Column, value) -> Column:
    expr = pl.when(condition.to_native()).then(_to_expr(value))
    return Column(expr)


def get_json_object(col: Union[str, Column], path: str) -> Column:
    """
    Mimics pyspark.sql.functions.get_json_object by extracting a JSON field.

    Args:
        col (str | Column): The column containing the JSON string.
        path (str): The JSON path in the format '$.field.subfield'.

    Returns:
        Column: A column representing the extracted JSON value.
    """
    if not isinstance(path, str) or not path.startswith("$."):
        raise ValueError("Path must be a string starting with '$.'")

    col_expr = col.to_native() if isinstance(col, Column) else pl.col(col)

    return Column(col_expr.str.json_path_match(path))
