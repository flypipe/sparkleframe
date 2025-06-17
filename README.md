![SparkleFrame](docs/images/logo_white.png#only-dark)

SparkleFrame implements the PySpark DataFrame API in order to enable running transformation pipelines
directly on [Polars Dataframe](https://docs.pola.rs/api/python/stable/reference/index.html) - no Spark clusters or 
dependencies required.

Apache Spark is designed for distributed, large-scale data processing, but it is not optimized for low-latency use 
cases. There are scenarios, however, where you need to quickly re-compute certain data—for example, 
regenerating features for a machine learning model in real time or near-real time.

SparkleFrame is great for:

* Users who want to run PySpark code quickly locally without the overhead of starting a Spark session
* Users who want to run PySpark DataFrame code without the complexity of using Spark for processing
* Useful for unit testing, feature prototyping, or serving small pipelines in microservices.

You can learn **more about the design motivation behind Sparkleframe** in this 
    [discussion thread](https://github.com/eakmanrq/sqlframe/issues/409).

## Documentation

Full documentation is available at [https://flypipe.github.io/sparkleframe/](https://flypipe.github.io/sparkleframe/).

## Installation

```bash
pip install sparkleframe
```

## Usage

SparkleFrame can be used in two ways:

* Directly importing the `sparkleframe.polarsdf` package 
* Using the `activate` function to allow for continuing to use `pyspark.sql` but have it use SparkleFrame behind the scenes.

### Directly importing

If converting a PySpark pipeline, all `pyspark.sql` should be replaced with `sparkleframe.polarsdf`.

```python
# PySpark import
# from pyspark.sql import SparkSession
# from pyspark.sql import functions as F
# from pyspark.sql.dataframe import DataFrame
# SparkleFrame import
from sparkleframe.polarsdf.session import SparkSession
from sparkleframe.polarsdf import functions as F
from sparkleframe.polarsdf.dataframe import DataFrame
```

## Activating SparkleFrame

SparkleFrame can either replace pyspark imports or be used alongside them. To replace pyspark imports, 
use the activate function to set the engine to use.

```python
from sparkleframe.activate import activate

# Activate SparkleFrame
activate()

from pyspark.sql import SparkSession
session = SparkSession.builder.getOrCreate()
```

`SparkSession` will now be a SparkleFrame Session object and everything will be run on Polars Dataframe directly.

SparkleFrame can also be directly imported which both maintains pyspark imports:

```python
from sparkleframe.polarsdf.session import SparkSession
session = SparkSession.builder.getOrCreate()
```

## Example Usage

```python
from sparkleframe.activate import activate

# Activate SparkleFrame
activate()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

session = SparkSession.builder.getOrCreate()
df = session.createDataFrame(data=[{"col1": 1, "col2": 2}])
df = df.withColumn("col3", F.col("col2") + F.col("col1"))
```
```python
>>> print(type(df))
<class 'sparkleframe.polarsdf.dataframe.DataFrame'>
```
```python
>>> df.show()
shape: (1, 3)
┌──────┬──────┬──────┐
│ col1 ┆ col2 ┆ col3 │
│ ---  ┆ ---  ┆ ---  │
│ i64  ┆ i64  ┆ i64  │
╞══════╪══════╪══════╡
│ 1    ┆ 2    ┆ 3    │
└──────┴──────┴──────┘
```

!!! note

    If you encounter any transformation that is not implemented, please open an [issue on GitHub](https://github.com/flypipe/sparkleframe/issues/new) so it can be prioritized.


## Source Code

API code is available at [https://github.com/flypipe/sparkleframe](https://github.com/flypipe/sparkleframe).