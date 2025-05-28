from sparkleframe import activate
activate()
from sparkleframe.polarsdf import DataFrame
from sparkleframe.polarsdf import functions as F
from sparkleframe.polarsdf import SparkSession

import pyspark.sql.functions as F
print(type(F), type(F.col("a")))

