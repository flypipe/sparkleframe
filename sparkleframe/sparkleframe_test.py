from sparkleframe import activate
activate()

import pyspark.sql.functions as F
print(type(F), type(F.col("a")))

