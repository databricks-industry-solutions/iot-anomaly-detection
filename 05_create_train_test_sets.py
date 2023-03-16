# Databricks notebook source
# DBTITLE 1,Reading data from Silver Layer and Writing into Gold
dbutils.widgets.text("source_table", "silver")
dbutils.widgets.text("target_table_train", "train")
dbutils.widgets.text("target_table_test", "test")
dbutils.widgets.text("database", "rvp_iot_sa")
checkpoint_path = "/dbfs/tmp/checkpoints"

source_table = getArgument("source_table")
target_table_train = getArgument("target_table_train")
target_table_test = getArgument("target_table_test")
database = getArgument("database")
checkpoint_location_target = f"{checkpoint_path}/dataset"

# Remove this if not testing
dbutils.fs.rm(checkpoint_location_target, recurse = True)
spark.sql(f"drop table if exists {database}.{target_table_train}")
spark.sql(f"drop table if exists {database}.{target_table_test}")

# COMMAND ----------

from pyspark.sql import functions as F

static_df = spark.sql(f"SELECT * FROM {database}.{source_table}")
train_df = static_df.filter(F.col("datetime") < "2021-09-01")
test_df = static_df.filter(
  (F.col("datetime") >= "2021-09-01")
  & (F.col("datetime") < "2022-01-01")
)

print(f"Training set contains {train_df.count()} rows")
print(f"Testing set contains {test_df.count()} rows")

# COMMAND ----------

# DBTITLE 1,Labelling our Data
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import functions as F

@pandas_udf("int")
def label_data(sensor: pd.Series) -> pd.Series:

  anomaly = (sensor < 20) | (sensor > 80)
  anomaly = anomaly.astype(int)
  return anomaly

train_df = train_df.withColumn("anomaly", label_data("sensor_1"))
test_df = test_df.withColumn("anomaly", label_data("sensor_1"))

train_df.write.saveAsTable(f"{database}.train", mode = "overwrite")
test_df.write.saveAsTable(f"{database}.test", mode = "overwrite")

# COMMAND ----------


