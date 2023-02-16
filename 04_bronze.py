# Databricks notebook source
# DBTITLE 1,Reading data from Raw Layer and writing into Bronze
dbutils.widgets.text("source_table", "raw")
dbutils.widgets.text("target_table", "bronze")
dbutils.widgets.text("database", "rvp_iot_sa")
checkpoint_path = "/dbfs/tmp/checkpoints"

source_table = getArgument("source_table")
target_table = getArgument("target_table")
database = getArgument("database")
checkpoint_location_target = f"{checkpoint_path}/{target_table}"

# Remove this if not testing
dbutils.fs.rm(checkpoint_location_target, recurse = True)
spark.sql(f"drop table if exists {database}.{target_table}")

# COMMAND ----------

from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType

json_schema = StructType([
  StructField("timestamp", IntegerType(), True),
  StructField("device_id", IntegerType(), True),
  StructField("device_model", StringType(), True),
  StructField("sensor_1", FloatType(), True),
  StructField("sensor_2", FloatType(), True),
  StructField("sensor_3", FloatType(), True),
  StructField("state", StringType(), True)
])

startingOffsets = "earliest"

bronze_df = spark.readStream \
  .format("delta") \
  .option("startingOffsets", startingOffsets) \
  .table(f"{database}.{source_table}") \
  .withColumn(
    "parsedJson", from_json(col("parsedValue"), json_schema)
  ) \
  .select("parsedJson.*") \
  .dropna().writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", checkpoint_location_target) \
  .option("mergeSchema", "true") \
  .trigger(once = True) \
  .table(f"{database}.{target_table}")

# COMMAND ----------

# DBTITLE 1,Analyzing our Data
static_df = spark.sql(f"select * from {database}.{target_table}")
display(static_df)

# COMMAND ----------

from pyspark.sql import functions as F

train_df = static_df.filter(F.from_unixtime("timestamp") < "2021-12-01")
test_df = static_df.filter(
  (F.from_unixtime("timestamp") >= "2021-12-01")
  & (F.from_unixtime("timestamp") < "2022-01-01")
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

# COMMAND ----------

train_df.write.saveAsTable(f"{database}.train", mode = "overwrite")
test_df.write.saveAsTable(f"{database}.test", mode = "overwrite")
