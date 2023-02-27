# Databricks notebook source
# DBTITLE 1,Reading Data from Bronze Layer and writing into Silver
dbutils.widgets.text("source_table", "bronze")
dbutils.widgets.text("target_table", "silver")
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

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType

startingOffsets = "earliest"

silver_df = spark.readStream \
  .format("delta") \
  .option("startingOffsets", startingOffsets) \
  .table(f"{database}.{source_table}")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Parsing data from our payload into columns
json_schema = StructType([
  StructField("timestamp", IntegerType(), True),
  StructField("device_id", IntegerType(), True),
  StructField("device_model", StringType(), True),
  StructField("sensor_1", FloatType(), True),
  StructField("sensor_2", FloatType(), True),
  StructField("sensor_3", FloatType(), True),
  StructField("state", StringType(), True)
])

silver_df = silver_df \
  .withColumn(
    "struct_payload",
    F.from_json("parsedValue", schema = json_schema)
  ) \
  .select("struct_payload.*") \
  .withColumn("datetime", F.from_unixtime("timestamp"))

# COMMAND ----------

# DBTITLE 1,Writing into Silver
silver_df \
  .writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("mergeSchema", "true") \
  .option("checkpointLocation", checkpoint_location_target) \
  .trigger(once = True) \
  .table(f"{database}.{target_table}")

# COMMAND ----------


