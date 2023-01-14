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
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

json_schema = StructType([
  StructField("r_0", FloatType(), True),
  StructField("r_1", FloatType(), True),
  StructField("r_2", FloatType(), True),
  StructField("r_3", FloatType(), True),
  StructField("r_4", FloatType(), True),
  StructField("r_5", FloatType(), True),
  StructField("code1", IntegerType(), True),
  StructField("code2", IntegerType(), True),
  StructField("code3", IntegerType(), True)
])

startingOffsets = "earliest"

bronze_df = spark.readStream \
  .format("delta") \
  .option("startingOffsets", startingOffsets) \
  .table(f"{database}.{source_table}") \
  .withColumn(
    "parsedJson", from_json(col("parsedValue"), json_schema)
  ).select("parsedJson.*", "timestamp")

display(bronze_df)

# COMMAND ----------


bronze_df.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", checkpoint_location_target) \
  .option("mergeSchema", "true") \
  .trigger(processingTime="30 seconds") \
  .table(f"{database}.{target_table}")

# COMMAND ----------

display(spark.sql(f"select count(*) from {database}.{target_table}"))

# COMMAND ----------


