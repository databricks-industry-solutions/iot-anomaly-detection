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

startingOffsets = "earliest"

silver_df = spark.readStream \
  .format("delta") \
  .option("startingOffsets", startingOffsets) \
  .table(f"{database}.{source_table}") \
  .withColumn("datetime", F.from_unixtime(F.col("timestamp"))).writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("mergeSchema", "true") \
  .option("checkpointLocation", checkpoint_location_target) \
  .trigger(once = True) \
  .table(f"{database}.{target_table}")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select count(1) from rvp_iot_sa.silver

# COMMAND ----------


