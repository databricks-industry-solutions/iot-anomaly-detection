# Databricks notebook source
# DBTITLE 1,Reading data from Silver Layer and Writing into Gold
dbutils.widgets.text("source_table", "silver")
dbutils.widgets.text("target_table", "gold")
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

gold_df = spark.readStream \
  .format("delta") \
  .option("startingOffsets", startingOffsets) \
  .table(f"{database}.{source_table}").writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", checkpoint_location_target) \
  .option("mergeSchema", "true") \
  .trigger(once = True) \
  .table(f"{database}.{target_table}")


# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select count(1) from rvp_iot_sa.gold

# COMMAND ----------


