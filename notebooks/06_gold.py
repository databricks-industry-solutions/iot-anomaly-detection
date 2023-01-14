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
  .table(f"{database}.{source_table}")

display(gold_df)

# COMMAND ----------

# TODO: Calculate streaming queries / aggregations and write into gold, e.g.:
from pyspark.sql import functions as F

def generate_aggregations(
  input_df,
  columns = ["r_0", "r_1", "r_2"],
  aggregations = [F.avg, F.stddev],
  timestamp_column = "timestamp",
  window_length = "20 seconds",
  tumble_length = "10 seconds"
):

  output = []
  for column in columns:
    for agg_function in aggregations:
      agg_column = input_df \
        .withWatermark("timestamp", window_length) \
        .groupBy(
          F.window(timestamp_column, window_length, tumble_length),
          timestamp_column,
      ) \
        .agg(avg(column)) \
        .withColumn("start", F.col("window.start")) \
        .withColumn("end", F.col("window.end")) \
        .drop("window") \
        .withColumn("uuid", F.expr("uuid()")))
      output.append(agg_column)

  return output

list_df = generate_aggregations(
  input_df = gold_df
)

# COMMAND ----------

agg_column = gold_df \
  .withWatermark("timestamp", "50 seconds") \
  .select("r_0", "r_1", "r_2", "timestamp") \
  .groupBy(
    "timestamp",
    F.window("timestamp", "50 seconds", "30 seconds"),
) \
  .agg(
    F.avg("r_0"),
    F.avg("r_1"),
    F.avg("r_2"),
    F.stddev("r_0"),
    F.stddev("r_1"),
    F.stddev("r_2")
  ) \
  .withColumn("start", F.col("window.start")) \
  .withColumn("end", F.col("window.end")) \
  .withColumn("unix_ts", F.col("timestamp").cast("double")) \
  .drop("window")

# COMMAND ----------

display(agg_column)

# COMMAND ----------

#Write Stream from Gold DF

gold_df.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("mergeSchema", "true") \
  .option("checkpointLocation", checkpoint_location_target) \
  .trigger(processingTime = "30 seconds") \
  .table(f"{database}.{target_table}")

# COMMAND ----------

#Write Aggregations DF Stream

agg_column \
  .withColumnRenamed("avg(r_0)", "avg_r_0") \
  .withColumnRenamed("avg(r_1)", "avg_r_1") \
  .withColumnRenamed("avg(r_2)", "avg_r_2") \
  .withColumnRenamed("stddev_samp(r_0)", "stddev_r_0") \
  .withColumnRenamed("stddev_samp(r_1)", "stddev_r_1") \
  .withColumnRenamed("stddev_samp(r_2)", "stddev_r_2") \
  .writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("mergeSchema", "true") \
  .option("checkpointLocation", f"{checkpoint_location_target}/agg") \
  .trigger(processingTime = "30 seconds") \
  .table(f"{database}.{target_table}_agg")

# COMMAND ----------


