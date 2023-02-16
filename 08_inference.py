# Databricks notebook source
dbutils.widgets.text("database", "rvp_iot_sa")
dbutils.widgets.text("source_table", "gold")
dbutils.widgets.text("target_table", "gold_pred")
dbutils.widgets.text("experiment_name", "/Shared/rvp_iot_sa_anomaly")
dbutils.widgets.text("model_path", "model")

database = getArgument("database")
source_table = getArgument("source_table")
target_table = getArgument("target_table")
experiment_name = getArgument("experiment_name")
model_path = getArgument("model_path")

checkpoint_path = "/dbfs/tmp/checkpoints"
checkpoint_location_target = f"{checkpoint_path}/{target_table}"
dbutils.fs.rm(checkpoint_location_target, recurse = True)
spark.sql(f"drop table if exists {database}.{target_table}")

# COMMAND ----------

import mlflow

experiment = mlflow.get_experiment_by_name(name = experiment_name)
experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(name = experiment_name)

df_runs = mlflow.search_runs(
  experiment_ids = [experiment_id],
  order_by = ["metrics.test_auroc DESC"]
)

run_id = df_runs.loc[0, "run_id"]
print(f"Best run ID: {run_id}")
df_runs

# COMMAND ----------

logged_model = f"runs:/{run_id}/{model_path}"
pipeline_model = mlflow.spark.load_model(logged_model)

# COMMAND ----------

from pyspark.sql import functions as F

startingOffsets = "earliest"

gold_df = spark.readStream \
  .format("delta") \
  .option("startingOffsets", startingOffsets) \
  .table(f"{database}.{source_table}")

inference_df = gold_df.withColumn("datetime", F.to_date(F.col("datetime"))).filter("datetime > '2022-01-01'")
gold_df_pred = pipeline_model.transform(inference_df)

# COMMAND ----------

gold_df_pred \
  .select("device_id", "state", "prediction", "datetime", "anomaly") \
  .writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", f"{checkpoint_location_target}") \
  .option("mergeSchema", "true") \
  .trigger(once = True) \
  .table(f"{database}.{target_table}_pred")

# COMMAND ----------

spark.table(f"{database}.{target_table}_pred").display()
