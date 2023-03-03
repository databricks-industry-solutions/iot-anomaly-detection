# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Predict Anomalous Events
# MAGIC 
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/iot-anomaly-detection/raw/main/resource/images/06_inference.jpg" width="40%">
# MAGIC 
# MAGIC This notebook will use the trained model to identify anomalous events.

# COMMAND ----------

dbutils.widgets.text("database", "rvp_iot_sa")
dbutils.widgets.text("source_table", "silver")
dbutils.widgets.text("target_table", "gold")
dbutils.widgets.text("model_name", "iot_anomaly_detection_xgboost")

database = getArgument("database")
source_table = getArgument("source_table")
target_table = getArgument("target_table")
model_name = getArgument("model_name")

#Cleanup from previous run(s)
checkpoint_path = "/dbfs/tmp/checkpoints"
checkpoint_location_target = f"{checkpoint_path}/{target_table}"
dbutils.fs.rm(checkpoint_location_target, recurse = True)
spark.sql(f"drop table if exists {database}.{target_table}")

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_info = client.get_registered_model(model_name)
production_version = [
  version for version
  in model_info.latest_versions
  if version.current_stage == 'Production'
][0]
print(f"Production version: {production_version.version}")

#Load model artifact
pipeline_model = mlflow.spark.load_model(production_version.source)

# COMMAND ----------

from pyspark.sql import functions as F

gold_df = (
  spark.readStream
    .format("delta")
    .table(f"{database}.{source_table}")
)

gold_df_pred = pipeline_model.transform(gold_df)

#Uncomment to display gold_df_pred
#display(gold_df_pred)

# COMMAND ----------

display(gold_df_pred)

# COMMAND ----------

(gold_df_pred
  .select("device_id", "datetime", "prediction")
  .writeStream
  .format("delta")
  .outputMode("append")
  .option("checkpointLocation", f"{checkpoint_location_target}")
  .trigger(once = True)
  .table(f"{database}.{target_table}")
)

# COMMAND ----------

display(spark.table(f"{database}.{target_table}"))
