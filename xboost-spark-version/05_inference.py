# Databricks notebook source
# MAGIC %md You may find this series of notebooks at https://github.com/databricks-industry-solutions/iot-anomaly-detection. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Predict Anomalous Events
# MAGIC 
# MAGIC <img src="https://github.com/databricks-industry-solutions/iot-anomaly-detection/raw/main/resource/images/06_inference.jpg" width="20%">
# MAGIC 
# MAGIC This notebook will use the trained model to identify anomalous events.

# COMMAND ----------

# DBTITLE 1,Define configs that are consistent throughout the accelerator
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# DBTITLE 1,Define config for this notebook 
dbutils.widgets.text("source_table", "silver")
dbutils.widgets.text("target_table", "gold")
source_table = getArgument("source_table")
target_table = getArgument("target_table")
checkpoint_location_target = f"{checkpoint_path}/{target_table}"

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

#Load model artifact

pipeline_model = mlflow.spark.load_model(f"models:/{model_name}/production")

# COMMAND ----------

from pyspark.sql import functions as F

silver_df = spark.readStream \
  .format("delta") \
  .table(f"{database}.{source_table}")

silver_df_pred = pipeline_model.transform(silver_df)

# COMMAND ----------

silver_df_pred \
  .select("device_id", "datetime", "prediction") \
  .writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", f"{checkpoint_location_target}") \
  .option("mergeSchema", "true") \
  .trigger(once = True) \
  .table(f"{database}.{target_table}") \
  .awaitTermination()

# COMMAND ----------

spark.table(f"{database}.{target_table}").display()

# COMMAND ----------


