# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Predict Anomalous Events
# MAGIC 
# MAGIC <img src="https://github.com/databricks-industry-solutions/iot-anomaly-detection/raw/main/resource/images/06_inference.jpg" width="20%">
# MAGIC 
# MAGIC This notebook will use the trained model to identify anomalous events.

# COMMAND ----------

# MAGIC %md
# MAGIC Setup

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

# MAGIC %md
# MAGIC Read Silver Data

# COMMAND ----------

#Read Silver Data
silver_df = (
  spark.readStream
    .format("delta")
    .table(f"{database}.{source_table}")
)

# COMMAND ----------

# MAGIC %md Create a function to featurize and make the prediction

# COMMAND ----------

import pyspark.pandas as ps
import mlflow

#Feature and predict function
def predict_anomalies(data, epoch_id):
  
  #Conver to Pandas for Spark
  data_pdf = data.to_koalas()

  #OHE
  data_pdf = ps.get_dummies(data_pdf, 
                        columns=['device_model', 'state'],dtype = 'int64')

  #Convert to Spark
  data_sdf = data_pdf.to_spark() 
  
  # Load the model
  model = f'models:/{model_name}/production'
  model_fct = mlflow.pyfunc.spark_udf(spark, model_uri=model)

  # Make the prediction
  prediction_df = data_sdf.withColumn('prediction', model_fct(*data_sdf.drop('datetime', 'device_id').columns))
  
  # Clean up the output
  clean_pred_df = (prediction_df.select('device_id', 'datetime', 'sensor_1', 'sensor_2', 'sensor_3', 'prediction'))
  
  # Write the output to a Gold Delta table
  clean_pred_df.write.format('delta').mode('append').saveAsTable(f"{database}.{target_table}")
   
#  return data_sdf

# COMMAND ----------

# MAGIC %md Stream the predicted results using the function

# COMMAND ----------

# Stream predicted outputs
(
  silver_df
    .writeStream
    .foreachBatch(predict_anomalies)
    .trigger(once = True)
    .start()
)

# COMMAND ----------

display(spark.table(f"{database}.{target_table}"))
