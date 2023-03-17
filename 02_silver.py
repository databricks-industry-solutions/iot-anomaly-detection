# Databricks notebook source
# MAGIC %md You may find this series of notebooks at https://github.com/databricks-industry-solutions/iot-anomaly-detection. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Parse/Transform the data from Bronze and load to Silver
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/iot-anomaly-detection/main/images/04_silver.jpg" width="50%">
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC This notebook will stream new events from the Bronze table, parse/transform them, and load them to a Delta table called "Silver".

# COMMAND ----------

# DBTITLE 1,Define configs that are consistent throughout the accelerator
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# DBTITLE 1,Define config for this notebook 
dbutils.widgets.text("source_table", "bronze")
dbutils.widgets.text("target_table", "silver")
source_table = getArgument("source_table")
target_table = getArgument("target_table")
checkpoint_location_target = f"{checkpoint_path}/{target_table}"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Incrementally Read data from Bronze

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType

bronze_df = (
  spark.readStream
    .format("delta")
    .table(f"{database}.{source_table}")
)

#Uncomment to view the bronze data
#display(bronze_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parse/Transform the Bronze data

# COMMAND ----------

#Schema for the Payload column
json_schema = StructType([
  StructField("timestamp", IntegerType(), True),
  StructField("device_id", IntegerType(), True),
  StructField("device_model", StringType(), True),
  StructField("sensor_1", FloatType(), True),
  StructField("sensor_2", FloatType(), True),
  StructField("sensor_3", FloatType(), True),
  StructField("state", StringType(), True)
])

#Parse/Transform
transformed_df = (
  bronze_df
    .withColumn("struct_payload", F.from_json("parsedValue", schema = json_schema)) #Apply schema to payload
    .select("struct_payload.*", F.from_unixtime("struct_payload.timestamp").alias("datetime"))
    .drop('timestamp')
            )

#Uncomment to display the transformed data
#display(transformed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write transformed data to Silver

# COMMAND ----------

(
  transformed_df
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_location_target)
    .trigger(once = True) # or use .trigger(processingTime='30 seconds') to continuously stream and feel free to modify the processing window
    .table(f"{database}.{target_table}")
)

# COMMAND ----------

#Display Silver Table
display(spark.table(f"{database}.{target_table}"))
