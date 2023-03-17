# Databricks notebook source
# MAGIC %md You may find this series of notebooks at https://github.com/databricks-industry-solutions/iot-anomaly-detection. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Stream the Data from Kafka into a Bronze Delta Table
# MAGIC 
# MAGIC <hr/>
# MAGIC <center><img src="https://github.com/databricks-industry-solutions/iot-anomaly-detection/blob/main/images/03_bronze.jpg?raw=true" width="30%"></center>
# MAGIC 
# MAGIC 
# MAGIC This notebook will read the IoT data from Kafka and put it into a Delta Lake table called "Bronze".

# COMMAND ----------

# DBTITLE 1,Define configs that are consistent throughout the accelerator
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# DBTITLE 1,Define config for this notebook 
# define target table for this notebook
dbutils.widgets.text("target_table", "bronze")
target_table = getArgument("target_table")
checkpoint_location_target = f"{checkpoint_path}/{target_table}"

# COMMAND ----------

# spark.sql(f"drop database if exists {database} cascade") # uncomment if you want to reinitialize the accelerator database

# COMMAND ----------

# DBTITLE 1,Ingest Raw Data from Kafka
options = {
  "kafka.ssl.endpoint.identification.algorithm": "https",
  "kafka.sasl.jaas.config": sasl_config,
  "kafka.sasl.mechanism": sasl_mechanism,
  "kafka.security.protocol" : security_protocol,
  "kafka.bootstrap.servers": kafka_bootstrap_servers,
  "group.id": 1,
  "subscribe": topic,
  "topic": topic,
  "checkpointLocation": checkpoint_path
}

#Stream the Kafka records into a Dataframe
kafka_df = (
  spark.readStream
    .format("kafka")
    .options(**options)
    .load()
)

#Uncomment to display the Kafka records
# display(kafka_df)

# COMMAND ----------

# MAGIC %md Here we generate some data and pump the data into the kafka topic. For your use case, if there is a Kafka topic with data continuously arriving, you can skip the following data generation step.

# COMMAND ----------

# MAGIC %md
# MAGIC Write the Kafka data to Bronze Delta table

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

query = (
  kafka_df
    .withColumn(
      "parsedValue",
      F.col("value").cast("string")
    )
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_location_target)
    .trigger(processingTime='30 seconds') # Use `.trigger(availableNow=True)` if you do NOT want to run the stream continuously, only to process available data since the last time it ran
    .toTable(f"{database}.{target_table}")
)

# COMMAND ----------

# MAGIC %run ./util/generate-iot-data

# COMMAND ----------

# DBTITLE 1,We can stop the query shortly after the data loading is finished 
import time
time.sleep(300)
query.stop()

# COMMAND ----------

#Display records from the Bronze table
display(spark.table(f"{database}.{target_table}"))
