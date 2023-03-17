# Databricks notebook source
# MAGIC %md 
# MAGIC %md 
# MAGIC # Real-time Monitoring and Anomaly Detection on Streaming IoT pipelines in Manufacturing 
# MAGIC 
# MAGIC Manufacturers today face challenges working with IoT data due to high investment costs, security, and connectivity outages. These challenges lead to more time and money being spent on trying to make things work rather than innovating on data products that drive business value.  The Databricks Lakehouse platform reduces these challenges with a reliable, secure platform capable of ingesting and transforming IoT data at massive scale, building analytics and AI assets on that data, and serving those assets where they are needed
# MAGIC 
# MAGIC In this solution accelerator, we show how to build a streaming pipeline for IoT data, train a machine learning model on that data, and use that model to make predictions on new IoT data.
# MAGIC 
# MAGIC The pattern shown consumes data from an Apache Kafka stream. Kafka is a distributed event streaming message bus that combines the best features of queuing and publish-subscribe technologies. [Kafka connectors for Spark\\({^T}{^M}\\) Structured Streaming](https://docs.databricks.com/structured-streaming/kafka.html) are packaged together within the Databricks runtime, making it easy to get started. Using these connectors, data from Kafka streams can easily be persisted into Delta Lakehouse. From there, advanced analytics or machine learning algorithms may be executed on the data.
# MAGIC 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/iot-anomaly-detection. 
# MAGIC 
# MAGIC <p></p>
# MAGIC 
# MAGIC <img src="https://github.com/databricks-industry-solutions/iot-anomaly-detection/blob/main/images/iot_streaming_lakehouse.png?raw=true" width=75%/>
# MAGIC 
# MAGIC <p></p>
# MAGIC 
# MAGIC ## Stream the Data from Kafka into a Bronze Delta Table
# MAGIC 
# MAGIC <p></p>
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

# MAGIC %md 
# MAGIC 
# MAGIC ### Write the Kafka data to Bronze Delta table
# MAGIC 
# MAGIC Here we generate some data and pump the data into the kafka topic. For your use case, if there is a Kafka topic with data continuously arriving, you can skip the following data generation step.

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
