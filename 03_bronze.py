# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Stream the Data from Kafka into a Bronze Delta Table
# MAGIC 
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/iot-anomaly-detection/raw/main/resource/images/03_bronze.jpg" width="30%">
# MAGIC 
# MAGIC This notebook will read the IoT data from Kafka and put it into a Delta Lake table called "Bronze".

# COMMAND ----------

# MAGIC %md
# MAGIC Setup

# COMMAND ----------

dbutils.widgets.text("target_table", "bronze")
dbutils.widgets.text("database", "rvp_iot_sa")
checkpoint_path = "/dbfs/tmp/checkpoints"

target_table = getArgument("target_table")
database = getArgument("database")
checkpoint_location_target = f"{checkpoint_path}/{target_table}"

# Cleans up previous run(s)
dbutils.fs.rm(checkpoint_location_target, recurse = True)
spark.sql(f"create database if not exists {database}")
spark.sql(f"drop table if exists {database}.{target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC Ingest Raw Data from Kafka

# COMMAND ----------

#Kafka config
kafka_bootstrap_servers = "pkc-1wvvj.westeurope.azure.confluent.cloud:9092"
security_protocol = "SASL_SSL"
sasl_mechanism = "PLAIN"
sasl_username = "ER4NIW5GR6FLOTMD"
sasl_password = "TaeXQO2jSEbk6vhK32obQGF2O3WMoX9KrwL0zfaFBnKORLQSqVCKNPsaRe7IO0tT"
topic = "iot_msg_topic"
sasl_config = f'org.apache.kafka.common.security.plain.PlainLoginModule required username="{sasl_username}" password="{sasl_password}";'
checkpoint_path = "/dbfs/tmp/checkpoints"

options = {
  "kafka.ssl.endpoint.identification.algorithm": "https",
  "kafka.sasl.jaas.config": sasl_config,
  "kafka.sasl.mechanism": sasl_mechanism,
  "kafka.security.protocol" : security_protocol,
  "kafka.bootstrap.servers": kafka_bootstrap_servers,
  "group.id": 1,
  "subscribe": topic,
  "topic": topic,
  "checkpointLocation": checkpoint_path,
#  "startingOffsets": "earliest"
}

#Stream the Kafka records into a Dataframe
kafka_df = (
  spark.readStream
    .format("kafka")
    .options(**options)
    .option("startingOffsets", "latest")
    .load()
)

#Uncomment to display the Kafka records
#display(kafka_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Write the Kafka data to Bronze Delta table

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

(
  kafka_df
    .withColumn(
      "parsedValue",
      F.col("value").cast("string")
    )
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_location_target)
    .trigger(once = True) #Comment to leave stream running
    .toTable(f"{database}.{target_table}")
)

# COMMAND ----------

#Display records from the Bronze table
display(spark.table(f"{database}.{target_table}"))
