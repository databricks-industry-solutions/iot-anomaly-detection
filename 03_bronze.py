# Databricks notebook source
dbutils.widgets.text("target_table", "bronze")
dbutils.widgets.text("database", "rvp_iot_sa")
checkpoint_path = "/dbfs/tmp/checkpoints"

target_table = getArgument("target_table")
database = getArgument("database")
checkpoint_location_target = f"{checkpoint_path}/{target_table}"

# Remove this if not testing
dbutils.fs.rm(checkpoint_location_target, recurse = True)
spark.sql(f"create database if not exists {database}")
spark.sql(f"drop table if exists {database}.{target_table}")

# COMMAND ----------

# DBTITLE 1,Ingesting data from Kafka into Raw Layer
# kafka config
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
  "startingOffsets": "earliest"
}

# COMMAND ----------

starting_offsets = "earliest"
kafka_df = (
  spark.readStream
    .format("kafka")
    .options(**options)
    .load()
).repartition(16)

# COMMAND ----------

# DBTITLE 1,Sample from our ingested Kafka IoT Stream
#display(kafka_df)

# COMMAND ----------

# DBTITLE 1,Write into Raw Layer
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
    .trigger(once = True)
    .toTable(f"{database}.{target_table}")
)

# COMMAND ----------

#display(spark.readStream.table("rvp_iot_sa.raw"))
