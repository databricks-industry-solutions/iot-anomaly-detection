# Databricks notebook source
# DBTITLE 1,Ingesting data from Kafka into Raw Layer
# get secret credentials
kafka_bootstrap_servers = "b-1.oetrta-kafka.oz8lgl.c3.kafka.us-west-2.amazonaws.com:9094,b-2.oetrta-kafka.oz8lgl.c3.kafka.us-west-2.amazonaws.com:9094"
topic = "iot_msg_01_2023"

# COMMAND ----------

starting_offsets = "earliest"
msk_df = (
  spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
    .option("kafka.security.protocol", "SSL")
    .option("subscribe", topic)
    .option("startingOffsets", starting_offsets)
    .load()
).repartition(16)

# COMMAND ----------

# DBTITLE 1,Sample from our ingested Kafka IoT Stream
display(msk_df)

# COMMAND ----------

# DBTITLE 1,Write into Raw Layer
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

spark.sql("create database if not exists rvp_iot_sa")
spark.sql("drop table if exists rvp_iot_sa.raw")
checkpoint_path = "/dbfs/tmp/checkpoints/"
table_checkpoint_path = f"{checkpoint_path}/raw_table"
dbutils.fs.rm(table_checkpoint_path, recurse = True)

(
  msk_df
    .withColumn(
      "parsedValue",
      F.col("value").cast("string")
    )
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", table_checkpoint_path)
    .trigger(once = True)
    .toTable("rvp_iot_sa.raw")
)

# COMMAND ----------

display(spark.readStream.table("rvp_iot_sa.raw"))

# COMMAND ----------

# Shut down all streaming queries after a while
import time
time.sleep(300) # wait 5 minutes
for s in spark.streams.active:
  s.stop()
