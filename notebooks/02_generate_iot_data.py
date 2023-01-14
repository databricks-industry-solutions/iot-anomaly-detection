# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## IoT Data Generation
# MAGIC 
# MAGIC In this notebook, we use `dbldatagen` to generate fictitious data and push into a Kafka topic.

# COMMAND ----------

!pip install dbldatagen -q

# COMMAND ----------

import dbldatagen as dg
from pyspark.sql.types import IntegerType, FloatType, StringType

#Clean previous data / checkpoints
spark.sql("drop table if exists iot_stream_example")

column_count = 6
data_rows = 1000 * 1000 * 10 # 10 Million records
df_spec = (
  dg.DataGenerator(
    spark,
    name="test_data_set1",
    rows=data_rows,
    partitions=12
  )
  .withIdOutput()
  .withColumn(
    "r", FloatType(),
    expr="floor(rand() * 350) * (86400 + 3600)",
    numColumns=column_count
  )
  .withColumn("code1", IntegerType(), minValue=100, maxValue=200)
  .withColumn("code2", IntegerType(), minValue=0, maxValue=10)
  .withColumn("code3", StringType(), values=['a', 'b', 'c'])
  .withColumn("code4", StringType(), values=['a', 'b', 'c'], random=True)
  .withColumn("code5", StringType(), values=['a', 'b', 'c'], random=True, weights=[9, 1, 1])
)
                            
df = df_spec.build()
df.write.saveAsTable("iot_stream_example")

# COMMAND ----------

# DBTITLE 1,First Glance into our Generated Data
# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM iot_stream_example LIMIT 5

# COMMAND ----------

# DBTITLE 1,Writing into Kafka
from pyspark.sql.functions import to_json, struct, col, cast
from pyspark.sql.types import BinaryType
import time

# We first read from our generated data

streaming_df = (
  spark.readStream
  .format("delta")
  .table("iot_stream_example")
  .select(
    col("id").cast(BinaryType()).alias("key"),
    to_json(
      struct(
        col('r_0'),
        col('r_1'),
        col('r_2'),
        col('r_3'),
        col('r_4'),
        col('r_5'),
        col('code1'),
        col('code2'),
        col('code3')
      )
    ).alias("value")
  )
)

# COMMAND ----------

import time

# kafka config
kafka_bootstrap_servers = "b-1.oetrta-kafka.oz8lgl.c3.kafka.us-west-2.amazonaws.com:9094,b-2.oetrta-kafka.oz8lgl.c3.kafka.us-west-2.amazonaws.com:9094"
topic = "iot_msg_11_2022"
checkpoint_path = "/dbfs/tmp/checkpoints"
kafka_checkpoint_path = f"{checkpoint_path}/kafka"
dbutils.fs.rm(kafka_checkpoint_path, recurse = True)

def delay(row):
    # Wait 10 seconds for each row
    time.sleep(10)
    pass

(
  streaming_df
    .writeStream
    .foreachBatch(delay)
    .format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
    .option("kafka.security.protocol", "SSL")
    .option("checkpointLocation", kafka_checkpoint_path)
    .option("topic", topic)
) \
  .trigger(processingTime='10 seconds') \
  .start()

# COMMAND ----------


