# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## IoT Data Generation
# MAGIC 
# MAGIC In this notebook, we use `dbldatagen` to generate fictitious data and push into a Kafka topic.

# COMMAND ----------

dbutils.widgets.text("row_delay_seconds", "0.1")
row_delay_seconds = float(getArgument("row_delay_seconds"))

# COMMAND ----------

!pip install dbldatagen -q

# COMMAND ----------

import dbldatagen as dg
from pyspark.sql.types import IntegerType, FloatType, StringType, LongType

states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY' ]

table_name = "iot_stream_example_"
spark.sql(f"drop table if exists {table_name}")

data_rows = 10000
df_spec = (
  dg.DataGenerator(
    spark,
    name="test_data_set1",
    rows=data_rows,
    partitions=10
  )
  .withIdOutput()
  .withColumn("device_id", IntegerType(), minValue=1, maxValue=1000)
  .withColumn(
    "device_model",
    StringType(),
    values=['mx2000', 'xft-255', 'db-1000', 'db-2000', 'mlr-12.0'],
    random=True
  )
  .withColumn("timestamp", LongType(), minValue=1577833200, maxValue=1673714337, random=True)
  .withColumn("sensor_1", IntegerType(), minValue=-10, maxValue=100, random=True)
  .withColumn("sensor_2", IntegerType(), minValue=0, maxValue=10, random=True)
  .withColumn("sensor_3", FloatType(), minValue=0.0001, maxValue=1.0001, random=True)
  .withColumn("state", StringType(), values=states, random=True)
)
                            
df = df_spec.build()
df.write.saveAsTable(table_name)

# COMMAND ----------

# DBTITLE 1,First Glance into our Generated Data
sample_df = spark.sql(f"select * from {table_name} where id < 10")
display(sample_df)

# COMMAND ----------

# DBTITLE 1,Writing into Kafka
from pyspark.sql.functions import to_json, struct, col, cast
from pyspark.sql.types import BinaryType
import time

# We first read from our generated data

streaming_df = (
  spark.readStream
  .format("delta")
  .schema(sample_df.schema)
  .table(table_name)
  .select(
    col("id").cast(BinaryType()).alias("key"),
    to_json(
      struct(
        [col(column) for column in sample_df.columns]
      )
    ).cast(BinaryType()).alias("value")
  )
)

# COMMAND ----------

import time

# kafka config
kafka_bootstrap_servers = "b-1.oetrta-kafka.oz8lgl.c3.kafka.us-west-2.amazonaws.com:9094,b-2.oetrta-kafka.oz8lgl.c3.kafka.us-west-2.amazonaws.com:9094"
topic = "iot_msg_01_2023"
checkpoint_path = "/dbfs/tmp/checkpoints"
kafka_checkpoint_path = f"{checkpoint_path}/kafka/{topic}"
dbutils.fs.rm(kafka_checkpoint_path, recurse = True)

def delay(row):
    # Wait 1 second for each row
    time.sleep(row_delay_seconds)
    pass

(
  streaming_df
    .writeStream
    .foreach(delay)
    .format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
    .option("kafka.security.protocol", "SSL")
    .option("checkpointLocation", kafka_checkpoint_path)
    .option("topic", topic)
) \
  .trigger(once = True) \
  .start()
