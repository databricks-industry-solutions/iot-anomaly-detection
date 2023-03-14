# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # IoT Data Generation
# MAGIC 
# MAGIC In this notebook, we use `dbldatagen` to generate fictitious data and push into a Kafka topic.
# MAGIC 
# MAGIC We first generate data using [Databricks Labs Data Generator](https://databrickslabs.github.io/dbldatagen/public_docs/index.html) (`dbldatagen`). The data generator provides an easy way to generate large volumes of synthetic data within a Databricks notebook. The data that is generated is defined by a schema. The output is a PySpark dataframe.
# MAGIC 
# MAGIC The generated data consists of the following columns: 
# MAGIC - `device_id`
# MAGIC - `device_model`
# MAGIC - `timestamp`
# MAGIC - `sensor_1`
# MAGIC - `sensor_2`
# MAGIC - `sensor_3`
# MAGIC - `us_state`
# MAGIC 
# MAGIC where `sensor 1..3` are sensor values. 

# COMMAND ----------

dbutils.fs.rm("/dbfs/tmp/checkpoints", True)

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

data_rows = 2000
df_spec = (
  dg.DataGenerator(
    spark,
    name="test_data_set1",
    rows=data_rows,
    partitions=4
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

# MAGIC %md
# MAGIC ## Writing into Kafka
# MAGIC The Producer sends the dataframe to a Confluent Kafka broker as follows. 
# MAGIC 
# MAGIC You will need to replace the `kafka_bootstrap_servers`, `sasl_username` and `sasl_password` variables to your on Kafka broker.

# COMMAND ----------

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
    "checkpointLocation": checkpoint_path
}


kafka_checkpoint_path = f"{checkpoint_path}/kafka/{topic}"
dbutils.fs.rm(kafka_checkpoint_path, recurse = True)

def delay(row):
    # Wait x seconds for each batch
    time.sleep(row_delay_seconds)
    pass

(
  streaming_df
    .writeStream
    .foreachBatch(delay)
    .format("kafka")
    .options(**options)
) \
  .trigger(once = True) \
  .start()

# COMMAND ----------


