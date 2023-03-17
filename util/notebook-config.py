# Databricks notebook source
# DBTITLE 1,Kafka config - see the RUNME notebook for instructions on setting up secrets
# Either configure a secret scope containing the credentials for authenticating with Kafka, or replace with your own credentials

kafka_bootstrap_servers = dbutils.secrets.get("solution-accelerator-cicd", "iot-anomaly-kafka-bootstrap-server")
security_protocol = "SASL_SSL"
sasl_mechanism = "PLAIN"
sasl_username = dbutils.secrets.get("solution-accelerator-cicd", "iot-anomaly-sasl-username")
sasl_password = dbutils.secrets.get("solution-accelerator-cicd", "iot-anomaly-sasl-password")
topic = "iot_msg_topic"
sasl_config = f'org.apache.kafka.common.security.plain.PlainLoginModule required username="{sasl_username}" password="{sasl_password}";'

# COMMAND ----------

# DBTITLE 1,Set database and streaming checkpoint
checkpoint_path = "/dbfs/tmp/iot-anomaly-detection/checkpoints" 
database = "rvp_iot_sa"

# COMMAND ----------

# dbutils.fs.rm(checkpoint_path, True) # resetting checkpoint - uncomment this out if you want to reset data for this accelerator 
# spark.sql(f"drop database if exists {database} cascade") # resetting database - comment this out if you want data to accumulate in tables for this accelerator over time

# COMMAND ----------

# DBTITLE 1,Database settings
spark.sql(f"create database if not exists {database}")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
model_name = "iot_anomaly_detection"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/iot_anomaly_detection'.format(username))

# COMMAND ----------


