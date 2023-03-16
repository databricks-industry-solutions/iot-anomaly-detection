# Databricks notebook source
# MAGIC %md 
# MAGIC # Real-time Monitoring and Anomaly Detection on Streaming IoT Pipelines in Manufacturing 
# MAGIC 
# MAGIC Manufacturers today face challenges working with IoT data due to high investment costs, security, and connectivity outages. These challenges lead to more time and money being spent on trying to make things work rather than innovating on data products that drive business value.  The Databricks Lakehouse platform reduces these challenges with a reliable, secure platform capable of ingesting and transforming IoT data at massive scale, building analytics and AI assets on that data, and serving those assets where they are needed
# MAGIC 
# MAGIC In this solution accelerator, we show how to build a streaming pipeline for IoT data, train a machine learning model on that data, and use that model to make predictions on new IoT data.
# MAGIC 
# MAGIC The pattern shown consumes data from an Apache Kafka stream. Kafka is a distributed event streaming message bus that combines the best features of queuing and publish-subscribe technologies. [Kafka connectors for Spark\\({^T}{^M}\\) Structured Streaming](https://docs.databricks.com/structured-streaming/kafka.html) are packaged together within the Databricks runtime, making it easy to get started. Using these connectors, data from Kafka streams can easily be persisted into Delta Lakehouse. From there, advanced analytics or machine learning algorithms may be executed on the data.
# MAGIC 
# MAGIC </br>
# MAGIC <img src="https://github.com/databricks-industry-solutions/iot-anomaly-detection/raw/main/resource/images/01_overview.jpg" alt="Data Flow" width="70%">
# MAGIC 
# MAGIC This solution accelerator will walk through creating and using an anomaly detection model with IoT data being ingested from Kafka, data written to and read from Delta Lake, and a model served from the MLflow model registry.
# MAGIC </br></br>
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/iot-anomaly-detection. 

# COMMAND ----------


