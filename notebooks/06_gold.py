# Databricks notebook source
# DBTITLE 1,Reading data from Silver Layer and Writing into Gold
dbutils.widgets.text("source_table", "silver")
dbutils.widgets.text("target_table", "gold")
dbutils.widgets.text("database", "rvp_iot_sa")
checkpoint_path = "/dbfs/tmp/checkpoints"

source_table = getArgument("source_table")
target_table = getArgument("target_table")
database = getArgument("database")
checkpoint_location_target = f"{checkpoint_path}/{target_table}"

# Remove this if not testing
dbutils.fs.rm(checkpoint_location_target, recurse = True)
spark.sql(f"drop table if exists {database}.{target_table}")

# COMMAND ----------

from pyspark.sql import functions as F

startingOffsets = "earliest"

gold_df = spark.readStream \
  .format("delta") \
  .option("startingOffsets", startingOffsets) \
  .table(f"{database}.{source_table}")

display(gold_df)

# COMMAND ----------

gold_df.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", checkpoint_location_target) \
  .option("mergeSchema", "true") \
  .trigger(once = True) \
  .table(f"{database}.{target_table}")

# COMMAND ----------

# DBTITLE 1,Training an ML Model
from pyspark.ml.tuning import TrainValidationSplitModel
from pyspark.sql import functions as F

input_features = ["sensor_1", "sensor_2", "sensor_3", "device_model", "state"]
label = "anomaly"
columns = input_features + [label]
df = spark.sql(f'select {",".join(columns)} from {database}.{target_table}').sample(0.1)
df.count()
df.persist()
df.count()

df = df.withColumn("anomaly", F.round(F.rand(seed=42)))
train_df, val_df, test_df = df.randomSplit([0.7, 0.15, 0.15], seed=26)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler

train_features = [feature for feature in input_features if "sensor" in feature]
train_features.extend(["device_model_idx", "state_idx"])
print(train_features)

model_indexer = StringIndexer()\
  .setInputCol("device_model")\
  .setOutputCol("device_model_idx")\
  .setHandleInvalid("keep")
    
state_indexer = StringIndexer()\
  .setInputCol("state")\
  .setOutputCol("state_idx")\
  .setHandleInvalid("keep")

vector_assembler = VectorAssembler()\
  .setInputCols(train_features)\
  .setOutputCol("features")

# COMMAND ----------

from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline
import mlflow

xgboost = SparkXGBClassifier(
    features_col="features", 
    label_col=label, 
    prediction_col="prediction",
    n_estimators = 3,
    max_depth = 5
)

# setup the pipeline
pipeline = Pipeline(stages=[model_indexer, state_indexer, vector_assembler, xgboost])

with mlflow.start_run(run_name = "xgboost_iot") as run:

  pipeline_model = pipeline.fit(train_df)
  train_df_pred = pipeline_model.transform(train_df)

display(train_df_pred)

# COMMAND ----------

display(train_df_pred.groupBy("prediction").count())

# COMMAND ----------

gold_df_pred = pipeline_model.transform(gold_df)
display(gold_df_pred)

# COMMAND ----------

# DBTITLE 1,Write Predictions as a Streaming Delta Table
gold_df_pred.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", f"{checkpoint_location_target}_pred") \
  .option("mergeSchema", "true") \
  .trigger(once = True) \
  .table(f"{database}.{target_table}_pred")
