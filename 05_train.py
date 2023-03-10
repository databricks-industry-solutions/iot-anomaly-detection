# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Build Test/Train Datasets and Train Model
# MAGIC 
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/iot-anomaly-detection/raw/main/resource/images/05_train_model.jpg" width="25%">
# MAGIC 
# MAGIC This notebook will label the Silver data, create training and test datasets from the labeled data, train a machine learning model, and deploy the model the MLflow model registry.

# COMMAND ----------

# MAGIC %md
# MAGIC Setup

# COMMAND ----------

dbutils.widgets.text("source_table", "silver")
dbutils.widgets.text("target_table", "feaures")
dbutils.widgets.text("database", "rvp_iot_sa")
dbutils.widgets.text("model_name", "iot_anomaly_detection_xgboost")
checkpoint_path = "/dbfs/tmp/checkpoints"

source_table = getArgument("source_table")
target_table = getArgument("target_table")
database = getArgument("database")
model_name = getArgument("model_name")
checkpoint_location_target = f"{checkpoint_path}/dataset"

#Cleanup Previous Run(s)
dbutils.fs.rm(checkpoint_location_target, recurse = True)
spark.sql(f"drop table if exists {database}.{target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC Read and label the Silver data

# COMMAND ----------

from pyspark.sql import functions as F

#Read the Silver Data
silver_df = spark.table(f"{database}.{source_table}")

#Uncomment to display silver_df
#display(silver_df)

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import *

#Label the Silver data
labeled_df = (
  silver_df
    .withColumn("anomaly", when(col('sensor_1') > 80, 1).when(col('sensor_1') < 10, 1).when(col('sensor_1') > 65, round(rand(1))).when(col('sensor_1') < 25, round(rand(1))).otherwise(0))
)

#Display the labeled data
display(labeled_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Create features and store to feature table

# COMMAND ----------

import pyspark.pandas as ps

def compute_features(data):
  
  #Conver to Pandas for Spark
  data_pdf = data.to_koalas()

  #OHE
  data_pdf = ps.get_dummies(data_pdf, 
                        columns=['device_model', 'state'],dtype = 'int64')

  #Convert to Spark
  data_sdf = data_pdf.to_spark() 
    
  return data_sdf

# COMMAND ----------

features_df = (
  compute_features(labeled_df)
)

display(features_df)

# COMMAND ----------

#Save Feature to Delta Table
features_df.write.saveAsTable(f"{database}.{target_table}", mode = "overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC Create Training and Test Datasets

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
mlflow.spark.autolog()
mlflow.sklearn.autolog()

# Read data
data = features_df.toPandas().drop(["device_id", "datetime"], axis=1)

train, test = train_test_split(data, test_size=0.30, random_state=206)
colLabel = 'anomaly'

# The predicted column is colLabel which is a scalar from [3, 9]
train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[colLabel]
test_y = test[colLabel]

# COMMAND ----------

# MAGIC %md
# MAGIC Initial Training Run

# COMMAND ----------

from sklearn.metrics import *
mlflow.spark.autolog()
mlflow.sklearn.autolog()

# Begin training run
max_depth = 4
max_leaf_nodes = 32

with mlflow.start_run(run_name="skl") as run:
    run_id = run.info.run_uuid
    
    model = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    
#You can look at the experiment logging including parameters, metrics, recall curves, etc. by clicking the "experiment" link below or the MLflow Experiments icon in the right navigation pane

# COMMAND ----------

# MAGIC %md
# MAGIC Hyper-Parameter Tuning with Hyperopt

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
import numpy as np

search_space = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(0,46)),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', range(4,128))
}

def train_model(params):
  mlflow.sklearn.autolog()
  
  with mlflow.start_run(nested=True):
   
   # Fit, train, and score the model
    model = DecisionTreeClassifier(**params)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    return {'status': STATUS_OK, 'loss': f1_score(test_y, predictions)} #, 'params': model.get_params()}
  
with mlflow.start_run(run_name='sklearn_hyperopt') as run:
  
  best_params = fmin(
    fn = train_model,
    space = search_space,
    algo = tpe.suggest,
    max_evals = 20,
    trials = SparkTrials()
  )
  
run_id = run.info.run_uuid
experiment_id = run.info.experiment_id

#You can look at the experiment logging including parameters, metrics, recall curves, etc. by clicking the "experiment" link below or the MLflow Experiments icon in the right navigation pane

# COMMAND ----------

# MAGIC %md
# MAGIC Find and register the best model

# COMMAND ----------

from pyspark.sql.functions import *

experiment_Df = spark.read.format("mlflow-experiment").load(experiment_id)

#Find the best run based on F1 score
best_run = (
  experiment_Df
    .filter(
      experiment_Df.tags["mlflow.rootRunId"]==run_id)
    .orderBy(experiment_Df.metrics["training_f1_score"].desc())
    .limit(1)
    .first()['run_id']
)

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_uri = f"runs:/{best_run}/model"

#Register the model
model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

#Transition the model to "Production" stage in the registry
client.transition_model_version_stage(
  name = model_name,
  version = model_details.version,
  stage="Production"
)

# COMMAND ----------

#Run if a previous model was already deployed to transition the previous model to "None" stage in the registry
# client.transition_model_version_stage(
#   name = model_name,
#   version = int(model_details.version) - 1,
#   stage="None"
# )

# COMMAND ----------


