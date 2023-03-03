# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Build Test/Train Datasets and Train Model
# MAGIC 
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/iot-anomaly-detection/raw/main/resource/images/05_train_model.jpg" width="40%">
# MAGIC 
# MAGIC This notebook will label the Silver data, create training and test datasets from the labeled data, train a machine learning model, and deploy the model the MLflow model registry.

# COMMAND ----------

# MAGIC %md
# MAGIC Setup

# COMMAND ----------

dbutils.widgets.text("source_table", "silver")
dbutils.widgets.text("target_table_train", "train")
dbutils.widgets.text("target_table_test", "test")
dbutils.widgets.text("database", "rvp_iot_sa")
dbutils.widgets.text("model_name", "iot_anomaly_detection_xgboost")
checkpoint_path = "/dbfs/tmp/checkpoints"

source_table = getArgument("source_table")
target_table_train = getArgument("target_table_train")
target_table_test = getArgument("target_table_test")
database = getArgument("database")
model_name = getArgument("model_name")
checkpoint_location_target = f"{checkpoint_path}/dataset"

#Cleanup Previous Run(s)
dbutils.fs.rm(checkpoint_location_target, recurse = True)
spark.sql(f"drop table if exists {database}.{target_table_train}")
spark.sql(f"drop table if exists {database}.{target_table_test}")

# COMMAND ----------



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
from pyspark.sql.functions import pandas_udf
from pyspark.sql import functions as F

#UDF to label anomalous records
@pandas_udf("int")
def label_data(sensor: pd.Series) -> pd.Series:

  anomaly = (sensor < 20) | (sensor > 80)
  anomaly = anomaly.astype(int)
  return anomaly

#Label the Silver data
labeled_df = (
  silver_df.withColumn("anomaly", label_data("sensor_1"))
)

#Display the labeled data
display(labeled_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Create Training and Test Datasets

# COMMAND ----------

#Split Training/Test datasets
train_df = labeled_df.filter(F.col("datetime") < "2021-09-01")
test_df = labeled_df.filter(
  (F.col("datetime") >= "2021-09-01")
  & (F.col("datetime") < "2022-01-01")
)

#Write Training/Test datasets to Delta tables
train_df.write.saveAsTable(f"{database}.train", mode = "overwrite")
test_df.write.saveAsTable(f"{database}.test", mode = "overwrite")

#Print Training/Test dataset counts
print(f"Training set contains {train_df.count()} rows")
print(f"Testing set contains {test_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC Prep data/features for training

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplitModel
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler

input_features = [
  "sensor_1",
  "sensor_2",
  "sensor_3",
  "device_model",
  "state"
]

label_col = "anomaly"
columns = input_features + [label_col]

train_features = [feature for feature in input_features if "sensor" in feature]
train_features.extend(["device_model_idx", "state_idx"])

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

# MAGIC %md
# MAGIC Setup model for training

# COMMAND ----------

from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/iot_anomaly'.format(username))

def make_pipeline(
  num_estimators: int,
  max_depth: int,
  features_col: str = "features",
  prediction_col: str = "prediction",
  label_col: str = "anomaly"
):

  evaluator = BinaryClassificationEvaluator()
  evaluator.setRawPredictionCol(prediction_col)
  evaluator.setLabelCol(label_col)

  xgboost = SparkXGBClassifier(
      features_col = features_col, 
      label_col = label_col, 
      prediction_col = prediction_col,
      num_estimators = num_estimators,
      max_depth = max_depth
  )

  pipeline = Pipeline(
    stages = [
      model_indexer,
      state_indexer,
      vector_assembler,
      xgboost
    ]
  )

  return pipeline, evaluator

# COMMAND ----------

from mlflow import spark as mlflow_spark

def train_gbt(max_depth, num_estimators):
  '''
  This train() function:
   - takes hyperparameters as inputs (for tuning later)
   - returns the F1 score on the validation dataset
 
  Wrapping code as a function makes it easier to reuse the code later with Hyperopt.
  '''
  # Use MLflow to track training.
  # Specify "nested=True" since this single model will be logged as a child run of Hyperopt's run.
  with mlflow.start_run(nested=True):
    mlflow.log_param("Model Type", "XGBoost")
    
    pipeline, evaluator = make_pipeline(
      max_depth = max_depth,
      num_estimators = num_estimators
    )
    
    # Train model.  This also runs the indexers.
    pipeline = pipeline.fit(train_df)
    
    # Make predictions.
    predictions = pipeline.transform(test_df)
    validation_metric = evaluator.evaluate(predictions)
  
  return pipeline, validation_metric

# COMMAND ----------

# MAGIC %md
# MAGIC Setup Hyperopt for hyper-parameter tuning

# COMMAND ----------

from hyperopt import hp, fmin, tpe, STATUS_OK

space = {
  'max_depth': hp.uniform('max_depth', 2, 15),
  'num_estimators': hp.uniform('num_estimators', 10, 50)
}

algo = tpe.suggest

def train_with_hyperopt_train_gbt(params):
  """
  An example train method that calls into MLlib.
  This method is passed to hyperopt.fmin().
  
  :param params: hyperparameters as a dict. Its structure is consistent with how search space is defined. See below.
  :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
  """
  # For integer parameters, make sure to convert them to int type if Hyperopt is searching over a continuous range of values.
  max_depth = int(params['max_depth'])
  num_estimators = int(params['num_estimators'])
 
  model, log_loss_score = train_gbt(max_depth, num_estimators)
  mlflow.log_metric("test_log_loss", log_loss_score)
  
  # Hyperopt expects you to return a loss (for which lower is better), so take the negative of the f1_score (for which higher is better).
  return {'loss': (-log_loss_score), 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC Train the Model

# COMMAND ----------

with mlflow.start_run() as run:
  best_params = fmin(
    fn = train_with_hyperopt_train_gbt,
    space = space,
    algo = algo,
    max_evals = 10
  )
  
  gradient_final_model, final_gradient_val_log_loss = train_gbt(
    int(best_params['max_depth']),
    int(best_params['num_estimators'])
  )
  
  mlflow.spark.log_model(
    gradient_final_model,
    "model",
    registered_model_name = model_name
  )

  # Capture the run_id to use when registering our model
  run_id = run.info.run_id
  experiment_id = run.info.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy best model to MLflow Model Registry

# COMMAND ----------

#load the experiments from the latest training
experiment_Df = spark.read.format("mlflow-experiment").load(experiment_id)


#filter the best run
best_run = (
  experiment_Df
    .filter(
      experiment_Df.tags["mlflow.rootRunId"]==run_id)
    .orderBy(experiment_Df.metrics["log_loss_score"].desc())
    .limit(1)
    .first()['run_id']
)

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_info = client.get_registered_model(model_name)
latest_version = model_info.latest_versions[-1]

client.transition_model_version_stage(
  name = model_name,
  version = latest_version.version,
  stage="Production"
)
