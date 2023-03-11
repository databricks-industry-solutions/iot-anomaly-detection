# Databricks notebook source
# MAGIC %md You may find this series of notebooks at https://github.com/databricks-industry-solutions/iot-anomaly-detection. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Build Test/Train Datasets and Train Model
# MAGIC 
# MAGIC <img src="https://github.com/databricks-industry-solutions/iot-anomaly-detection/raw/main/resource/images/05_train_model.jpg" width="25%">
# MAGIC 
# MAGIC This notebook will label the Silver data, create training and test datasets from the labeled data, train a machine learning model, and deploy the model the MLflow model registry.

# COMMAND ----------

# DBTITLE 1,Define configs that are consistent throughout the accelerator
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# DBTITLE 1,Define config for this notebook 
dbutils.widgets.text("source_table", "silver")
dbutils.widgets.text("target_table", "feaures")
source_table = getArgument("source_table")
target_table = getArgument("target_table")
checkpoint_location_target = f"{checkpoint_path}/dataset"

# COMMAND ----------

# DBTITLE 1,Reading data from Silver Layer and Writing into Gold
dbutils.widgets.text("source_table", "silver")
dbutils.widgets.text("target_table_train", "train")
dbutils.widgets.text("target_table_test", "test")

source_table = getArgument("source_table")
target_table_train = getArgument("target_table_train")
target_table_test = getArgument("target_table_test")
checkpoint_location_target = f"{checkpoint_path}/dataset"

# COMMAND ----------

from pyspark.sql import functions as F

static_df = spark.sql(f"SELECT * FROM {database}.{source_table}")
train_df = static_df.filter(F.col("datetime") < "2021-09-01")
test_df = static_df.filter(
  (F.col("datetime") >= "2021-09-01")
  & (F.col("datetime") < "2022-01-01")
)

print(f"Training set contains {train_df.count()} rows")
print(f"Testing set contains {test_df.count()} rows")

# COMMAND ----------

# MAGIC %md TODO: explain how the method below is overly simplistic and would train a perfect model to identify 20-80 as acceptable range. Explain how to get real labels 

# COMMAND ----------

# DBTITLE 1,Labelling our Data
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import functions as F

@pandas_udf("int")
def label_data(sensor: pd.Series) -> pd.Series:

  anomaly = (sensor < 20) | (sensor > 80)
  anomaly = anomaly.astype(int)
  return anomaly

train_df = train_df.withColumn("anomaly", label_data("sensor_1"))
test_df = test_df.withColumn("anomaly", label_data("sensor_1"))

train_df.write.saveAsTable(f"{database}.{target_table_train}", mode = "overwrite")
test_df.write.saveAsTable(f"{database}.{target_table_test}", mode = "overwrite")

# COMMAND ----------

# DBTITLE 1,Training an Anomaly Detection Model with XGBoost
from pyspark.ml.tuning import TrainValidationSplitModel
from pyspark.sql import functions as F

input_features = [
  "sensor_1",
  "sensor_2",
  "sensor_3",
  "device_model",
  "state"
]

label_col = "anomaly"
columns = input_features + [label_col]
train_df = spark.sql(f'select {",".join(columns)} from {database}.{target_table_train}')
test_df = spark.sql(f'select {",".join(columns)} from {database}.{target_table_test}')

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler

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

from hyperopt import hp, fmin, tpe, STATUS_OK
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

# DBTITLE 1,Hyperopt
space = {
  'max_depth': hp.uniform('max_depth', 2, 15),
  'num_estimators': hp.uniform('num_estimators', 10, 50)
}

algo = tpe.suggest

# COMMAND ----------

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

# COMMAND ----------

# DBTITLE 1,Transition the best model to Production
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_uri = f"runs:/{run_id}/model"

#Register the model
model_details = mlflow.register_model(model_uri, model_name)

client.transition_model_version_stage(
  name = model_name,
  version = model_details.version,
  stage="Production",
  archive_existing_versions=True
)

# COMMAND ----------


