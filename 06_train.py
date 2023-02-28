# Databricks notebook source
dbutils.widgets.text("database", "rvp_iot_sa")
dbutils.widgets.text("train_table", "train")
dbutils.widgets.text("test_table", "test")
dbutils.widgets.text("experiment_name", "/Shared/rvp_iot_sa_anomaly")

database = getArgument("database")
train_table = getArgument("train_table")
test_table = getArgument("test_table")
experiment_name = getArgument("experiment_name")

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
train_df = spark.sql(f'select {",".join(columns)} from {database}.{train_table}')
test_df = spark.sql(f'select {",".join(columns)} from {database}.{test_table}')

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
model_name = "iot_anomaly_detection_xgboost"
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
  #mlflow.log_param("max_depth", max_depth)
  #mlflow.log_param("num_estimators", num_estimators)
  
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
