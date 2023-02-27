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
  label_col: str,
  num_estimators: int,
  max_depth: int,
  features_col: str = "features",
  prediction_col: str = "prediction"
):

  evaluator = BinaryClassificationEvaluator()
  evaluator.setRawPredictionCol("prediction")
  evaluator.setLabelCol("anomaly")

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

  return pipeline

# COMMAND ----------

# DBTITLE 1,Setting up Hyperopt for Hyperparameter Tuning
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
 
def train_with_hyperopt(params):
  """
  This method is passed to hyperopt.fmin().
  
  :param params: hyperparameters as a dict. Its structure is consistent with how search space is defined. See below.
  :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
  """
  # For integer parameters, make sure to convert them to int type if Hyperopt is searching over a continuous range of values.
  minInstancesPerNode = int(params['minInstancesPerNode'])
  maxBins = int(params['maxBins'])
 
  model, f1_score = train_tree(minInstancesPerNode, maxBins)
  
  # Hyperopt expects you to return a loss (for which lower is better), so take the negative of the f1_score (for which higher is better).
  loss = - f1_score
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# DBTITLE 1,Defining the search space


n_estimators = 3,
    max_depth = 5

# COMMAND ----------

# MLflow Experiment & Run

experiment = mlflow.get_experiment_by_name(name = experiment_name)
experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(name = experiment_name)
mlflow.end_run()

with mlflow.start_run(experiment_id = experiment_id, run_name = "xgboost_iot") as run:

  pipeline_model = pipeline.fit(train_df)
  train_df_pred = pipeline_model.transform(train_df)
  test_df_pred = pipeline_model.transform(test_df)
  train_auroc = evaluator.evaluate(train_df_pred, {evaluator.metricName: "areaUnderROC"})
  test_auroc = evaluator.evaluate(test_df_pred, {evaluator.metricName: "areaUnderROC"})
  mlflow.log_metric("train_auroc", train_auroc)
  mlflow.log_metric("test_auroc", test_auroc)
  mlflow.spark.log_model(spark_model = pipeline_model, artifact_path = "model")
