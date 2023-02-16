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

input_features = ["sensor_1", "sensor_2", "sensor_3", "device_model", "state"]
label = "anomaly"
columns = input_features + [label]
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

evaluator = BinaryClassificationEvaluator()
evaluator.setRawPredictionCol("prediction")
evaluator.setLabelCol("anomaly")

xgboost = SparkXGBClassifier(
    features_col="features", 
    label_col=label, 
    prediction_col="prediction",
    n_estimators = 3,
    max_depth = 5
)

# setup the pipeline
pipeline = Pipeline(stages=[model_indexer, state_indexer, vector_assembler, xgboost])


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
