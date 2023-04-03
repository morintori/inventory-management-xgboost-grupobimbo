import pyspark
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import gc
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, concat, log, rand, isnan, count, when, exp
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline, PipelineModel

cwd = os.getcwd()

spark = SparkSession.builder.appName('Practise').getOrCreate()
state = spark.read.option("header", True).csv('town_state.csv', inferSchema=True)
state_list = state.toPandas()['State'].unique()
for x in state_list:


    prdf = spark.read.parquet(cwd + '/old/predict_test' + x + '.parquet')
    prdf = prdf.drop('features', 'prediction')

    if x == 'MÉXICO, D.F.':
        os.rename('pipelineModelMÉXICO, D.F.','pipelineModelMÉXICO DF')
        pM = PipelineModel.load('pipelineModelMÉXICO DF')
    else
        pM = PipelineModel.load('pipelineModel' + x)

    pred10 = prdf.filter(col('Semana') == 10)
    # ids10 = pred10['id']
    pred10 = pM.transform(pred10)
    pred10 = pred10.withColumn('prediction', lit(exp(col('prediction')) - 1))

    pred10_lag = pred10[['Cliente_ID', 'Producto_ID', 'prediction']]
    pred10_lag = pred10_lag.groupBy(['Cliente_ID', 'Producto_ID']).mean("prediction")
    pred10_lag = pred10_lag.withColumnRenamed('avg(prediction)', 'Lag1')

    pred11 = prdf.filter(col('Semana') == 11)
    pred11 = pred11.drop('Lag1')
    pred11 = pred11.join(pred10_lag,
                         on=['Cliente_ID', 'Producto_ID'],
                         how='left')

    pred11 = pred11.fillna(0)
    pred11 = pM.transform(pred11)
    pred11 = pred11.withColumn('prediction', lit(exp(col('prediction')) - 1))

    predict_test = pred10['id', 'prediction'].unionByName(pred11['id', 'prediction'], allowMissingColumns=True)

    predict_test.show()
    predict_test.write.save('predict_test' + x + '.parquet')
    del predict_test
    gc.collect()

for x in state_list:
    prdf = spark.read.parquet('predict_test' + x + '.parquet')
    data = data.unionByName(prdf, allowMissingColumns=True)


data = data.withColumn('prediction', round('prediction',1))
data = data.withColumnRenamed('prediction','Demanda_uni_equil')
data.show()
data.coalesce(1).write.options(header='True', delimiter = ',').mode().csv('pipeline_results')