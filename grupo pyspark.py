import pyspark
import pandas as pd
import numpy as np
import xgboost as xgb

import gc
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, concat, log, rand, isnan, count, when,exp
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName('Practise').getOrCreate()
state = spark.read.option("header", True).csv('town_state.csv', inferSchema=True)
state_list = state.toPandas()['State'].unique()

for x in state_list:

    # if x == 'MÉXICO, D.F.':
    #     x = 'MÉXICO DF'

    train = spark.read.option("header", True).csv('train.csv', inferSchema=True)
    train.show(20)
    train = train.filter(train.Semana > 7)

    train = train.join(state,
                       on=['Agencia_ID'],
                       how='left')

    train = train.filter(train.State == x)
    train.show(20)
    test = spark.read.option("header", True).csv('test.csv', inferSchema=True)
    test = test.join(state,
                     on=['Agencia_ID'],
                     how='left')

    test = test.filter(test.State == x)
    train = train.withColumn("Demanda_uni_equil", train.Demanda_uni_equil.cast('int'))
    train = train.withColumn("Target", col("Demanda_uni_equil"))
    train = train.drop("Demanda_uni_equil")
    train = train.withColumn("tst", lit(0))
    test = test.withColumn('tst', lit(1))

    data = train.unionByName(test, allowMissingColumns=True)
    train.unpersist()
    test.unpersist()
    gc.collect()

    data.show(20)

    for i in range(1, 2):
        lag = 'Lag' + str(i)
        print('Lag:', lag)

        data1 = data[['Semana', 'Cliente_ID', 'Producto_ID', 'target']]
        data1 = data1.withColumn("Semana", lit(col("Semana") + i))

        data1 = data1.groupBy(['Semana', 'Cliente_ID', 'Producto_ID']).mean("target")
        data1 = data1.withColumnRenamed('avg(target)', lag)
        data = data.join(data1,
                         on=['Semana', 'Cliente_ID', 'Producto_ID'],
                         how='left')
        del data1
        gc.collect()


    data = data.filter(data.Semana > 8)
    data.show(20)

    nAgencia = data.filter(col('target').isNotNull()).groupBy(['Agencia_ID', 'Semana']).count()
    nAgencia = nAgencia.withColumnRenamed('count', 'nAgencia')
    nAgencia = nAgencia.groupBy(['Agencia_ID']).mean('nAgencia')
    nAgencia.show(20)

    data = data.join(nAgencia,
                     on=['Agencia_ID'],
                     how='left')

    del nAgencia
    gc.collect()

    nRuta_SAK = data.filter(col('target').isNotNull()).groupBy(['Ruta_SAK', 'Semana']).count()
    nRuta_SAK = nRuta_SAK.withColumnRenamed('count', 'nRuta_SAK')
    nRuta_SAK = nRuta_SAK.groupBy(['Ruta_SAK']).mean('nRuta_SAK')
    nRuta_SAK.show(20)

    data = data.join(nRuta_SAK,
                     on=['Ruta_SAK'],
                     how='left')

    del nRuta_SAK
    gc.collect()

    nCliente_ID = data.filter(col('target').isNotNull()).groupBy(['Cliente_ID', 'Semana']).count()
    nCliente_ID = nCliente_ID.withColumnRenamed('count', 'nCliente_ID')
    nCliente_ID = nCliente_ID.groupBy(['Cliente_ID']).mean('nCliente_ID')
    nCliente_ID.show()
    data = data.join(nCliente_ID,
                     on=['Cliente_ID'],
                     how='left')

    del nCliente_ID

    gc.collect()

    nProducto_ID = data.filter(col('target').isNotNull()).groupBy(['Producto_ID', 'Semana']).count()
    nProducto_ID = nProducto_ID.withColumnRenamed('count', 'nProducto_ID')
    nProducto_ID = nProducto_ID.groupBy(['Producto_ID']).mean('nProducto_ID')

    data = data.join(nProducto_ID,
                     on=['Producto_ID'],
                     how='left')

    del nProducto_ID

    gc.collect()

    data = data.fillna(0)
    train = data.filter(col('tst') == 0)
    predict = data.filter(col('tst') == 1)
    predict = predict['Producto_ID', 'Cliente_ID', 'Ruta_SAK', 'Agencia_ID', 'Semana', 'Canal_ID',
    'Lag1', 'avg(nAgencia)', 'avg(nRuta_SAK)', 'avg(nCliente_ID)', 'avg(nProducto_ID)', 'id']
    train = train['Producto_ID', 'Cliente_ID', 'Ruta_SAK', 'Agencia_ID', 'Semana', 'Canal_ID',
    'Lag1', 'avg(nAgencia)', 'avg(nRuta_SAK)', 'avg(nCliente_ID)', 'avg(nProducto_ID)', 'Target']

    train = train.withColumn('Target', log(lit(col('Target') + 1)))

    del data
    gc.collect()

    train = train.withColumn('validationIndicatorCol', rand(1) > 0.99)
    featuresCols = train.columns
    featuresCols.remove('Target')
    featuresCols.remove('validationIndicatorCol')
    vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol='features')


    xlf = SparkXGBRegressor(
        validation_indicator_col='validationIndicatorCol',
        label_col='Target',
        learning_rate=0.1,
        objective='reg:squarederror',
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=0.85,
        colsample_bytree=0.7,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        seed=1440,
        missing=0.0,
        num_workers=2,
    )
    paramGrid = ParamGridBuilder() \
        .addGrid(xlf.max_depth, [10]) \
        .addGrid(xlf.n_estimators, [10]) \
        .build()

    reg_eval = RegressionEvaluator(metricName='rmse',
                                   labelCol=xlf.getLabelCol(),
                                   predictionCol=xlf.getPredictionCol())

    cv = CrossValidator(estimator=xlf, evaluator=reg_eval, estimatorParamMaps=paramGrid)
    pipeline = Pipeline(stages=[vectorAssembler, cv])
    pipelineModel = pipeline.fit(train)

    pred10 = predict.filter(col('Semana')==10)
    pred10 = pipelineModel.transform(pred10)
    pred10 = pred10.withColumn('prediction',lit(exp(col('prediction'))-1))

    pred10_lag = pred10[['Cliente_ID','Producto_ID','prediction']]
    pred10_lag = pred10_lag.groupBy(['Cliente_ID', 'Producto_ID']).mean("prediction")
    pred10_lag = pred10_lag.withColumnRenamed('avg(prediction)', 'Lag1')


    pred11 = predict.filter(col('Semana') == 11)
    pred11 = pred11.drop('Lag1')
    pred11 = pred11.join(pred10_lag,
                     on=['Cliente_ID', 'Producto_ID'],
                     how='left')

    pred11 = pred11.fillna(0)
    pred11 =pipelineModel.transform(pred11)
    pred11 = pred11.withColumn('prediction', lit(exp(col('prediction')) - 1))

    predict_test = pred10['id','prediction'].unionByName(pred11['id','prediction'], allowMissingColumns=True)
    pipeline.write().overwrite().save('pipeline' + x)
    pipelineModel.write().overwrite().save('pipelineModel' + x)

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
data.coalesce(1).write.options(header='True', delimiter = ',').mode('overwrite').csv('results2')