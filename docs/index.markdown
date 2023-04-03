---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: I created a XGBoost model using Pyspark
---
Main project repo: https://github.com/morintori/inventory-management-xgboost-grupobimbo <br/>
Dash app repo: https://github.com/morintori/Grupo-App <br/>
Dash app: https://grupo.shannonc.ca/ <br/>
For the data I used this competition from Kaggle https://www.kaggle.com/competitions/grupo-bimbo-inventory-demand/overview.
I took the results and created a dashboard using dash, check it out in the link above.
The goal of this competition is to predict the weekly demand for each product, sold through 45,000 routes for the Mexican Bakery, Grupo Bimbo.
Because bakery items tend to have a short shelf life, the demand for each product has to be accurately determined as we do not want clients to be facing
empty shelves and for products to be sent back because they have been expired.
The data is structured and tabulated, a neural network was considered, but based on the format that the data is in a decision tree algorithm is superior.
One such algorithm is XGBoost. Based on the discussions and code that was submitted by other teams I decided to use this as it fits all of my criteria
and is also easy to use.

Because I am limited by my RAM I decided to run my code using Pyspark and used the XGBoost for spark module that comes with XGBoost. This was a fairly new
feature so I had to learn with limited resources for my code to run. I used Python because it is the language that I currently have the most familiarity with.

```python
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
```
Firstly, an important feature for predicting the current weeks demand is to compare it to the demand of the precious week. This is what the code snippet above does. The demand variable (renamed in my code as "Target") is based on taking the sales of that week and decreasing it by the return. The sales is what is provided to the end store, and the return is the unsold inventory that the store returned. So if in a week that end store returned more items, it means it had some unsold product. Therefore that would be a signal for the next week that the store would not need as many of that particular product.

```python
    nAgencia = data.filter(col('target').isNotNull()).groupBy(['Agencia_ID', 'Semana']).count()
    nAgencia = nAgencia.withColumnRenamed('count', 'nAgencia')
    nAgencia = nAgencia.groupBy(['Agencia_ID']).mean('nAgencia')


    data = data.join(nAgencia,
                     on=['Agencia_ID'],
                     how='left')
```
Second, I made frequency variables for the number of Sales Depot (Agencia), Sales Routes (Ruta_SAK), Clients and Products. For example, the Sales Depot frequency variable gives the number of times that that Sales Depot had deliveries sent to them in a given week.
```python
+----------+-------------+
|Agencia_ID|avg(nAgencia)|
+----------+-------------+
|      3226|       3724.0|
|      3213|      57741.0|
|      1143|       7725.0|
|      1223|      60862.0|
|      1160|          5.0|
|      1212|      48429.0|
|      1259|      16857.0|
|      1156|       2415.0|
|      3211|      37313.0|
|      1215|      47036.0|
|      1114|       6723.0|
|      2647|       2855.0|
|      1218|      33177.0|
|      1275|       6313.0|
|      1113|      30922.0|
|      1146|       8547.0|
|      1250|        651.0|
|      1222|      71356.0|
|      1255|       1618.0|
|      1252|        662.0|
+----------+-------------+
```
I then took my data set after adding these features, separated them into train and test, randomly assigned validation data in the train set, took the columns that I am going to use as features, and then applied vectorization to the categorical features.
```python
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
   ```
And now for the juicy part, the model selection, parameter setting, appling crossvalidation and creating a pipeline model.

```python
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
```
Training...
```python
[0]	training-rmse:1.33813	validation-rmse:1.32777
[1]	training-rmse:1.22594	validation-rmse:1.21626
[2]	training-rmse:1.12678	validation-rmse:1.11776
[3]	training-rmse:1.04131	validation-rmse:1.03241
[4]	training-rmse:0.96461	validation-rmse:0.95615
[5]	training-rmse:0.90059	validation-rmse:0.89287
[6]	training-rmse:0.84064	validation-rmse:0.83346
[7]	training-rmse:0.79198	validation-rmse:0.78553
[8]	training-rmse:0.75377	validation-rmse:0.74766
[9]	training-rmse:0.71272	validation-rmse:0.70761
```
Now, because I need the lagged values from week 10 for prediction week 11 demand, I first predicted week 10 demand to create the week 11 data for prediction.

```python
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
```

I combined week 10 and 11 data, then saved the model. One limitation to how I created and made this model is that I am running it locally and I simply do not have the RAM to run my script in full. If I had access to an adequete machine or a cloud computing instance I could run it in full. However for my script I am separating my model into training by state instead of the entire country. The 'x' in my code is where the state name lies.
```python
    predict_test = pred10['id','prediction'].unionByName(pred11['id','prediction'], allowMissingColumns=True)
    pipeline.write().overwrite().save('pipeline' + x)
    pipelineModel.write().overwrite().save('pipelineModel' + x)

    predict_test.show()
    predict_test.write.save('predict_test' + x + '.parquet')
    del predict_test
    gc.collect()
```
I then took all the states and combined into one csv for submitting to the Kaggle Competition.

```python
for x in state_list:


    prdf = spark.read.parquet('predict_test' + x + '.parquet')
    data = data.unionByName(prdf, allowMissingColumns=True)


data = data.withColumn('prediction', round('prediction',1))
data = data.withColumnRenamed('prediction','Demanda_uni_equil')
data.show()
data.coalesce(1).write.options(header='True', delimiter = ',').mode('overwrite').csv('results2')
```
My score does not crack the leaderboard, but I know where I can make improvements. If I had an instance I could run the dataset through my model in its entirety. Furthermore I could use this model as the first layer and run another model and take the weighted average of the two. I could also cluster products into cluster and use that as another feature. I tried to do that but some of the clusters I got were not useful, the products are not easily separated into this method because a spanish NLP may not easily separate different types of products, these products have names such as "Gansito" (little goose) which would be hard for a nlp to separate if it does not have the context. 

Thank you for reading and viewing my XGBoost model for inventory demand. Feel free to clone a copy and run my script. You will need to download the dataset from the kaggle link above. I also have the pipeline models in the repository, create_predictions_with_plmodels.py can be run with the pretrained models to create predictions.




   
