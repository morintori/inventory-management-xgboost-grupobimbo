---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
# I created a XGBoost model using Pyspark
For the data I used this competition from Kaggle https://www.kaggle.com/competitions/grupo-bimbo-inventory-demand/overview.
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
/home/morin/miniconda3/envs/rapids-23.02/bin/python /home/morin/Documents/grupo/Model files/Model files/grupo pyspark.py 
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
23/03/31 16:03:46 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
+------+----------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+
|Semana|Agencia_ID|Canal_ID|Ruta_SAK|Cliente_ID|Producto_ID|Venta_uni_hoy|Venta_hoy|Dev_uni_proxima|Dev_proxima|Demanda_uni_equil|
+------+----------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+
|     3|      1110|       7|    3301|     15766|       1212|            3|    25.14|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|       1216|            4|    33.52|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|       1238|            4|    39.32|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|       1240|            4|    33.52|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|       1242|            3|    22.92|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|       1250|            5|     38.2|              0|        0.0|                5|
|     3|      1110|       7|    3301|     15766|       1309|            3|    20.28|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|       3894|            6|     56.1|              0|        0.0|                6|
|     3|      1110|       7|    3301|     15766|       4085|            4|     24.6|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|       5310|            6|    31.68|              0|        0.0|                6|
|     3|      1110|       7|    3301|     15766|      30531|            8|    62.24|              0|        0.0|                8|
|     3|      1110|       7|    3301|     15766|      30548|            4|    21.52|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|      30571|           12|     75.0|              0|        0.0|               12|
|     3|      1110|       7|    3301|     15766|      31309|            7|    43.75|              0|        0.0|                7|
|     3|      1110|       7|    3301|     15766|      31506|           10|     62.5|              0|        0.0|               10|
|     3|      1110|       7|    3301|     15766|      32393|            5|     15.1|              0|        0.0|                5|
|     3|      1110|       7|    3301|     15766|      32933|            3|    21.12|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|      32936|            3|    21.12|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|      34053|            8|     36.0|              0|        0.0|                8|
|     3|      1110|       7|    3301|     15766|      35651|           12|     90.0|              0|        0.0|               12|
+------+----------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+
only showing top 20 rows

+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+------------------+------------+
|Agencia_ID|Semana|Canal_ID|Ruta_SAK|Cliente_ID|Producto_ID|Venta_uni_hoy|Venta_hoy|Dev_uni_proxima|Dev_proxima|Demanda_uni_equil|              Town|       State|
+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+------------------+------------+
|      1110|     8|       7|    3301|     15766|       1212|            4|    33.52|              0|        0.0|                4|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|       1216|            5|     41.9|              0|        0.0|                5|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|       1220|            1|     7.64|              0|        0.0|                1|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|       1238|            3|    29.49|              0|        0.0|                3|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|       1240|            2|    16.76|              0|        0.0|                2|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|       1242|            1|     7.64|              0|        0.0|                1|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|       1250|            8|    61.12|              0|        0.0|                8|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|       1309|            3|    20.28|              0|        0.0|                3|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|       3894|            2|     18.7|              0|        0.0|                2|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      30531|           17|   132.26|              0|        0.0|               17|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      30548|            3|    16.14|              0|        0.0|                3|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      31309|            8|     50.0|              0|        0.0|                8|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      31506|            9|    56.25|              0|        0.0|                9|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      31688|            4|     26.0|              0|        0.0|                4|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      32393|            2|     6.04|              0|        0.0|                2|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      32819|            4|    35.56|              0|        0.0|                4|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      32936|            2|    14.08|              0|        0.0|                2|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      35303|           15|    93.75|              0|        0.0|               15|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      35452|            6|    26.64|              0|        0.0|                6|2008 AG. LAGO FILT|MÉXICO, D.F.|
|      1110|     8|       7|    3301|     15766|      35455|            5|     22.7|              0|        0.0|                5|2008 AG. LAGO FILT|MÉXICO, D.F.|
+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+------------------+------------+
only showing top 20 rows

+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+------------------+------------+------+---+----+
|Agencia_ID|Semana|Canal_ID|Ruta_SAK|Cliente_ID|Producto_ID|Venta_uni_hoy|Venta_hoy|Dev_uni_proxima|Dev_proxima|              Town|       State|Target|tst|  id|
+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+------------------+------------+------+---+----+
|      1110|     8|       7|    3301|     15766|       1212|            4|    33.52|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     4|  0|null|
|      1110|     8|       7|    3301|     15766|       1216|            5|     41.9|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     5|  0|null|
|      1110|     8|       7|    3301|     15766|       1220|            1|     7.64|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     1|  0|null|
|      1110|     8|       7|    3301|     15766|       1238|            3|    29.49|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     3|  0|null|
|      1110|     8|       7|    3301|     15766|       1240|            2|    16.76|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     2|  0|null|
|      1110|     8|       7|    3301|     15766|       1242|            1|     7.64|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     1|  0|null|
|      1110|     8|       7|    3301|     15766|       1250|            8|    61.12|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     8|  0|null|
|      1110|     8|       7|    3301|     15766|       1309|            3|    20.28|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     3|  0|null|
|      1110|     8|       7|    3301|     15766|       3894|            2|     18.7|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     2|  0|null|
|      1110|     8|       7|    3301|     15766|      30531|           17|   132.26|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|    17|  0|null|
|      1110|     8|       7|    3301|     15766|      30548|            3|    16.14|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     3|  0|null|
|      1110|     8|       7|    3301|     15766|      31309|            8|     50.0|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     8|  0|null|
|      1110|     8|       7|    3301|     15766|      31506|            9|    56.25|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     9|  0|null|
|      1110|     8|       7|    3301|     15766|      31688|            4|     26.0|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     4|  0|null|
|      1110|     8|       7|    3301|     15766|      32393|            2|     6.04|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     2|  0|null|
|      1110|     8|       7|    3301|     15766|      32819|            4|    35.56|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     4|  0|null|
|      1110|     8|       7|    3301|     15766|      32936|            2|    14.08|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     2|  0|null|
|      1110|     8|       7|    3301|     15766|      35303|           15|    93.75|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|    15|  0|null|
|      1110|     8|       7|    3301|     15766|      35452|            6|    26.64|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     6|  0|null|
|      1110|     8|       7|    3301|     15766|      35455|            5|     22.7|              0|        0.0|2008 AG. LAGO FILT|MÉXICO, D.F.|     5|  0|null|
+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+------------------+------------+------+---+----+
only showing top 20 rows

Lag: Lag1
23/03/31 16:09:19 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:09:20 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:09:29 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:09:29 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
+------+----------+-----------+----------+--------+--------+-------------+---------+---------------+-----------+--------------------+------------+------+---+-------+----+
|Semana|Cliente_ID|Producto_ID|Agencia_ID|Canal_ID|Ruta_SAK|Venta_uni_hoy|Venta_hoy|Dev_uni_proxima|Dev_proxima|                Town|       State|Target|tst|     id|Lag1|
+------+----------+-----------+----------+--------+--------+-------------+---------+---------------+-----------+--------------------+------------+------+---+-------+----+
|     9|     15766|       5310|      1110|       7|    3301|            6|    31.68|              0|        0.0|  2008 AG. LAGO FILT|MÉXICO, D.F.|     6|  0|   null|null|
|     9|     15766|      31309|      1110|       7|    3301|           11|    68.75|              0|        0.0|  2008 AG. LAGO FILT|MÉXICO, D.F.|    11|  0|   null| 8.0|
|     9|     15766|      32393|      1110|       7|    3301|            5|     15.1|              0|        0.0|  2008 AG. LAGO FILT|MÉXICO, D.F.|     5|  0|   null| 2.0|
|     9|     15766|      32936|      1110|       7|    3301|            3|    21.12|              0|        0.0|  2008 AG. LAGO FILT|MÉXICO, D.F.|     3|  0|   null| 2.0|
|     9|     46350|       1150|      2083|       4|    6601|            9|    132.3|              0|        0.0|2083 AZCAPOTZALCO...|MÉXICO, D.F.|     9|  0|   null|12.0|
|     9|   4561637|       2233|      1218|       1|    1004|            6|   119.64|              0|        0.0|     2040 AG. CENTRO|MÉXICO, D.F.|     6|  0|   null| 3.0|
|     9|   4561637|      32802|      1218|       1|    1004|            7|    83.37|              0|        0.0|     2040 AG. CENTRO|MÉXICO, D.F.|     7|  0|   null| 8.0|
|    10|     73871|       1232|      1111|       1|    1125|         null|     null|           null|       null|2002 AG. AZCAPOTZ...|MÉXICO, D.F.|  null|  1|3573412| 2.0|
|    10|    111354|       1146|      1217|       1|    1045|         null|     null|           null|       null|    2050 AG. MIXCOAC|MÉXICO, D.F.|  null|  1| 914153| 6.0|
|    10|    114659|       1064|      1116|       1|    1176|         null|     null|           null|       null|2011 AG. SAN ANTONIO|MÉXICO, D.F.|  null|  1|    114| 1.0|
|    10|    390858|      43067|      1223|       1|    4453|         null|     null|           null|       null|2070 AG. XOCHIMIL...|MÉXICO, D.F.|  null|  1|5344504|null|
|    10|    902445|      35141|      1124|       1|    1634|         null|     null|           null|       null|2021 AG. XOCHIMIL...|MÉXICO, D.F.|  null|  1|4458937| 4.0|
|    10|   1179641|       1250|      1120|       1|    1476|         null|     null|           null|       null|2018 AG. TEPALCAT...|MÉXICO, D.F.|  null|  1|4458988| 5.0|
|    10|   1345908|       1109|      3217|       1|    1018|         null|     null|           null|       null|2032 AG. SANTA LUCIA|MÉXICO, D.F.|  null|  1|    135|null|
|    10|   1637506|      37361|      1138|       1|    2104|         null|     null|           null|       null| 2015 AG. ROJO GOMEZ|MÉXICO, D.F.|  null|  1|3573363|14.0|
|    10|   4391535|      43285|      2647|      11|    3958|         null|     null|           null|       null|2647 BLM_AG. CAMP...|MÉXICO, D.F.|  null|  1|    172|20.0|
|    11|    167481|      32940|      1223|       1|    2153|         null|     null|           null|       null|2070 AG. XOCHIMIL...|MÉXICO, D.F.|  null|  1|     96|null|
|    11|    184165|      36747|      1116|       1|    2137|         null|     null|           null|       null|2011 AG. SAN ANTONIO|MÉXICO, D.F.|  null|  1| 914188|null|
|    11|    325074|       1150|      3211|       1|    1016|         null|     null|           null|       null|2029 AG.IZTAPALAPA 2|MÉXICO, D.F.|  null|  1|     17|null|
|    11|    587728|       1250|      1118|       1|    1427|         null|     null|           null|       null|   2007 AG. LA VILLA|MÉXICO, D.F.|  null|  1|2687810|null|
+------+----------+-----------+----------+--------+--------+-------------+---------+---------------+-----------+--------------------+------------+------+---+-------+----+
only showing top 20 rows

23/03/31 16:11:39 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
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
only showing top 20 rows

+--------+--------------+
|Ruta_SAK|avg(nRuta_SAK)|
+--------+--------------+
|    2142|         949.0|
|    6620|        1236.0|
|    2122|        3300.0|
|    2866|         170.0|
|    3226|         849.0|
|    1127|         528.0|
|    2811|         263.0|
|    1084|         506.0|
|    1025|        4406.0|
|    1460|         445.0|
|    6623|         394.0|
|    1139|         863.0|
|    3213|         512.0|
|      31|          80.0|
|    1143|         718.0|
|    2821|         258.0|
|    1618|        3628.0|
|    6622|         351.0|
|      85|         118.0|
|    3220|         301.0|
+--------+--------------+
only showing top 20 rows

+----------+----------------+
|Cliente_ID|avg(nCliente_ID)|
+----------+----------------+
|   4387549|            17.0|
|   1447808|             4.0|
|     45307|             4.0|
|   4187373|            22.0|
|    924545|            20.0|
|   1013559|            12.0|
|   2455630|             6.0|
|   2479123|             2.0|
|   1381697|             4.0|
|    166624|             8.0|
|     28836|            16.0|
|     75039|            77.0|
|   4492619|             4.0|
|   4243945|            20.0|
|     82672|            58.0|
|   1258422|            20.0|
|    706872|            23.0|
|     65867|            22.0|
|    195291|            22.0|
|   4736712|            14.0|
+----------+----------------+
only showing top 20 rows

23/03/31 16:22:39 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:22:40 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:22:45 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:22:46 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
[16:26:42] task 1 got new rank 0
[16:26:42] task 0 got new rank 1
[0]	training-rmse:1.33811	validation-rmse:1.33003
[1]	training-rmse:1.22607	validation-rmse:1.21808
[2]	training-rmse:1.12702	validation-rmse:1.11901
[3]	training-rmse:1.04135	validation-rmse:1.03353
[4]	training-rmse:0.96465	validation-rmse:0.95676
[5]	training-rmse:0.90079	validation-rmse:0.89365
[6]	training-rmse:0.84103	validation-rmse:0.83399
[7]	training-rmse:0.79254	validation-rmse:0.78612
[8]	training-rmse:0.75444	validation-rmse:0.74849
[9]	training-rmse:0.71358	validation-rmse:0.70802
/home/morin/miniconda3/envs/rapids-23.02/lib/python3.10/site-packages/xgboost/sklearn.py:808: UserWarning: Loading a native XGBoost model with Scikit-Learn interface.
  warnings.warn("Loading a native XGBoost model with Scikit-Learn interface.")
23/03/31 16:32:41 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:32:48 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:32:51 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:32:52 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:32:55 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:33:02 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:37:35 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:37:35 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:37:40 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:37:41 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
[16:44:01] task 1 got new rank 0
[16:44:01] task 0 got new rank 1
[0]	training-rmse:1.33950	validation-rmse:1.34816
[1]	training-rmse:1.22737	validation-rmse:1.23415
[2]	training-rmse:1.12806	validation-rmse:1.13325
[3]	training-rmse:1.04252	validation-rmse:1.04682
[4]	training-rmse:0.96572	validation-rmse:0.96875
[5]	training-rmse:0.90168	validation-rmse:0.90435
[6]	training-rmse:0.84172	validation-rmse:0.84349
[7]	training-rmse:0.79307	validation-rmse:0.79469
[8]	training-rmse:0.75481	validation-rmse:0.75665
[9]	training-rmse:0.71389	validation-rmse:0.71489
23/03/31 16:49:21 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:49:28 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:49:28 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:49:31 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:49:31 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:49:37 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:57:48 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:57:49 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:57:52 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 16:57:52 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
[17:00:50] task 1 got new rank 0
[17:00:50] task 0 got new rank 1
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
23/03/31 17:03:17 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:03:23 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:03:23 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:03:27 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:03:27 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:03:33 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:10:52 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:10:53 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:10:58 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:10:58 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:16:30 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:16:38 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:16:39 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:16:39 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:16:40 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:16:40 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:16:40 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:18:04 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:18:04 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:18:09 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:18:09 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:23:50 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:23:51 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:23:51 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:23:51 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:23:53 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:23:53 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:23:53 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:23:53 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:23:59 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:24:00 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:24:00 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:24:00 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:24:00 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:24:00 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:24:01 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:24:01 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
[17:24:16] task 1 got new rank 0
[17:24:16] task 0 got new rank 1
[0]	training-rmse:1.33855	validation-rmse:1.33767
[1]	training-rmse:1.22640	validation-rmse:1.22474
[2]	training-rmse:1.12725	validation-rmse:1.12520
[3]	training-rmse:1.04146	validation-rmse:1.03964
[4]	training-rmse:0.96475	validation-rmse:0.96255
[5]	training-rmse:0.90089	validation-rmse:0.89880
[6]	training-rmse:0.84101	validation-rmse:0.83849
[7]	training-rmse:0.79261	validation-rmse:0.79011
[8]	training-rmse:0.75444	validation-rmse:0.75204
[9]	training-rmse:0.71362	validation-rmse:0.71096
23/03/31 17:25:22 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:25:23 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:25:26 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:25:26 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:20 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:32 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:32 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:32 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:32 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:32 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:32 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:32 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:32 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:36 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:36 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:36 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:36 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:36 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:36 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:36 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:36 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:31:45 WARN DAGScheduler: Broadcasting large task binary with size 1404.6 KiB
+-------+------------------+
|     id|        prediction|
+-------+------------------+
|6305862| 2.190861923344928|
|4165573| 2.310697239155029|
|1330769|3.5212924081822345|
|3357903|2.3008708574060526|
| 981133|2.8170009478528426|
|4462543|3.4267709876971457|
|2704237| 3.514107051911922|
|3814037|3.5849506647665947|
|3325994| 3.347872941723545|
|3879922| 5.244410447219792|
|4353906| 3.774259384481507|
|5247509| 4.499359148534456|
|6809909|13.290152932797756|
| 966631| 8.560507759994929|
|2626971|11.629948056794651|
|6914211| 3.693432330597494|
|3566557|2.8772685000167746|
|3086773| 6.896838715445084|
|4930554| 4.366376390143966|
| 103627| 7.952349295166995|
+-------+------------------+
only showing top 20 rows

23/03/31 17:32:30 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:32:31 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:32:33 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:32:33 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:36 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:46 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:46 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:46 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:46 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:46 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:46 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:47 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:48 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:50 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:50 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:50 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:50 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:50 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:50 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:50 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:38:51 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:39:01 WARN DAGScheduler: Broadcasting large task binary with size 1606.9 KiB
23/03/31 17:39:07 WARN MemoryManager: Total allocation exceeds 95.00% (1,020,054,720 bytes) of heap memory
Scaling row group sizes to 95.00% for 8 writers
+------+----------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+
|Semana|Agencia_ID|Canal_ID|Ruta_SAK|Cliente_ID|Producto_ID|Venta_uni_hoy|Venta_hoy|Dev_uni_proxima|Dev_proxima|Demanda_uni_equil|
+------+----------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+
|     3|      1110|       7|    3301|     15766|       1212|            3|    25.14|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|       1216|            4|    33.52|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|       1238|            4|    39.32|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|       1240|            4|    33.52|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|       1242|            3|    22.92|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|       1250|            5|     38.2|              0|        0.0|                5|
|     3|      1110|       7|    3301|     15766|       1309|            3|    20.28|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|       3894|            6|     56.1|              0|        0.0|                6|
|     3|      1110|       7|    3301|     15766|       4085|            4|     24.6|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|       5310|            6|    31.68|              0|        0.0|                6|
|     3|      1110|       7|    3301|     15766|      30531|            8|    62.24|              0|        0.0|                8|
|     3|      1110|       7|    3301|     15766|      30548|            4|    21.52|              0|        0.0|                4|
|     3|      1110|       7|    3301|     15766|      30571|           12|     75.0|              0|        0.0|               12|
|     3|      1110|       7|    3301|     15766|      31309|            7|    43.75|              0|        0.0|                7|
|     3|      1110|       7|    3301|     15766|      31506|           10|     62.5|              0|        0.0|               10|
|     3|      1110|       7|    3301|     15766|      32393|            5|     15.1|              0|        0.0|                5|
|     3|      1110|       7|    3301|     15766|      32933|            3|    21.12|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|      32936|            3|    21.12|              0|        0.0|                3|
|     3|      1110|       7|    3301|     15766|      34053|            8|     36.0|              0|        0.0|                8|
|     3|      1110|       7|    3301|     15766|      35651|           12|     90.0|              0|        0.0|               12|
+------+----------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+
only showing top 20 rows

+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+-------------------+----------------+
|Agencia_ID|Semana|Canal_ID|Ruta_SAK|Cliente_ID|Producto_ID|Venta_uni_hoy|Venta_hoy|Dev_uni_proxima|Dev_proxima|Demanda_uni_equil|               Town|           State|
+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+-------------------+----------------+
|      1112|     8|       1|    1001|    327267|        693|            3|     28.8|              0|        0.0|                3|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327267|       1109|            1|    15.01|              0|        0.0|                1|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327267|       1125|            2|     19.2|              0|        0.0|                2|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327267|       1129|            4|     70.4|              0|        0.0|                4|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327267|       1146|            3|    64.17|              0|        0.0|                3|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327267|       1150|            1|    13.96|              0|        0.0|                1|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327267|       2233|            5|     99.7|              0|        0.0|                5|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327267|       3631|            1|    16.35|              0|        0.0|                1|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327267|      32802|            2|    23.82|              0|        0.0|                2|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327267|      48077|            3|    42.78|              0|        0.0|                3|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|        693|            2|     19.2|              0|        0.0|                2|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|       1109|            7|   105.07|              0|        0.0|                7|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|       1125|           12|    115.2|              0|        0.0|               12|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|       1129|            5|     88.0|              0|        0.0|                5|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|       1146|            5|   106.95|              0|        0.0|                5|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|       1150|            4|    55.84|              0|        0.0|                4|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|       1160|            1|    18.86|              0|        0.0|                1|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|       2233|            4|    79.76|              0|        0.0|                4|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|       2665|            2|     32.0|              0|        0.0|                2|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
|      1112|     8|       1|    1001|    327344|       3631|            2|     32.7|              0|        0.0|                2|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|
+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-----------------+-------------------+----------------+
only showing top 20 rows

+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-------------------+----------------+------+---+----+
|Agencia_ID|Semana|Canal_ID|Ruta_SAK|Cliente_ID|Producto_ID|Venta_uni_hoy|Venta_hoy|Dev_uni_proxima|Dev_proxima|               Town|           State|Target|tst|  id|
+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-------------------+----------------+------+---+----+
|      1112|     8|       1|    1001|    327267|        693|            3|     28.8|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     3|  0|null|
|      1112|     8|       1|    1001|    327267|       1109|            1|    15.01|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     1|  0|null|
|      1112|     8|       1|    1001|    327267|       1125|            2|     19.2|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     2|  0|null|
|      1112|     8|       1|    1001|    327267|       1129|            4|     70.4|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     4|  0|null|
|      1112|     8|       1|    1001|    327267|       1146|            3|    64.17|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     3|  0|null|
|      1112|     8|       1|    1001|    327267|       1150|            1|    13.96|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     1|  0|null|
|      1112|     8|       1|    1001|    327267|       2233|            5|     99.7|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     5|  0|null|
|      1112|     8|       1|    1001|    327267|       3631|            1|    16.35|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     1|  0|null|
|      1112|     8|       1|    1001|    327267|      32802|            2|    23.82|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     2|  0|null|
|      1112|     8|       1|    1001|    327267|      48077|            3|    42.78|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     3|  0|null|
|      1112|     8|       1|    1001|    327344|        693|            2|     19.2|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     2|  0|null|
|      1112|     8|       1|    1001|    327344|       1109|            7|   105.07|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     7|  0|null|
|      1112|     8|       1|    1001|    327344|       1125|           12|    115.2|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|    12|  0|null|
|      1112|     8|       1|    1001|    327344|       1129|            5|     88.0|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     5|  0|null|
|      1112|     8|       1|    1001|    327344|       1146|            5|   106.95|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     5|  0|null|
|      1112|     8|       1|    1001|    327344|       1150|            4|    55.84|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     4|  0|null|
|      1112|     8|       1|    1001|    327344|       1160|            1|    18.86|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     1|  0|null|
|      1112|     8|       1|    1001|    327344|       2233|            4|    79.76|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     4|  0|null|
|      1112|     8|       1|    1001|    327344|       2665|            2|     32.0|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     2|  0|null|
|      1112|     8|       1|    1001|    327344|       3631|            2|     32.7|              0|        0.0|2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     2|  0|null|
+----------+------+--------+--------+----------+-----------+-------------+---------+---------------+-----------+-------------------+----------------+------+---+----+
only showing top 20 rows

Lag: Lag1
23/03/31 17:43:15 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:43:15 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:43:22 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:43:22 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:43:31 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
23/03/31 17:43:31 WARN RowBasedKeyValueBatch: Calling spill() on RowBasedKeyValueBatch. Will not spill but return 0.
+------+----------+-----------+----------+--------+--------+-------------+---------+---------------+-----------+--------------------+----------------+------+---+-------+----+
|Semana|Cliente_ID|Producto_ID|Agencia_ID|Canal_ID|Ruta_SAK|Venta_uni_hoy|Venta_hoy|Dev_uni_proxima|Dev_proxima|                Town|           State|Target|tst|     id|Lag1|
+------+----------+-----------+----------+--------+--------+-------------+---------+---------------+-----------+--------------------+----------------+------+---+-------+----+
|     9|       106|      34264|      2061|       2|    7202|           10|    196.5|              0|        0.0|2175 TOLUCA AEROP...|ESTADO DE MÉXICO|    10|  0|   null| 6.0|
|     9|     44881|       1109|      1219|       1|    1001|            2|    30.02|              0|        0.0| 2042 AG. TEPOZOTLAN|ESTADO DE MÉXICO|     2|  0|   null| 2.0|
|     9|     44881|       2233|      1219|       1|    1001|            6|   119.64|              0|        0.0| 2042 AG. TEPOZOTLAN|ESTADO DE MÉXICO|     6|  0|   null| 9.0|
|     9|    113826|        972|      2011|       4|    6601|           12|   234.72|              0|        0.0|2175 TOLUCA AEROP...|ESTADO DE MÉXICO|    12|  0|   null|null|
|     9|    113826|       1150|      2011|       4|    6601|            6|     88.2|              0|        0.0|2175 TOLUCA AEROP...|ESTADO DE MÉXICO|     6|  0|   null|12.0|
|     9|    113826|       1240|      2011|       4|    6601|           23|    193.2|              0|        0.0|2175 TOLUCA AEROP...|ESTADO DE MÉXICO|    23|  0|   null|16.0|
|     9|    113826|       1250|      2011|       4|    6601|           27|    207.9|              0|        0.0|2175 TOLUCA AEROP...|ESTADO DE MÉXICO|    27|  0|   null|19.0|
|     9|    321985|       3631|      1219|       1|    1001|            2|     32.7|              0|        0.0| 2042 AG. TEPOZOTLAN|ESTADO DE MÉXICO|     2|  0|   null| 3.0|
|     9|    321985|       4910|      1219|       1|    1001|            4|    35.68|              0|        0.0| 2042 AG. TEPOZOTLAN|ESTADO DE MÉXICO|     4|  0|   null| 4.0|
|     9|    327267|       2233|      1112|       1|    1001|            5|     99.7|              0|        0.0| 2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     5|  0|   null| 5.0|
|     9|    327267|      32802|      1112|       1|    1001|            3|    35.73|              0|        0.0| 2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     3|  0|   null| 2.0|
|     9|    327344|       1150|      1112|       1|    1001|            3|    41.88|              0|        0.0| 2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     3|  0|   null| 4.0|
|     9|    327344|       1160|      1112|       1|    1001|            1|    18.86|              0|        0.0| 2004 AG. CUAUTITLAN|ESTADO DE MÉXICO|     1|  0|   null| 1.0|
|    10|    352618|      36746|      1140|       1|    2835|         null|     null|           null|       null|    2078 AG. TEXCOCO|ESTADO DE MÉXICO|  null|  1| 914200| 1.0|
|    10|    940398|       1242|      1137|       1|    1473|         null|     null|           null|       null|       2014 AG. NEZA|ESTADO DE MÉXICO|  null|  1|1802115| 6.0|
|    10|   1275274|      43285|      1213|       1|    4481|         null|     null|           null|       null|  2041 AG. TULTITLAN|ESTADO DE MÉXICO|  null|  1|5344488|50.0|
|    10|   1499292|      43316|      2012|       1|    2010|         null|     null|           null|       null|2176 TOLUCA SAN A...|ESTADO DE MÉXICO|  null|  1|6230171| 1.0|
|    10|   1544467|       2233|      1117|       1|    1082|         null|     null|           null|       null|   2001 AG. ATIZAPAN|ESTADO DE MÉXICO|  null|  1|     58|11.0|
|    10|   4514930|       1220|      1243|       1|    4511|         null|     null|           null|       null|  2065 TOLUCA CENTRO|ESTADO DE MÉXICO|  null|  1|    104| 1.0|
|    11|     77042|       2505|      3221|       1|    1637|         null|     null|           null|       null|2013 AG. MEGA NAU...|ESTADO DE MÉXICO|  null|  1|2687768|null|
+------+----------+-----------+----------+--------+--------+-------------+---------+---------------+-----------+--------------------+----------------+------+---+-------+----+
only showing top 20 rows

+----------+-------------+
|Agencia_ID|avg(nAgencia)|
+----------+-------------+
|      1127|      57906.0|
|      1139|       6268.0|
|      1165|       4416.0|
|      1243|      36709.0|
|      2094|         82.0|
|      1276|       1277.0|
|      2015|      80357.0|
|      2013|      95079.0|
|      1176|       2526.0|
|      1170|       3392.0|
|      1122|      63400.0|
|      1172|       2276.0|
|      1137|      49514.0|
|      1155|       4006.0|
|      1130|      63999.0|
|      1171|        741.0|
|      1167|        407.0|
|      1227|      70227.0|
|      1220|      92510.0|
|      1253|       1000.0|
+----------+-------------+
only showing top 20 rows

+--------+--------------+
|Ruta_SAK|avg(nRuta_SAK)|
+--------+--------------+
|    1645|         687.0|
|    1088|         444.0|
|    3918|         148.0|
|    2142|        1155.0|
|    7240|          73.0|
|    7253|          77.0|
|    6620|         511.0|
|     148|          68.0|
|    1238|        2384.0|
|    2122|        1484.0|
|    7340|           1.0|
|    1127|        1610.0|
|    2811|        1840.0|
|    1084|        1742.0|
|    1025|        3636.0|
|    1460|        4893.0|
|    6623|         609.0|
|    1483|        2602.0|
|    1270|        1115.0|
|      31|         126.0|
+--------+--------------+
only showing top 20 rows

+----------+----------------+
|Cliente_ID|avg(nCliente_ID)|
+----------+----------------+
|   2465315|            11.0|
|    406654|            10.0|
|     78478|            29.0|
|    818517|            14.0|
|   4403768|             1.0|
|   1013559|            22.0|
|    684156|             4.0|
|    324861|            52.0|
|    679877|            37.0|
|    883701|            61.0|
|   4218309|             8.0|
|   2365749|             2.0|
|   2383439|             4.0|
|   4707629|             4.0|
|   1599628|             7.0|
|   1283607|            31.0|
|   2173690|            14.0|
|    430447|            46.0|
|   4162171|             6.0|
|   4126339|             7.0|
+----------+----------------+
only showing top 20 rows

Traceback (most recent call last):
  File "/home/morin/miniconda3/envs/rapids-23.02/lib/python3.10/multiprocessing/pool.py", line 856, in next
    item = self._items.popleft()
IndexError: pop from an empty deque

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/morin/Documents/grupo/Model files/Model files/grupo pyspark.py", line 173, in <module>
    pipelineModel = pipeline.fit(train)
  File "/home/morin/miniconda3/envs/rapids-23.02/lib/python3.10/site-packages/pyspark/ml/base.py", line 205, in fit
    return self._fit(dataset)
  File "/home/morin/miniconda3/envs/rapids-23.02/lib/python3.10/site-packages/pyspark/ml/pipeline.py", line 134, in _fit
    model = stage.fit(dataset)
  File "/home/morin/miniconda3/envs/rapids-23.02/lib/python3.10/site-packages/pyspark/ml/base.py", line 205, in fit
    return self._fit(dataset)
  File "/home/morin/miniconda3/envs/rapids-23.02/lib/python3.10/site-packages/pyspark/ml/tuning.py", line 847, in _fit
    for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
  File "/home/morin/miniconda3/envs/rapids-23.02/lib/python3.10/multiprocessing/pool.py", line 861, in next
    self._cond.wait(timeout)
  File "/home/morin/miniconda3/envs/rapids-23.02/lib/python3.10/threading.py", line 320, in wait
    waiter.acquire()
  File "/home/morin/miniconda3/envs/rapids-23.02/lib/python3.10/site-packages/pyspark/context.py", line 363, in signal_handler
    raise KeyboardInterrupt()
KeyboardInterrupt
23/03/31 17:52:01 WARN TaskSetManager: Lost task 22.0 in stage 839.0 (TID 26774) (morin-IdeaPad-L340-15IRH-Gaming executor driver): TaskKilled (Stage cancelled)
23/03/31 17:52:01 WARN TaskSetManager: Lost task 23.0 in stage 839.0 (TID 26775) (morin-IdeaPad-L340-15IRH-Gaming executor driver): TaskKilled (Stage cancelled)
23/03/31 17:52:01 WARN TaskSetManager: Lost task 20.0 in stage 839.0 (TID 26772) (morin-IdeaPad-L340-15IRH-Gaming executor driver): TaskKilled (Stage cancelled)
23/03/31 17:52:01 WARN TaskSetManager: Lost task 21.0 in stage 839.0 (TID 26773) (morin-IdeaPad-L340-15IRH-Gaming executor driver): TaskKilled (Stage cancelled)
23/03/31 17:52:01 WARN TaskSetManager: Lost task 16.0 in stage 839.0 (TID 26768) (morin-IdeaPad-L340-15IRH-Gaming executor driver): TaskKilled (Stage cancelled)
23/03/31 17:52:02 WARN TaskSetManager: Lost task 17.0 in stage 839.0 (TID 26769) (morin-IdeaPad-L340-15IRH-Gaming executor driver): TaskKilled (Stage cancelled)
23/03/31 17:52:02 WARN TaskSetManager: Lost task 18.0 in stage 839.0 (TID 26770) (morin-IdeaPad-L340-15IRH-Gaming executor driver): TaskKilled (Stage cancelled)
23/03/31 17:52:02 WARN TaskSetManager: Lost task 19.0 in stage 839.0 (TID 26771) (morin-IdeaPad-L340-15IRH-Gaming executor driver): TaskKilled (Stage cancelled)

Process finished with exit code 130 (interrupted by signal 2: SIGINT)
```
