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
