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
The data is structured and tabulated, a neural network was considered, but based on the format that the data is in a decision tree algorithm is superior
One such algorithm is XGBoost. Based on the discussions and code that was submitted by other teams I decided to use this as it fits all of my criteria
and is also easy to use.

Because I am limited by my RAM I decided to run my code using Pyspark and used the XGBoost for spark module that comes with XGBoost. This was a fairly new
feature so I had to learn with limited resources for my code to run. I used Python because it is the language that I currently have the most familiarity with.

```python
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
```
