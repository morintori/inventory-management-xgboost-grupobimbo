import pandas as pd
import numpy as np
from dask import dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from coclust.clustering import SphericalKmeans
import xgboost as xgb
import scienceplots
import dask_xgboost
import gc
product_clust = pd.read_csv('products_w_clusters.csv')
train = dd.read_csv('train.csv')
test = dd.read_csv('test.csv')
train=train[train['Semana']>7]
train['target'] = train['Demanda_uni_equil']
train = train.drop(['Demanda_uni_equil'],axis=1)
train['tst']=0
test['tst'] = 1
data = dd.concat([train,test],axis=0,copy=True)
del train
del test
gc.collect()


i= 1
lag = 'Lag' + str(i)
print('Lag:', lag)

data1 = data[['Semana', 'Cliente_ID', 'Producto_ID', 'target']]
data1 = data1.assign(Semana=data1['Semana']+1)
data1 = data1.groupby(['Semana', 'Cliente_ID', 'Producto_ID']).mean()
data1 = data1.reset_index()
data1 = data1.rename(columns={'target': lag})
data = data.merge(data1,
                how='left',
                left_on=['Semana', 'Cliente_ID', 'Producto_ID'],
                right_on=['Semana', 'Cliente_ID', 'Producto_ID'],
                left_index=False, right_index=False,
                suffixes=('_x', '_y'))
del data1
gc.collect()
# train_df = train[train['Semana']>7]
# train_df['target']=train_df['Demanda_uni_equil']
print(data.shape[0].compute())