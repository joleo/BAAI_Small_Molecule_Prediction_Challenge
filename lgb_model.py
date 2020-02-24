# -*- coding: utf-8 -*-
"""
@Time ： 2020/2/22 11:42
@Auth ： joleo
@File ：lgb_model.py
"""
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cab
import re
import gc
import pickle
import random
import keras
import numpy as np
import pandas as pd

def smape(y_true, y_pred):
    return  2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

train = pd.read_pickle('data/train.pkl')
test = pd.read_pickle('data/test.pkl')

drop_features =[]
bin_features = []
mul_features = []
num_features = []
others_features = []
for col in train.columns:
    val_len = len(list(set(train[col])))
    if val_len <=1:
        drop_features.append(col)
    elif val_len == 2:
        bin_features.append(col)
    elif (val_len > 2 & val_len <100):
        if train[col].dtypes == np.int8:
            mul_features.append(col)
        elif train[col].dtypes == np.float16:
            num_features.append(col)
        else: others_features.append(col)
    else: others_features.append(col)

cat_features = bin_features + mul_features
nimcial_features = num_features

labels = ['p1','p2','p3','p4','p5','p6']
features = [x for x in train.columns if x not in labels+drop_features+['id']]

X_train = train[features].values
X_test = test[features].values
y_train = train[labels].values
print(X_train.shape,y_train.shape,X_test.shape)

model = 'lgb'
if model == 'lgb': # LGB
    single_model = lgb.LGBMRegressor(objective='regression', num_leaves=31,learning_rate=0.05, n_estimators=1000,random_state=88,n_jobs=-1)
    model = MultiOutputRegressor(single_model)
    # fit_param = {'verbose': False, 'early_stopping_rounds':50}#, 'eval_set':(eval_X, eval_y)}
    model.fit(X_train, y_train)#, fit_param=fit_param)
    y_pre_trn = model.predict(X_train)
    print(smape(y_train, y_pre_trn))
elif model == 'cat': # catboost
    clf=cab.CatBoostRegressor(iterations=750
                          ,learning_rate=0.05
                          ,depth=5
                          ,loss_function='RMSE'
                          ,silent=True
                          ,gpu_cat_features_storage=-1)
    model = MultiOutputRegressor(clf)
    model.fit(X_train,y_train)
    y_pre_trn = model.predict(X_train)
    print(smape(y_train, y_pre_trn))

all_predictions = model.predict(X_test)
path = 'data/molecule_open_data/'
targets = ['p1','p2','p3','p4','p5','p6']
submission = pd.read_csv(path+'sample_submission.csv')
print('Sample_submission.csv file have {} rows and {} columns'.format(submission.shape[0], submission.shape[1]))
sub = test[['id','3176']].merge(submission,how='left',on='id')
sub[targets] = all_predictions
sub[['id']+targets].to_csv("data/submission.csv", index = False)
