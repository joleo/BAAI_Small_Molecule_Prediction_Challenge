# -*- coding: utf-8 -*-
"""
@Time ： 2020/2/22 11:46
@Auth ： joleo
@File ：preprocessing.py
"""
import numpy as np
import pandas as pd

def read_data():
    path = 'data/molecule_open_data/'
    print('Reading train.csv file....')
    train = pd.read_csv(path+'candidate_train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))
    print('Reading specs.csv file....')
    specs = pd.read_csv(path +'train_answer.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))
    train = train.merge(specs,how='left',on='id')
    print('Reading test.csv file....')
    test = pd.read_csv(path + 'candidate_val.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))
    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv(path+'sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, sample_submission

def _reduce_mem_usage_(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


train, test, sample_submission = read_data()
train = _reduce_mem_usage_(train)
test = _reduce_mem_usage_(test)
train.to_pickle('data/train.pkl')
test.to_pickle('data/test.pkl')