# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def mse(y_true, y_pred):
    result = mean_squared_error(y_true, y_pred)
    return result
    
def mae(y_true, y_pred):
    result = mean_absolute_error(y_true, y_pred)
    return result
    
def mape(y_true, y_pred):
    result = mean_absolute_percentage_error(y_true, y_pred)
    return result
    
def rmse(y_true, y_pred):
    result = np.sqrt(mean_squared_error(y_true, y_pred))
    return result

def f1(y_true, y_pred, optimization=False):
    if isinstance(y_true, pd.DataFrame):
        y_true = np.array(y_true)
        
    if isinstance(y_pred, pd.DataFrame):
        y_pred = np.array(y_pred)
        
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
    
    average = None
    if len(y_pred.shape) == 1:
        average = 'binary'
        y_pred = (y_pred > 0.5).astype(int)
    else:
        average = 'micro'
        y_pred = np.argmax(y_pred, axis=1)
        
    result = f1_score(y_true, y_pred, average=average)
        
    return result

class Metrics:
    def __init__(self, metric):
        self.metric = metric

        if self.metric == 'f1':
            self.metric = f1
        elif self.metric == 'mae':
            self.metric = mae
        elif self.metric == 'rmse':
            self.metric = rmse
        elif self.metric == 'mape':
            self.metric = mape
        else:
            pass
            
    def __call__(self, y_true, y_pred):
        return self.metric(y_true, y_pred)
    
def call_metric(metric):
    method = None
    try:
        method = Metrics(metric)
    except Exception as e:
        print(f'{str(e)}')
        
    return method
