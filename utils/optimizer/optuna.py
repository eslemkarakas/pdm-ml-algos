# -*- coding: utf-8 -*-
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from metrics.measure import call_metric

class Optimizer:
    def __init__(self, X, y, metric_option):
        self.X = X
        self.y = y
        self.metric = call_metric(metric_option)
        self.optimize_hyperparameters()
        
    def objective(self, trial):
        hp = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 1, 10),
                'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
            }

        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size=0.2, shuffle=True, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        model = xgb.train(hp, dtrain, evals=[(dvalid, 'validation')], early_stopping_rounds=10, verbose_eval=False)

        y_pred = model.predict(dvalid)

        metric_result = self.metric(y_valid, y_pred)

        return metric_result

    def optimize_hyperparameters(self):
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=100)

    def get_best_params(self):
        return self.study.best_params

def call_optimizer(X, y, metric):
    optimizer = None
    try:
        optimizer = Optimizer(X, y, metric)
    except Exception as e:
        print(f'{str(e)}')
        
    return optimizer
