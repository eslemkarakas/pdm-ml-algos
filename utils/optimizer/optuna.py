# import standard packages
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import local packages
from utils.metrics.measure import call_metric
from utils.metrics.validate import call_validator
from sklearn.ensemble import GradientBoostingClassifier

class HyperparameterOptimizer:
    def __init__(self, model, m_option, cv_params, cv_option, search_space, X, y):
        self.X = X
        self.y = y
        self.study = None
        self.metric = call_metric(m_option)
        self.model = model
        self.search_space = search_space
        self.validator = call_validator(cv_option, cv_params)
        self.folds = self.validator.extract(self.X, self.y)
        
    def objective(self, trial):
        hyperparameters = {}
        for param_name, param_config in self.search_space.items():
            if param_config['type'] == 'int':
                hyperparameters[param_name] = trial.suggest_int(param_name, param_config['min'], param_config['max'])
            elif param_config['type'] == 'float':
                hyperparameters[param_name] = trial.suggest_float(param_name, param_config["min"], param_config["max"], log=param_config.get("log", False))
            else:
                pass
            
        self.model = GradientBoostingClassifier(**hyperparameters)
        
        results = []
        for train_idx, val_idx in self.folds:
            X_train, y_train = self.X.loc[train_idx], self.y.loc[train_idx]
            X_val, y_val = self.X.loc[val_idx], self.y.loc[val_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            
            result = self.metric(y_val, y_pred)
            results.append(result)
        
        return np.mean(results)
    
    def optuna_find_best_params(self, n_trials=50):
        self.study = optuna.create_study(direction='minimize') # direction='maximize', sampler=optuna.samplers.RandomSampler
        self.study.optimize(self.objective, n_trials=n_trials)
        
        return self.study.best_params
    
    def optuna_optimization_visualize(self, method):
        if method == 'plot':
            optuna.visualization.plot_optimization_history(self.study)
        elif method == 'coordinate':
            optuna.visualization.plot_parallel_coordinate(self.study)
        elif method == 'plot-slice':
            pass
        else:
            pass

# define a wrapper function for optuna hyperparameter optimizer
def call_hyperparameter_optimizer(model, m_option, cv_params, cv_option, search_space, X, y):
    optimizer = None
    try:
        optimizer = HyperparameterOptimizer(model, m_option, cv_params, cv_option, search_space, X, y)
    except Exception as e:
        print(f'{str(e)}')
    return optimizer