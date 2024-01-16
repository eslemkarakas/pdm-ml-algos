# import standard packages
import os
import sklearn
import numpy as np
from sklearn.model_selection import GridSearchCV

class GridSearchAlgorithm():
    tuning_algorithm_name = 'grid-search'
    
    def __init__(self, model, vector_space):
        self.model = GridSearchCV(estimator=model, param_grid=vector_space, cv=3,   )
    
    def fit_tuning(self, X_train, y_train, X_validation=None, y_validation=None, sample_weight=None,):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        
    def find_best_params(self):
        self.best_params = self.model.best_params_
        return self.best_params
    
    def set_best_params(self):
        self.model = self.model.set_params(**self.best_params)
        return self.model
