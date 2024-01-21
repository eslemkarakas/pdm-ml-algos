# import standard packages
import os
import sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# import local packages
from utils.tuning import GridSearchAlgorithm
from utils.template import BaseAlgorithm

class GradientBoostingClassifierAlgorithm(BaseAlgorithm):
    algorithm_name = "Gradient Boost" 
    
    def __init__(self, params):
        super(GradientBoostingClassifierAlgorithm, self).__init__(params)
        self.library_version = sklearn.__version__
        self.model = GradientBoostingClassifier(n_estimators=params.get('n_estimator', 10),
                                                learning_rate=params.get('learning_rate', 0.1),
                                                max_depth=params.get('max_depth', 3),
                                                random_state=params.get('seed', 1),)
    
    def interpret(self, X_train, y_train, X_validation, y_validation, model_file_path, learner_name, target_name=None, class_names=None, metric_name=None, ml_task=None, explain_level=2,):
        super(GradientBoostingClassifierAlgorithm, self).interpret(X_train, y_train, X_validation, y_validation, model_file_path, learner_name, target_name=None, class_names=None, metric_name=None, ml_task=None, explain_level=2,)

        # if explanation level is zero, no further interpretation or visualization is needed
        if explain_level == 0: return 0
        
        try:
            pass
        except Exception as e:
            pass   
            
class GradientBoostingRegressorAlgorithm(BaseAlgorithm):
    pass