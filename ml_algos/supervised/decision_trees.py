# import standard packages
import os
import sklearn
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# import local packages
from utils.template import BaseAlgorithm

class DecisionTreeClassifierAlgorithm(BaseAlgorithm):
    algorithm_name = "Decision Tree" 
    
    def __init__(self, params):
        super(DecisionTreeClassifierAlgorithm, self).__init__(params)
        self.library_version = sklearn.__version__
        self.model = DecisionTreeClassifier(criterion=params.get('criterion', 'gini'),
                                            max_depth=params.get('max_depth', 3),
                                            random_state=params.get('seed', 1),)
    
    def interpret(self, X_train, y_train, X_validation, y_validation, model_file_path, learner_name, target_name=None, class_names=None, metric_name=None, ml_task=None, explain_level=2,):
        super(DecisionTreeClassifierAlgorithm, self).interpret(X_train, y_train, X_validation, y_validation, model_file_path, learner_name, target_name=None, class_names=None, metric_name=None, ml_task=None, explain_level=2,)
        
        # if explanation level is zero, no further interpretation or visualization is needed
        if explain_level == 0: return 0
        
        try:
            pass
        except Exception as e:
            print(f'ERROR: Problem when visualizing classifier decision tree. {str(e)}')
            
class DecisionTreeRegressorAlgorithm(BaseAlgorithm):
    algorithm_name = "Decision Tree"
    
    def __init__(self, params):
        super(DecisionTreeRegressorAlgorithm, self).__init__(params)
        self.library_version = sklearn.__version__
        self.model = DecisionTreeRegressor(criterion=params.get('criterion', 'mse'),
                                           max_depth=params.get('max_depth', 3),
                                           random_state=params.get('seed',1),)
        
    def interpret(self, X_train, y_train, X_validation, y_validation, model_file_path, learner_name, target_name=None, class_names=None, metric_name=None, ml_task=None, explain_level=2,):
        super(DecisionTreeRegressorAlgorithm, self).interpret(X_train, y_train, X_validation, y_validation, model_file_path, learner_name, target_name=None, class_names=None, metric_name=None, ml_task=None, explain_level=2,)

        # if explanation level is zero, no further interpretation or visualization is needed
        if explain_level == 0: return 0
        
        try:
            pass
        except Exception as e:
            print(f'ERROR: Problem when visualizing regression decision tree. {str(e)}')    