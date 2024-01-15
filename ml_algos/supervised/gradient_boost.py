# import standard packages
import os
import sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor

# import local packages
from utils.template import BaseAlgorithm


class GradientBoostingClassifierAlgorithm(BaseAlgorithm):
    algorithm_name = "Gradient Boost" 
    
    def __init__(self, params):
        super(GradientBoostingClassifierAlgorithm, self).__init__(params)
        self.library_version = sklearn.__version__
        self.model = GradientBoostingClassifier(n_estimators=params.get('n_estimator', 10),
                                                learning_rate=params.get('learning_rate', 0.1),
                                                max_depth=params.get('max_depth', 3),)
        
class GradientBoostingRegressorAlgorithm(BaseAlgorithm):
    pass