# import standard packages
import uuid
import joblib

class BaseAlgorithm():
    def __init__(self, params):
        self.params = params
        self.stop_training = False
        self.library_version = None
        self.model = None
        self.uid = params.get('uid', str(uuid.uuid4()))
        self.ml_task = params.get('ml_task')
        self.model_file_path = None
        self.name = None
        
    def fit(self, X_train, y_train, X_validation=None, y_validation=None, sample_weight=None,):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
    
    def save(self, model_file_path):
        joblib.dump(self.model, model_file_path, compress=True)
        
    def load(self, model_file_path):
        self.model = joblib.load(model_file_path)
        
    def predict(self, X):
        if self.params['ml_task'] == 'binary_classification':
            return self.model.predict_proba(X)[:, 1]
        elif self.params['ml_task'] == 'multiclass_classification':
            return self.model.predict_proba(X)
        else:
            self.model.predict(X)
        