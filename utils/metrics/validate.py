# import standard packages
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, GroupKFold, TimeSeriesSplit

# define an abstract class to be template for concreted cv method classes
class Validator(ABC):
    @abstractmethod
    def extract(self, data):
        pass

# define the validator method classes
class KFoldMethod(Validator):
    '''
    Usage: This method is a standard method to validate any dataset.
    '''
    def __init__(self, X, y, params):
        self.X = X
        self.y = y
        self.validator = KFold(n_splits=params.get('cv_n_fold', 5), 
                               shuffle=params.get('cv_shuffle', False),
                               random_state=params.get('cv_random_state', 42),)
    
    def extract(self):
        return self.validator.split(self.X, self.y)
    
class StratifiedKFoldMethod(Validator):
    '''
    Usage: This method is usually good for validating small and imbalanced dataset.
    '''
    def __init__(self, X, y, params):
        self.X = X
        self.y = y
        self.validator = StratifiedKFold(n_splits=params.get('cv_n_fold', 5), 
                                         shuffle=params.get('cv_shuffle', False),
                                         random_state=params.get('cv_random_state', 42),)
    
    def extract(self):
        return self.validator.split(self.X, self.y)
    
class TimeSeriesSplitMethod(Validator):
    '''
    Usage: This method is usually good for validating time-series datasets.
    '''
    def __init__(self, X, y, params):
        self.X = X
        self.y = y
        self.validator = TimeSeriesSplit(n_splits=params.get('cv_n_fold', 5), 
                                         max_train_size=params.get('cv_max_train_size', None), 
                                         test_size=params.get('cv_test_size', None),)
    
    def extract(self):
        return self.validator.split(self.X, self.y)

    
# define a factory class to access all scaler in one place
class ValidatorFactory:
    @staticmethod
    def create_validator(option):
        if option == 'standard':
            return KFoldMethod()
        elif option == 'stratified':
            return StratifiedKFoldMethod()
        elif option == 'time-series':
            return TimeSeriesSplitMethod()
        else:
            raise ValueError(f'ERROR: Unsupported validator option - {option}')
    
# define a wrapper function for handling exceptions
def call_validator(option):
    validator = None
    try:
        validator = ValidatorFactory.create_validator(option)
    except Exception as e:
        print(f'{str(e)}')
    return validator