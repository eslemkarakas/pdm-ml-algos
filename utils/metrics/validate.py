# import standard packages
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, GroupKFold, TimeSeriesSplit

# define an abstract class to be template for concreted cv method classes
class Validator(ABC):
    @abstractmethod
    def extract(self):
        pass

# define the validator method classes
class KFoldMethod(Validator):
    '''
    Usage: This method is a standard method to validate any dataset.
    '''
    def __init__(self, params):
        self.validator = KFold(n_splits=params.get('cv_n_fold', 5), 
                               shuffle=params.get('cv_shuffle', False),
                               random_state=params.get('cv_random_state', None),)
    
    def extract(self, X, y):
        return self.validator.split(X, y)
    
class StratifiedKFoldMethod(Validator):
    '''
    Usage: This method is usually good for validating small and imbalanced dataset.
    '''
    def __init__(self, params):
        self.validator = StratifiedKFold(n_splits=params.get('cv_n_fold', 5), 
                                         shuffle=params.get('cv_shuffle', False),
                                         random_state=params.get('cv_random_state', None),)
    
    def extract(self, X, y):
        return self.validator.split(X, y)
    
class TimeSeriesSplitMethod(Validator):
    '''
    Usage: This method is usually good for validating time-series datasets.
    '''
    def __init__(self, params):
        self.validator = TimeSeriesSplit(n_splits=params.get('cv_n_fold', 5), 
                                         max_train_size=params.get('cv_max_train_size', None), 
                                         test_size=params.get('cv_test_size', None),)
    
    def extract(self, X, y):
        return self.validator.split(X, y)

    
# define a factory class to access all scaler in one place
class ValidatorFactory:
    @staticmethod
    def create_validator(option, params):
        if option == 'standard':
            return KFoldMethod(params)
        elif option == 'stratified':
            return StratifiedKFoldMethod(params)
        elif option == 'time-series':
            return TimeSeriesSplitMethod(params)
        else:
            raise ValueError(f'ERROR: Unsupported validator option - {option}')
    
# define a wrapper function for handling exceptions
def call_validator(option, params):
    validator = None
    try:
        validator = ValidatorFactory.create_validator(option, params)
    except Exception as e:
        print(f'{str(e)}')
    return validator