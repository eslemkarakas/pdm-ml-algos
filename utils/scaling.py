# import standard packages
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer

# define an abstract class to be template for concreted scaler method classes
class Scaler(ABC):
    @abstractmethod
    def scale(self, data):
        pass

# define the scaler method classes
class MinMaxScalerMethod(Scaler):
    def __init__(self):
        self.data = None
    
    def scale(self, data):
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(data)
        return self.data
    
class StandardScalerMethod(Scaler):
    def __init__(self):
        self.data = None
    
    def scale(self, data):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(data)
        return self.data

class RobustScalerMethod(Scaler):
    def __init__(self):
        self.data = None
    
    def scale(self, data):
        scaler = RobustScaler()
        self.data = scaler.fit_transform(data)
        return self.data
    
class MaxAbsScalerMethod(Scaler):
    def __init__(self):
        self.data = None
    
    def scale(self, data):
        scaler = MaxAbsScaler()
        self.data = scaler.fit_transform(data)
        return self.data

# define a factory class to access all scaler in one place
class ScalingFactory:
    @staticmethod
    def create_scaler(scaling_option):
        if scaling_option == 'min-max':
            return MinMaxScalerMethod()
        elif scaling_option == 'standard':
            return StandardScalerMethod()
        elif scaling_option == 'robust':
            return RobustScalerMethod()
        elif scaling_option == 'max-abs':
            return MaxAbsScalerMethod()
        else:
            raise ValueError(f'ERROR: Unsupported scaling option - {scaling_option}')
    
# define a wrapper function for handling exceptions
def call_scaler(scaling_option):
    scaler = None
    try:
        scaler = ScalingFactory.create_scaler(scaling_option)
    except Exception as e:
        print(f'{str(e)}')
    return scaler