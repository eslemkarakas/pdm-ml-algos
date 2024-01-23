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
        self.scaler = MinMaxScaler()
    
    def scale(self, data):
        self.data = self.scaler.fit_transform(data)
        return self.data
    
class StandardScalerMethod(Scaler):
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
    
    def scale(self, data):
        self.data = self.scaler.fit_transform(data)
        return self.data

class RobustScalerMethod(Scaler):
    def __init__(self):
        self.data = None
        self.scaler = RobustScaler()
        
    def scale(self, data):
        self.data = self.scaler.fit_transform(data)
        return self.data
    
class MaxAbsScalerMethod(Scaler):
    def __init__(self):
        self.data = None
        self.scaler = MaxAbsScaler()
    
    def scale(self, data):
        self.data = self.scaler.fit_transform(data)
        return self.data

# define a factory class to access all scaler in one place
class ScalerFactory:
    @staticmethod
    def create_scaler(option):
        if option == 'min-max':
            return MinMaxScalerMethod()
        elif option == 'standard':
            return StandardScalerMethod()
        elif option == 'robust':
            return RobustScalerMethod()
        elif option == 'max-abs':
            return MaxAbsScalerMethod()
        else:
            raise ValueError(f'ERROR: Unsupported scaling option - {option}')
    
# define a wrapper function for handling exceptions
def call_scaler(option):
    scaler = None
    try:
        scaler = ScalerFactory.create_scaler(option)
    except Exception as e:
        print(f'{str(e)}')
    return scaler