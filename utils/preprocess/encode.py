# import standard packages
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
 
# define an abstract class to be template for concreted encoder method classes
class Encoder(ABC):
    @abstractmethod
    def encode(self, data):
        pass
    
    @abstractmethod
    def decode(self, data):
        pass

# define the encoder method classes
class LabelEncoderMethod(Encoder):
    def __init__(self):
        self.data = None
        self.encoder = LabelEncoder()
    
    def encode(self, data):
        self.data = self.encoder.fit_transform(data)
        return self.data

    def decode(self, data):
        self.data = self.encoder.inverse_transform(data)
        return self.data
    
# define a factory class to access all scaler in one place
class EncoderFactory:
    @staticmethod
    def create_encoder(option):
        if option == 'label':
            return LabelEncoderMethod()
        elif option == 'one-hot':
            return None
        elif option == 'target':
            return None
        else:
            raise ValueError(f'ERROR: Unsupported encoding option - {option}')
    
# define a wrapper function for handling exceptions
def call_encoder(option):
    encoder = None
    try:
        encoder = EncoderFactory.create_encoder(option)
    except Exception as e:
        print(f'{str(e)}')
    return encoder