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
class EncodingFactory:
    @staticmethod
    def create_encoder(encoding_option):
        if encoding_option == 'label':
            return LabelEncoderMethod()
        elif encoding_option == 'one-hot':
            return None
        elif encoding_option == 'target':
            return None
        else:
            raise ValueError(f'ERROR: Unsupported encoding option - {encoding_option}')
    
# define a wrapper function for handling exceptions
def call_encoder(encoding_option):
    encoder = None
    try:
        encoder = EncodingFactory.create_encoder(encoding_option)
    except Exception as e:
        print(f'{str(e)}')
    return encoder