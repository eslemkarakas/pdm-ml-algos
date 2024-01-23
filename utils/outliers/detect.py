# import standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

# define an abstract class to be template for concreted scaler method classes
class OutlierDetector(ABC):
    @abstractmethod
    def detect(self, data):
        pass
    
    @abstractmethod
    def visualize(self, data):
        pass

# define the scaler method classes
class IsolationForestMethod(OutlierDetector):
    '''
    Usage: This method is suitable for large or high-dimensional datasets.
    '''
    def __init__(self, params):
        self.outliers = None
        self.detector = IsolationForest(contamination=params.get('od_contamination', 0.1),
                                        random_state=params.get('od_random_state', None),)
    
    def detect(self, data):
        self.outliers = self.detector.fit_predict(data)
        return self.outliers
    
    def visualize(self, data):
        # if outliers were not be found, run detect method to be sure
        if self.outliers is None: 
            self.detect(data)
            
        plt.figure(figsize=(10,6))
        plt.scatter(data[:,0], data[:,1], c=self.outliers, cmap='viridis',)
        plt.title("Isolation Forest Outliers")
        plt.show()

class OneClassSVMMethod(OutlierDetector):
    '''
    Usage: This method is suitable for
    '''
    def __init__(self, params):
        self.outliers = None
        self.detector = OneClassSVM(nu=params.get('od_nu', 0.1),)
    
    def detect(self, data):
        self.outliers = self.detector.fit_predict(data)
        return self.outliers
    
    def visualize(self, data):
        # if outliers were not be found, run detect method to be sure
        if self.outliers is None: 
            self.detect(data)
            
        plt.figure(figsize=(10,6))
        plt.scatter(data[:,0], data[:,1], c=self.outliers, cmap='viridis',)
        plt.title("Isolation Forest Outliers")
        plt.show()
        
class LocalOutlierFactorMethod(OutlierDetector):
    '''
    Usage: This method is suitable for ...
    '''
    def __init__(self, params):
        self.outliers = None
        self.detector = LocalOutlierFactor(contamination=params.get('od_contamination', 0.1),)
        
    def detect(self, data):
        self.outliers = self.detector.fit_predict(data)
        return self.outliers
    
    def visualize(self, data):
        # if outliers were not be found, run detect method to be sure
        if self.outliers is None: 
            self.detect(data)
            
        plt.figure(figsize=(10,6))
        plt.scatter(data[:,0], data[:,1], c=self.outliers, cmap='viridis',)
        plt.title("Local Outlier Factor")
        plt.show()

        
# define a factory class to access all scaler in one place
class OutlierDetectorFactory:
    @staticmethod
    def create_outlier_detector(option, params):
        if option == 'isolation-forest':
            return IsolationForestMethod(params)
        elif option == 'one-class-svm':
            return OneClassSVMMethod(params)
        elif option == 'local-outlier-factor':
            return LocalOutlierFactorMethod(params)
        else:
            raise ValueError(f'ERROR: Unsupported outlier detector option - {option}')
    
# define a wrapper function for handling exceptions
def call_outlier_detector(option, params):
    detector = None
    try:
        detector = OutlierDetectorFactory.create_outlier_detector(option, params)
    except Exception as e:
        print(f'{str(e)}')
    return detector