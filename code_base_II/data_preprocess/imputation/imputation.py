from abc import ABC, abstractmethod

class ImputationStrategy(ABC):
    @abstractmethod
    def impute(self, data):
        raise NotImplementedError("impute() method must be implemented by concrete strategy classes.")
        
class MeanImputation(ImputationStrategy):
    def impute(self, data):
        # Impute missing values using mean imputation strategy
        # ...
        pass
    

class MedianImputation(ImputationStrategy):
    def impute(self, data):
        # Impute missing values using median imputation strategy
        # ...
        pass
    
class KNNImputation(ImputationStrategy):
    def impute(self, data):
        # Impute missing values using K-nearest neighbors imputation strategy
        # ...
        pass
    