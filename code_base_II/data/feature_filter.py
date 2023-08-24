from abc import ABC, abstractmethod
    
from sklearn.base import BaseEstimator, TransformerMixin

from data_utils import find_feature_types
class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, data, features):
        pass
    
#! what if i want to remove not exact features to be removed

from abc import ABC, abstractmethod
    
from sklearn.base import BaseEstimator, TransformerMixin

from data_utils import find_feature_types
class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, data, features):
        pass
    
#! what if i want to remove not exact features to be removed

class FeatureFilter(Preprocessor):
    def __init__(self, features):
        self.features = features
        self._remove_features = None
        

    def _set_features(self, data):
        
        #! optimize
        should_remove_feature = []
        for i in data.columns:
            for j in self.features:
                if j.lower() in i.lower():
                    should_remove_feature.append(i)
        self._remove_features = should_remove_feature
        return
        
    
    def preprocess(self, data):
        features = find_feature_types(data)
        if(features is None):
            raise ValueError("features is a must")
        self._set_features(data)
        return data.drop(columns=self._remove_features, axis=1)
    

class CleanFeatures(Preprocessor):
    def __init__(self):
        
        self._remove_features = None
        

    def _set_features(self, data, features):
        
        #! optimize
        should_remove_feature = []
        for i in data.columns:
            for j in features:
                if j.lower() in i.lower():
                    should_remove_feature.append(i)
                
        self._remove_features = should_remove_feature
        return
        
    
    def preprocess(self, data, features):
        
        if(features is None):
            raise ValueError("features is a must")
        self._set_features(data, features)
        return data.drop(columns=self._remove_features, axis=1)
    
    
