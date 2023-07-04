from abc import ABC, abstractmethod
    
from sklearn.base import BaseEstimator, TransformerMixin
class Preprocessor(ABC):
    @abstractmethod
    def fit_transform(self, data):
        pass
    
class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, _remove_features, exact= False):
        self._remove_features = _remove_features
        self._features = None
        self.exact = exact
        
    def fit(self, X, y=None):
        
        # FeatureFilter doesn't require fitting, so this method is a no-op
        # return self
        return self

    def _set_features(self, data):
        
        #! optimize
        should_remove_feature = []
        for i in data.columns:
            for j in self._remove_features:
                if j.lower() in i.lower():
                    should_remove_feature.append(i)
                
        self._features = should_remove_feature
        
        
        return
        
        
        
        
    
    def transform(self, X):
        
        if(self.exact):
           #! validate
            return X.drop(columns=self._remove_features, axis=1)
                
        self._set_features(X)
        print(f"Columns : {self._features} are removed from the dataset")
        return X.drop(columns=self._features, axis=1)
    
    
