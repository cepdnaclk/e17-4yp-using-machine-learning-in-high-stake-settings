from abc import ABC, abstractmethod
    
from sklearn.base import BaseEstimator, TransformerMixin
class Preprocessor(ABC):
    @abstractmethod
    def fit_transform(self, data):
        pass
    
class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, remove_features):
        print("hiii",remove_features)
        self.remove_features = remove_features
        
    def fit(self, X, y=None):
        print("hi2")
        
        # FeatureFilter doesn't require fitting, so this method is a no-op
        # return self
        return self
    
    def transform(self, X):
        # FeatureFilter doesn't require fitting, so this method is a no-op
        # return self
        print("_"*10)
        # Select only the specified features from the input X
        
        should_remove_feature = []
        for i in X.columns:
            if "Unnamed:" in i.split(" ") or "ID" in i.split(" "):
                should_remove_feature.append(i)
                
        self.remove_features = should_remove_feature
        print("should_remove_feature")
        
        return X.drop(columns=should_remove_feature, axis=1)
    
    # def fit_transform(self, X):
    #     print("_"*10)
    #     # Select only the specified features from the input X
        
    #     should_remove_feature = []
    #     for i in X.columns:
    #         if "Unnamed:" in i.split(" ") or "ID" in i.split(" "):
    #             should_remove_feature.append(i)
                
        
    #     print(should_remove_feature)
        
    #     return X.drop(columns=should_remove_feature)
    
