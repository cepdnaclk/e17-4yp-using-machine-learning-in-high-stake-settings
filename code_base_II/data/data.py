
from sklearn import set_config

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer


from sklearn import config_context

"""_summary_
"""
class Data:
    def __init__(self , raw_data ):
        self._raw_data = raw_data
        self._preprocessed_data = raw_data
        self.preprocessor = None
        self._categorical_features = None
        self._numerical_features = None
        self._date_features = None
    
    
    
    def set_feature_types(self, type_of_features):
        
        self._categorical_features = type_of_features["categorical_features"]
        self._numerical_features = type_of_features["numeric_features"]
        self._date_features = type_of_features["date_features"]
        return
        
        
        
    def _load_data(self, file_path, rows):
        return pd.read_csv(file_path, nrows=100)
    
    
    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor 
        
    def preprocess_data(self):
        if(self.preprocessor is None):
            raise ValueError("Preprocessor must be set")
        
        print(self.preprocessor)
        
        
        # for preprocess in self.preprocessors:
        with config_context(transform_output="pandas"):
            self._preprocessed_data = self.preprocessor.fit_transform(self._preprocessed_data)
        

        
        
    def get_data(self):
        return self._raw_data
        
    def get_preprocessed_data(self):
        return self._preprocessed_data 
    
    # def set_feature_types_auto(self):
        
    #     type_of_features = {
    #         "numeric_features" : [],
    #         "categorical_features" : [],
    #         "date_features" : []
    #     }
        
    #     for column in self.data.get_data().columns:
            
    #         if ('date' in column.lower):
    #             type_of_features["date_features"].append(column)
                
    #         elif (self.data.get_data()[column].dtype in ["float64", "int64"]):
    #             type_of_features["numeric_features"].append(column)
                
    #         elif (self.data.get_data()[column].dtype == 'object'):
    #             type_of_features["categorical_features"].append(column)
        
    #     self.data.set_feature_types(type_of_features)
        
    #     return 


    
class DataBuilder:
    
    def __init__(self):
        self.file_path = None
        self.preprocessors = []
        
    def set_file_path(self, file_path):
        self.file_path = file_path
        return self
    
    def add_preprocessor(self, preprocessor):
        self.preprocessors.append(preprocessor)
        return self
    
    def build(self):
        if self.file_path is None or not self.preprocessors:
            raise ValueError("File path and preprocessor must be set")
        
        raw_data = pd.read_csv(self.file_path)
        
        data = Data(raw_data.copy())
        
        preprocessor = ColumnTransformer(transformers=[(str(type(preprocessor).__name__), preprocessor, list(raw_data.columns)) for i, preprocessor in enumerate(self.preprocessors)], remainder= "passthrough")
 
        data.set_preprocessor(preprocessor)
        
        return data
    
    
    
from feature_filter import FeatureFilter
    
    
    
        
if __name__ == "__main__":
    
    data = DataBuilder()
    
    raw_data = pd.read_csv("./test.csv")
    
    
    
    
    # preprocessor = data.with_feature_filter(("feture_detect",FeatureFilter(["Project ID"]))).build()
    
    data = DataBuilder()\
        .set_file_path("./test.csv")\
        .add_preprocessor(FeatureFilter(['Project ID',"unmapped: 0"]))\
        .build()

    data.preprocess_data()
    
    
    
        
    print(data.get_preprocessed_data())
    
    
    from abc import ABC, abstractmethod
from sklearn.compose import ColumnTransformer
import pandas as pd

class Preprocessor(ABC):
    @abstractmethod
    def fit_transform(self, data):
        pass

class NumericPreprocessor(Preprocessor):
    def fit_transform(self, data):
        numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_transformer = ColumnTransformer(transformers=[('num', 'passthrough', numeric_features)])
        return numeric_transformer.fit_transform(data)