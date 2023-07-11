from abc import ABC, abstractmethod
    

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, data):
        pass
    

class TypePreprocessor(Preprocessor):
    def __init__(self):
        # self.categorical_features = 
        self.encoder = OneHotEncoder()
        self.scaler = StandardScaler()
       
    # data should be 
    def preprocess(self, data: pd.DataFrame, feature_types):
        
        
        numeric_data = data[feature_types[feature_types["Type"]=="numerical"]["Feature"]]
        scaled_data = self.scaler.fit_transform(numeric_data)
        scaled_data = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        
        
        date_data = data[feature_types[feature_types["Type"]=="date"]["Feature"]].copy()
        
        # for column in date_data.columns:
        #     date_data[column] = pd.to_datetime(date_data[column])
        #     date_data[column + '_Year'] = date_data[column].dt.year
        #     date_data[column + '_Month'] = date_data[column].dt.month
        #     date_data[column + '_Day'] = date_data[column].dt.day
        #     date_data = date_data.drop(column,axis=1)
        for column in date_data.columns:
            date_data[column] = pd.to_datetime(date_data[column])
        
        categorical_data = data[feature_types[feature_types["Type"]=="categorical"]["Feature"]]
        encoded_data = self.encoder.fit_transform(categorical_data).toarray()
        encoded_columns = self.encoder.get_feature_names_out(categorical_data.columns)
        encoded_data = pd.DataFrame(encoded_data, columns=encoded_columns)
        
        preprocessed_data = pd.concat([date_data, scaled_data,encoded_data],axis=1)
    
        return preprocessed_data


