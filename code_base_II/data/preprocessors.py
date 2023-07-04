from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

class NumericPreprocessor:
    def __init__(self, numeric_features):
        self.numeric_features = numeric_features
        self.scaler = StandardScaler()
    
    def preprocess(self, data):
        numeric_data = data[self.numeric_features]
        scaled_data = self.scaler.fit_transform(numeric_data)
        scaled_data = pd.DataFrame(scaled_data, columns=self.numeric_features)
        return scaled_data

class CategoricalPreprocessor:
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.encoder = OneHotEncoder()
    
    def preprocess(self, data):
        categorical_data = data[self.categorical_features]
        encoded_data = self.encoder.fit_transform(categorical_data).toarray()
        encoded_columns = self.encoder.get_feature_names_out(self.categorical_features)
        encoded_data = pd.DataFrame(encoded_data, columns=encoded_columns)
        return encoded_data
