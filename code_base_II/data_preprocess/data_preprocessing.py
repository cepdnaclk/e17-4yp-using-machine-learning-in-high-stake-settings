from abc import ABC, abstractmethod


class DataPreprecessor(ABC):
    
    @abstractmethod
    def load_dataset(self, file_path):
        pass
    
    
    @abstractmethod
    def handle_missing_values(self, df):
        pass
    
    @abstractmethod
    def perform_feature_engineering(self, df):
        pass
    
    @abstractmethod
    def encode_categorical_variables(self, df ):
        pass

        
    def normalize_numerical_variables(self, df):
        pass


    def save_preprocessed_data(self, df, filename=None):
        pass