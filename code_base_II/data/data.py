
from sklearn import set_config
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import seaborn as sns

from sklearn import config_context
from data_utils import find_feature_types

from preprocessors import  TypePreprocessor


class Data:
    
    """
    A class that represents data and provides methods for preprocessing.

    Attributes:
        _raw_data (pandas.DataFrame): The raw data.
        _preprocessed_data (pandas.DataFrame): The preprocessed data.
        preprocessor: The data preprocessor.
        _categorical_features (list): List of categorical feature names.
        _numerical_features (list): List of numerical feature names.
        _date_features (list): List of date feature names.
    """
    
    
    def __init__(self , raw_data ):
        """
        Initializes a new instance of the Data class.

        Args:
            _raw_data (pandas.DataFrame): The raw data.
            _preprocessed data (pandas.DataFrame): The preprocessed data.
            _preprocessor (Preprocessor): These are the functions used to modify the data.
            _categorical_features = features that have categorical values
            _numerical_features = features that have numerical values
            _date_features = features that have date values
            
        """
        
        self._raw_data = raw_data
        self._preprocessed_data = raw_data
        self._preprocessors = None
        
        self._categorical_features = None
        self._numerical_features = None
        self._date_features = None
        self._clean_features = None
    
    
    
    def set_feature_types(self):
        
        
        type_of_features = find_feature_types(self._preprocessed_data)
        
        """
        Sets the feature types for the data.

        Args:
            type_of_features (dict): A dictionary containing the feature types.

        Example:
            type_of_features = {
                "categorical_features": ["feature1", "feature2"],
                "numeric_features": ["feature3", "feature4"],
                "date_features": ["feature5", "feature6"]
            }
            data.set_feature_types(type_of_features)
        """
        self.feature_types = type_of_features
        self._categorical_features = type_of_features[type_of_features["Type"]=="categorical"]["Feature"].values
        self._date_features = type_of_features[type_of_features["Type"]=="date"]["Feature"].values
        self._numerical_features = type_of_features[type_of_features["Type"]=="numerical"]["Feature"].values
        ##print(self._categorical_features)
        return
    
    def set_clean_features(self, clean_features):
        
        self._clean_features = clean_features
        
        return
    
    def set_preprocessors(self, preprocessors):
        """
        Sets the data preprocessor.

        Args:
            preprocessor: The preprocessor object.
        """
        self._preprocessors = preprocessors
        
    def preprocess_data(self):
        """
        Preprocesses the data using the set preprocessor.

        Raises:
            ValueError: If the preprocessor is not set.
        """
        
        if(self._preprocessors is None):
            raise ValueError("Preprocessor must be set")
        # for preprocess in self.preprocessors:
        # with sklearn.config_context(transform_output="pandas"):
        # with config_context(transform_output="pandas"):
        
        
        for preprocessor in self._preprocessors:
            
            # if(type(preprocessor).__name__ == "FeatureFilter"):
            #     self._preprocessed_data = preprocessor.preprocess(self._preprocessed_data)
            # elif(type(preprocessor).__name__ == "TypePreprocessor"):
            #     #! if this run again there will be an issue
            #     # self.set_feature_types()
            #     ##print("hello",self._categorical_features)
            self._preprocessed_data = preprocessor.preprocess(self._preprocessed_data)
            # self._preprocessed_data = preprocessor.preprocess(self._preprocessed_data, self._clean_features)
            # self._preprocessed_data = preprocessor.preprocess(self._preprocessed_data, self._clean_features)
       
        # self._preprocessed_data = self._preprocessor.fit_transform(self._preprocessed_data)
        
   
        
    def get_feature_types(self):
        return self.feature_types 
    
    def get_numerical_features(self):
        return self._numerical_features 
    
    def get_date_features(self):
        return self._date_features 
    
    def get_categorical_features(self):
        return self._categorical_features 
      
      
    def get_data(self):
        """
        Returns the raw data.

        Returns:
            pandas.DataFrame: The raw data.
        """
        return self._raw_data
        
    def get_preprocessed_data(self):
        """
        Returns the preprocessed data.

        Returns:
            pandas.DataFrame: The preprocessed data.
        """
        
        return self._preprocessed_data 
    

from dateutil.parser import parse
    
class DataBuilder:
    
    def __init__(self):
        self._raw_data = None
        self.preprocessors = []
        self.feature_types = None
        self.clean_features = None
        
    def _load_data(self, file_path, rows):
        return pd.read_csv(file_path, nrows=rows)
        
    def load_data(self, file_path, rows):
        self._raw_data = self._load_data(file_path, rows)
        return self
    
    def add_preprocessor(self, preprocessor):
        self.preprocessors.append(preprocessor)
        return self
    
    
    def add_clean_features(self, clean):
        
        self.clean_features = clean
        
        return self

        
        
        
    
    def build(self):
        if self._raw_data is None or not self.preprocessors:
            raise ValueError("raw data and preprocessor must be set")
        
        data = Data(self._raw_data.copy())
        data.set_preprocessors(self.preprocessors)
        data.set_clean_features(self.clean_features)
        return data
    

    
from feature_filter import FeatureFilter
    
if __name__ == "__main__":
    
    data = DataBuilder()
    
    raw_data = pd.read_csv("./test.csv")
    
    data = DataBuilder()\
        .load_data("./test.csv",1000)\
        .add_preprocessor(FeatureFilter(['ID',"Unnamed:"]))\
        .add_preprocessor(TypePreprocessor())\
        .build()
        
        
    data.preprocess_data()
    data.set_feature_types() 
    
    print(data.get_preprocessed_data())   
    feature_type = data.get_categorical_features()
    print(feature_type)
    
    feature_type = data.get_date_features()
    print(feature_type)
    
    
    
    num_feature_type = data.get_numerical_features()
    print(num_feature_type)
    numeric_features = ['Donation Amount', 'Donor Cart Sequence', 'Donor Zip', 'Teacher Project Posted Sequence']
    numeric_data = data.get_data()[num_feature_type]
    print(numeric_data)
    
    correlation_matrix = numeric_data.corr(method="kendall")
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

   
