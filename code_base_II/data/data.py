
from sklearn import set_config
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import seaborn as sns

from sklearn import config_context

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
    
    
    
    def set_feature_types(self, type_of_features):
        
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
        self._categorical_features = type_of_features[type_of_features["Type"]=="categorical"]["Feature"].values
        self._numerical_features = type_of_features[type_of_features["Type"]=="date"]["Feature"].values
        self._date_features = type_of_features[type_of_features["Type"]=="numerical"]["Feature"].values
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
            if(preprocessor.__name__ == "FeatureFilter"):
                ##print(preprocessor)
                self._preprocessed_data = preprocessor().preprocess(self._preprocessed_data, self._clean_features)
            elif(preprocessor.__name__ == "TypePreprocessor"):
                #! if this run again there will be an issue
                self.set_feature_types()
                ##print("hello",self._categorical_features)
                self._preprocessed_data = preprocessor().preprocess(self._preprocessed_data, self.get_feature_types())
            # self._preprocessed_data = preprocessor.preprocess(self._preprocessed_data, self._clean_features)
            # self._preprocessed_data = preprocessor.preprocess(self._preprocessed_data, self._clean_features)
       
        # self._preprocessed_data = self._preprocessor.fit_transform(self._preprocessed_data)
        
    def set_feature_types(self):
        """
        Identifies the feature types in the raw data and sets them in the Data object.

        Returns:
            self: Returns the instance of the Data object with the feature types set.
        Raises:
            ValueError: Raised when the raw data is not set.
            Exception: Raised when a feature has an unsupported data type.

        This method identifies the feature types (numerical, categorical, or date) in the raw data and sets them in the Data object.
        It first checks if the raw data is set; if not, a ValueError is raised.
        The feature types are stored in a pandas DataFrame with columns for 'Feature', 'Type', and 'Format'.
        The method iterates over each column in the raw data and determines its data type.
        - If the column's data type is 'float64' or 'int64', it is considered a numerical feature.
        - If the column's data type is 'object', it attempts to parse the column as a date using the 'MM/DD/YYYY' format.
        If the parsing is successful, the column is considered a date feature; otherwise, it is considered a categorical feature.
        - If a column has any other data type, an exception is raised.
        The method sets the feature types in the Data object using the 'set_feature_types' method.
        The Data object should have a 'set_feature_types' method to receive and store the feature types.
        The method returns the updated Data object.

        Example:
            data.add_feature_types()
        """
        
        if self._preprocessed_data is None:
            raise ValueError("raw data must be set")
        

        records = []
        for column in self._preprocessed_data.columns:
            column_type = self._preprocessed_data[column].dtype
            
            if column_type in ['float64', 'int64']:
                record = {'Feature': column, 'Type': 'numerical', 'Format': ''}
            
            elif column_type == 'datetime64[ns]':
                record = {'Feature': column, 'Type': 'date', 'Format': 'MM/DD/YYYY'}
                
                
            elif column_type == 'object':
                try:
                    # Attempt to parse the column as a date
                    parse(self._preprocessed_data[column].iloc[0], dayfirst=True)
                    record = {'Feature': column, 'Type': 'date', 'Format': 'MM/DD/YYYY'}
                except ValueError:
                    record = {'Feature': column, 'Type': 'categorical', 'Format': ''}
            else:
                raise Exception(f"Feature ({column} : {self._preprocessed_data[column].iloc[0]}) is not among float64|int64|string|datetime", )
            records.append( record )
            
        feature_types = pd.DataFrame.from_records(records,columns=['Feature', 'Type', 'Format'])
        
        self.feature_types = feature_types
        return feature_types 
    
    def get_feature_types(self):
        return self.feature_types 
      
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
        
        
        # ##print(self._raw_data.columns)
       
        # preprocessor = ColumnTransformer(transformers=[(str(type(preprocessor).__name__), preprocessor, []) for i, preprocessor in enumerate(self.preprocessors)], remainder= "passthrough")

        
        
        data.set_preprocessors(self.preprocessors)
        
 
        # data.set_feature_types(self.feature_types)
        
        data.set_clean_features(self.clean_features)
        
        return data
    

    
from feature_filter import FeatureFilter
    
if __name__ == "__main__":
    
    data = DataBuilder()
    
    raw_data = pd.read_csv("./test.csv")
    
    
    
    
    # preprocessor = data.with_feature_filter(("feture_detect",FeatureFilter(["Project ID"]))).build()
    
    data = DataBuilder()\
        .load_data("./test.csv",1000)\
        .add_clean_features(['ID',"Unnamed:"])\
        .add_preprocessor(FeatureFilter)\
        .add_preprocessor(TypePreprocessor)\
        .build()
        
        
    data.preprocess_data()
    
    
    print("........................")
    
    print(data.get_preprocessed_data().head())
    
    print(data.get_feature_types())
    
    feature_type = data.get_feature_types()[data.get_feature_types()["Type"]=="categorical"]
    print(feature_type)
    
    feature_type = data.get_feature_types()[data.get_feature_types()["Type"]=="date"]
    print(feature_type)
    
    num_feature_type = data.get_feature_types()[data.get_feature_types()["Type"]=="numerical"]["Feature"]
    print(num_feature_type)
    numeric_features = ['Donation Amount', 'Donor Cart Sequence', 'Donor Zip', 'Teacher Project Posted Sequence']
    numeric_data = data.get_data()[num_feature_type]
    print(numeric_data)
    
    correlation_matrix = numeric_data.corr(method="kendall")
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

   
