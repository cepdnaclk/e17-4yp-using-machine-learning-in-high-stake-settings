#Import packages
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import os

from dateutil.parser import parse

# Function to load the data
def load_data(file_path: str) -> DataFrame:
    if not os.path.isfile(file_path):
         raise ValueError("File '%s' does not exist"%file_path)
     
    try:
        data = pd.read_csv(file_path)
        
    except:
        raise ValueError("File '%s' is a correpted csv"%file_path)
    
    return data


# Export the dataframe as csv file
def save_data_frame(data: DataFrame, save_path: str) -> DataFrame:
    
    if not os.path.isfile(save_path):
         raise ValueError("File '%s' does not exist"%save_path)
    
    data.to_csv(save_path)
    
    return



def set_data_types_to_datetime(data_frame: DataFrame, date_type_cols: list) -> DataFrame:
    for col in date_type_cols:
        data_frame[col] = pd.to_datetime(data_frame[col])
    return data_frame

def find_feature_types(data : DataFrame):
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
    
    if data is None:
        raise ValueError("raw data must be set")
    

    records = []
    for column in data.columns:
        column_type = data[column].dtype
        
        if column_type in ['float64', 'int64']:
            record = {'Feature': column, 'Type': 'numerical', 'Format': ''}
        
        elif column_type == 'datetime64[ns]':
            record = {'Feature': column, 'Type': 'date', 'Format': 'MM/DD/YYYY'}
            
            
        elif column_type == 'object':
            try:
                # Attempt to parse the column as a date
                parse(data[column].iloc[0], dayfirst=True)
                record = {'Feature': column, 'Type': 'date', 'Format': 'MM/DD/YYYY'}
            except ValueError:
                record = {'Feature': column, 'Type': 'categorical', 'Format': ''}
        
        elif column_type == 'category':
            record = {'Feature': column, 'Type': 'categorical', 'Format': ''}
            
        else:
            raise Exception(f"Feature ({column} : {data[column].iloc[0]}) is not among float64|int64|string|datetime", )
        records.append( record )
        
    feature_types = pd.DataFrame.from_records(records,columns=['Feature', 'Type', 'Format'])
    
    # self.feature_types = feature_types
    return feature_types 




