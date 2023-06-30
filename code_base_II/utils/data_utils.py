#Import packages
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import os


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



