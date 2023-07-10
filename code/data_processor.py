import config
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from datetime import timedelta

def load_data_to_df(path: str, rows: int=None):
    if rows:
        data = pd.read_csv(path, nrows=rows)
    else:
        data = pd.read_csv(path)
    return data

def export_data_frame(data: DataFrame, path: str, columns: list=None):
    data.to_csv(path, columns=columns)

def set_data_types_to_datetime(data_frame: DataFrame, date_type_cols: list):
    for col in date_type_cols:
        data_frame[col] = pd.to_datetime(data_frame[col])
    return data_frame

def impute_data(data: DataFrame):
    data["Donor City"] = data["Donor City"].fillna("Unknown")
    data["Donor Zip"] = data["Donor Zip"].fillna("Unknown")
    data["School City"] = data["School City"].fillna("Unknown")

    data["School Percentage Free Lunch"]=data["School Percentage Free Lunch"]\
        .replace(np.NaN, data["School Percentage Free Lunch"].median())

    data["School County"] = data["School County"] \
        .fillna(data["School County"].mode()[0])
    
    data["School Name"] = data["School Name"].fillna("Unknown")
    data["School District"] = data["School District"].fillna("Unknown")
    data["School State"] = data["School State"].fillna("Unknown")

    data["Teacher Prefix"] = data["Teacher Prefix"] \
        .fillna(data["Teacher Prefix"].mode()[0])
    
    data["Project Title"] = data["Project Title"].fillna("No Title")
    data["Project Essay"] = data["Project Essay"].fillna("None")
    data["Project Short Description"] = data["Project Short Description"].fillna("No Description")
    data["Project Need Statement"] = data["Project Need Statement"].fillna("None")
    
    return data

def encode_data(data: DataFrame, categorical_cols: list):
    """One hot encodes data."""
    return pd.get_dummies(data, columns=categorical_cols)

def split_time_series_train_test_data(data: DataFrame, filter_date: str):
    train_set = data[
        data["Project Posted Date"] < pd.to_datetime(filter_date)
        ].drop(["Project ID", "Project Posted Date"], axis=1)
    x_train = train_set.loc[:, train_set.columns != "Label"]
    y_train = train_set.loc[:, ["Label"]]

    test_set = data[
        data["Project Posted Date"] >= pd.to_datetime(filter_date)
        ].drop(["Project ID", "Project Posted Date"], axis=1)
    x_test = test_set.loc[:, test_set.columns != "Label"]
    y_test = test_set.loc[:, ["Label"]]

    return x_train, y_train, x_test, y_test

def split_temporal_train_test_data(
        data: DataFrame,
        start_date: str,
        train_months: int = 6, 
        test_months: int = 1, 
        leak_offset: int = 4) -> DataFrame:
    
    # data split format = train - offset - validate - offset - test
    # validation period = test period
    train_start = pd.Timestamp(start_date)
    train_end = train_start + timedelta(days=train_months*30)
    test_start = train_end + timedelta(days=leak_offset*30)
    test_end = test_start + timedelta(days=test_months*30)

    print("-----")
    print(f"train    {str(train_start)[:10]} - {str(train_end)[:10]}")
    print(f"test     {str(test_start)[:10]} - {str(test_end)[:10]}")

    train_set = data[
        (data["Project Posted Date"] > pd.to_datetime(train_start))
        ]
    train_set = data[
        (data["Project Posted Date"] < pd.to_datetime(train_end))
        ].drop(["Project ID", "Project Posted Date"], axis=1)
    
    x_train = train_set.loc[:, train_set.columns != "Label"]
    y_train = train_set.loc[:, ["Label"]]
    
    test_set = data[
        (data["Project Posted Date"] > pd.to_datetime(test_start))]
    test_set = data[
        (data["Project Posted Date"] < pd.to_datetime(test_end))
        ].drop(["Project ID", "Project Posted Date"], axis=1)
    
    x_test = test_set.loc[:, test_set.columns != "Label"]
    y_test = test_set.loc[:, ["Label"]]

    print("Training set shape = ", x_train.shape)
    print("Testing set shape = ", x_test.shape)
    print("-----")
    
    return x_train, y_train, x_test, y_test
    
if __name__ == "__main__":
    print("Start data pre processing")
    data = load_data_to_df(config.DATA_SOURCE, rows=10)

    data = set_data_types_to_datetime(data, config.DATE_COLS)

    data = impute_data(data)

    

