import config
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

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

    data["Teacher Prefix"] = data["Teacher Prefix"] \
        .fillna(data["Teacher Prefix"].mode()[0])
    
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



if __name__ == "__main__":
    print("Start data pre processing")
    data = load_data_to_df(config.DATA_SOURCE, rows=10)

    data = set_data_types_to_datetime(data, config.DATE_COLS)

    data = impute_data(data)

    

