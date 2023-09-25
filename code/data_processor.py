import config
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Union


def load_data_to_df(path: str, rows: int = None):
    if rows:
        data = pd.read_csv(path, nrows=rows)
    else:
        data = pd.read_csv(path)
    return data


def export_data_frame(data: DataFrame, path: str, columns: list = None):
    data.to_csv(path, columns=columns)


def save_json(dict_obj: Union[dict, list], path: str):
    writable_json = json.dumps(dict_obj, indent=4)
    with open(path, 'w') as file:
        file.write(writable_json)


def set_data_types_to_datetime(data_frame: DataFrame, date_type_cols: list):
    for col in date_type_cols:
        data_frame[col] = pd.to_datetime(data_frame[col])
    return data_frame


def impute_data(data: DataFrame):
    data["School City"] = data["School City"].fillna("Unknown")
    data["School Percentage Free Lunch"] = data["School Percentage Free Lunch"]\
        .replace(np.NaN, data["School Percentage Free Lunch"].median())
    data["School County"] = data["School County"].fillna("Unknown")
    data["School District"] = data["School District"].fillna("Unknown")
    data["School State"] = data["School State"].fillna("Unknown")
    data["School Metro Type"] = data["School Metro Type"].fillna("Unknown")

    data["Teacher Prefix"] = data["Teacher Prefix"] \
        .fillna(data["Teacher Prefix"].mode()[0])

    data["Project Subject Category Tree"] = data["Project Subject Category Tree"].fillna(
        "Unknown")
    data["Project Subject Subcategory Tree"] = data["Project Subject Subcategory Tree"].fillna(
        "Unknown")
    data["Project Resource Category"] = data["Project Resource Category"].fillna(
        "Unknown")

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
    print(train_set.shape)
    train_set = train_set[
        (train_set["Project Posted Date"] < pd.to_datetime(train_end))
    ].drop(["Project Posted Date"], axis=1)
    print(train_set.shape)

    x_train = train_set.loc[:, train_set.columns != "Label"]
    print(x_train.shape)
    y_train = train_set.loc[:, ["Label"]]
    print(y_train.shape)

    test_set = data[
        (data["Project Posted Date"] > pd.to_datetime(test_start))]
    print(test_set.shape)
    
    test_set = test_set[
        (test_set["Project Posted Date"] < pd.to_datetime(test_end))
    ].drop(["Project Posted Date"], axis=1)
    print(test_set.shape)

    x_test = test_set.loc[:, test_set.columns != "Label"]
    print(x_test.shape)

    y_test = test_set.loc[:, ["Label"]]
    print(x_test.shape)

    print("Training set shape = ", x_train.shape)
    print("Testing set shape = ", x_test.shape)
    print("-----")

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    print("Start data pre processing")
    data = load_data_to_df(config.DATA_SOURCE, rows=config.MAX_ROWS)
    print(data.shape)

    data = set_data_types_to_datetime(data, config.DATE_COLS)
    print(data.shape)

    # data = impute_data(data)

    data.set_index("Project Posted Date", inplace=True)
    data_count_by_month = data.resample('M').size()
    # Plot the data count distribution
    plt.figure(figsize=(10, 6))
    ax = data_count_by_month.plot(kind='bar')

    # Remove the time part from the date labels
    x_labels = [d.strftime('%Y-%b') for d in data_count_by_month.index]

    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    plt.xlabel('Month')
    plt.ylabel('Projects Count')
    plt.title('Projects Count Distribution by Month')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
