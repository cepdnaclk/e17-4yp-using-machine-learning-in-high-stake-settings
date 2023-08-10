# Inmporting libraries
import pandas as pd
from pandas.tseries.offsets import DateOffset


# Function to calculate the teacher success rate 
def calculate_teacher_sucess_rate(row_number, df, x):

    # Store teacher ID and posted date
    row = df.iloc[row_number, :]
    posted_date = row["Project Posted Date"]
    teacher_id = row["Teacher ID"]
    posted_date_minus_x = row["Project Posted Date"] - DateOffset(months=x)
    print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["Teacher ID"] == teacher_id) & 
                    (df["Project Posted Date"] <= posted_date) & 
                    (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    print(df_filtered)
    print(df_filtered_row_count)
    print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    return success_filtered_count/df_filtered_row_count

# Function to create the teacher success rate for the past 4 months
def create_teacher_success_rate_feature(df):

  # Get length of dataframe
    df_len = df.shape[0]
    teacher_success_rate_values = []
    for row in range(df_len):
        print(f'Row number = {row} --------------------------------------------------------')
        success_rate = calculate_teacher_sucess_rate(row, df, 4)
        print(f'success rate = {success_rate}')
        teacher_success_rate_values.append(success_rate)


    df["Teacher Success Rate"] = teacher_success_rate_values

    return df


def calculate_school_city_sucess_rate(row_number, df, x):

    # Store School State and posted date
    row = df.iloc[row_number, :]
    posted_date = row["Project Posted Date"]
    school_city = row["School City"]
    posted_date_minus_x = row["Project Posted Date"] - DateOffset(months=x)
    print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School City"] == school_city) & 
                    (df["Project Posted Date"] <= posted_date) & 
                    (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    print(df_filtered)
    print(df_filtered_row_count)
    print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    return success_filtered_count/df_filtered_row_count


def create_school_city_success_rate_feature(df):

    # Get length of dataframe
    df_len = df.shape[0]
    school_success_rate_values = []
    for row in range(df_len):
        print(f'Row number = {row} --------------------------------------------------------')
        success_rate = calculate_school_city_sucess_rate(row, df, 4)
        print(f'success rate = {success_rate}')
        school_success_rate_values.append(success_rate)


    df["School Success Rate"] = school_success_rate_values

    return df


# Function to add new features that are not static
def add_new_features(df):

    # Add the teacher success rate column
    modified_df_tsr = create_teacher_success_rate_feature(df)
    # Add the school city success rate
    modified_df_scr = create_school_city_success_rate_feature(modified_df_tsr)

    return modified_df_scr