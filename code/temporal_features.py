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
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["Teacher ID"] == teacher_id) & 
                    (df["Project Posted Date"] <= posted_date) & 
                    (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    # print(df_filtered)
    # print(df_filtered_row_count)
    # print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    return success_filtered_count/df_filtered_row_count

# Function to create the teacher success rate for the past 4 months
def create_teacher_success_rate_feature(df, x):

  # Get length of dataframe
    df_len = df.shape[0]
    teacher_success_rate_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        success_rate = calculate_teacher_sucess_rate(row, df, x)
        # print(f'Teacher success rate = {success_rate}')
        teacher_success_rate_values.append(success_rate)


    df["Teacher Success Rate"] = teacher_success_rate_values

    return df


def calculate_school_city_sucess_rate(row_number, df, x):

    # Store School City and posted date
    row = df.iloc[row_number, :]
    posted_date = row["Project Posted Date"]
    school_city = row["School City"]
    posted_date_minus_x = row["Project Posted Date"] - DateOffset(months=x)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School City"] == school_city) & 
                    (df["Project Posted Date"] <= posted_date) & 
                    (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    # print(df_filtered)
    # print(df_filtered_row_count)
    # print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    return success_filtered_count/df_filtered_row_count


def create_school_city_success_rate_feature(df, x):

    # Get length of dataframe
    df_len = df.shape[0]
    school_success_rate_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        success_rate = calculate_school_city_sucess_rate(row, df, x)
        # print(f'School city success rate = {success_rate}')
        school_success_rate_values.append(success_rate)


    df["School City Success Rate"] = school_success_rate_values

    return df


def calculate_school_success_rate(row_number, df, x):

    # Store school ID and posted date
    row = df.iloc[row_number, :]
    posted_date = row["Project Posted Date"]
    school_id = row["School ID"]
    posted_date_minus_x = row["Project Posted Date"] - DateOffset(months=x)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School ID"] == school_id) & 
                    (df["Project Posted Date"] <= posted_date) & 
                    (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    # print(df_filtered)
    # print(df_filtered_row_count)
    # print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    return success_filtered_count/df_filtered_row_count



def create_school_success_rate_feature(df, x):

    # Get length of dataframe
    df_len = df.shape[0]
    school_success_rate_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        success_rate = calculate_school_success_rate(row, df, x)
        # print(f'School success rate = {success_rate}')
        school_success_rate_values.append(success_rate)


    df["School Success Rate"] = school_success_rate_values

    return df


def calculate_school_county_sucess_rate(row_number, df, x):

    # Store School County and posted date
    row = df.iloc[row_number, :]
    posted_date = row["Project Posted Date"]
    school_county = row["School County"]
    posted_date_minus_x = row["Project Posted Date"] - DateOffset(months=x)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School County"] == school_county) & 
                    (df["Project Posted Date"] <= posted_date) & 
                    (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    # print(df_filtered)
    # print(df_filtered_row_count)
    # print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    return success_filtered_count/df_filtered_row_count


def create_school_county_success_rate_feature(df, x):

    # Get length of dataframe
    df_len = df.shape[0]
    school_success_rate_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        success_rate = calculate_school_county_sucess_rate(row, df, x)
        # print(f'School county success rate = {success_rate}')
        school_success_rate_values.append(success_rate)


    df["School County Success Rate"] = school_success_rate_values

    return df


# for the number of projects in a certain state
def calculate_project_count(row_number, df, x):

    row = df.iloc[row_number, :]
    posted_date = row["Project Posted Date"]
    school_state = row["School State"]
    posted_date_minus_x = row["Project Posted Date"] - DateOffset(months=x)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School State"] == school_state) &
                    (df["Project Posted Date"] <= posted_date) &
                    (df["Project Posted Date"] >= posted_date_minus_x)]

    # Find the number of projects from that state
    df_filtered_row_count = df_filtered.shape[0]
    # print(df_filtered)
    # print(df_filtered_row_count)
    # print(df_filtered.shape)

    return df_filtered_row_count

def create_projects_in_a_state_feature(df, x):

    # Get length of dataframe
    df_len = df.shape[0]
    number_of_projects_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        project_count = calculate_project_count(row, df, x)
        # print(f'Project count = {project_count}')
        number_of_projects_values.append(project_count)


    df["Project Count in State"] = number_of_projects_values

    return df


# Function to add new features that are not static
def add_new_features(df):

    # Add the teacher success rate column
    modified_df_tsr = create_teacher_success_rate_feature(df, 4)
    # Add the school city success rate
    modified_df_scr = create_school_city_success_rate_feature(modified_df_tsr, 4)
    # Add the school id sucess rate
    modified_df_ssr = create_school_success_rate_feature(modified_df_scr, 4)
    # Add the school county success rate
    modified_df_sctr = create_school_county_success_rate_feature(modified_df_ssr, 4)
    # Add the number of projects in a state for a selected period of time
    modified_df_project_count = create_projects_in_a_state_feature(modified_df_sctr, 4)

    return modified_df_project_count