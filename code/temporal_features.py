# Inmporting libraries
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from helper import (log_intermediate_output_to_file)
import config
from sklearn.feature_extraction.text import TfidfVectorizer


# Function to calculate the teacher success rate
def calculate_teacher_sucess_rate(row_number, df, x):

    success_rate = 0
    success_rate_imputed = 1
    # Store teacher ID and posted date
    row = df.iloc[row_number, :]
    posted_date_minus_one = row["Project Posted Date"] - \
        DateOffset(months=config.LABEL_PERIOD)
    teacher_id = row["Teacher ID"]
    posted_date_minus_x = row["Project Posted Date"] - \
        DateOffset(months=x+config.LABEL_PERIOD)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["Teacher ID"] == teacher_id) &
                     (df["Project Posted Date"] < posted_date_minus_one) &
                     (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    # print(df_filtered)
    # print(df_filtered_row_count)
    # print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    if df_filtered_row_count > 0:
        success_rate = success_filtered_count/df_filtered_row_count
        success_rate_imputed = 0

    return success_rate, success_rate_imputed

# Function to create the teacher success rate for the past 4 months


def create_teacher_success_rate_feature(df, x):

  # Get length of dataframe
    df_len = df.shape[0]
    teacher_success_rate_values = []
    teacher_success_rate_imputed_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        success_rate, success_rate_imputed = calculate_teacher_sucess_rate(
            row, df, x)
        # print(f'Teacher success rate = {success_rate}')
        teacher_success_rate_values.append(success_rate)
        teacher_success_rate_imputed_values.append(success_rate_imputed)

    df["Teacher Success Rate"] = teacher_success_rate_values
    df["Teacher Success Rate Imputed"] = teacher_success_rate_imputed_values

    return df


def calculate_school_city_sucess_rate(row_number, df, x):

    success_rate = 0
    success_rate_imputed = 1
    # Store School City and posted date
    row = df.iloc[row_number, :]
    posted_date_minus_one = row["Project Posted Date"] - \
        DateOffset(months=config.LABEL_PERIOD)
    school_city = row["School City"]
    posted_date_minus_x = row["Project Posted Date"] - \
        DateOffset(months=x+config.LABEL_PERIOD)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School City"] == school_city) &
                     (df["Project Posted Date"] < posted_date_minus_one) &
                     (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    # print(df_filtered)
    # print(df_filtered_row_count)
    # print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    if df_filtered_row_count > 0:
        success_rate = success_filtered_count/df_filtered_row_count
        success_rate_imputed = 0

    return success_rate, success_rate_imputed


def create_school_city_success_rate_feature(df, x):

    # Get length of dataframe
    df_len = df.shape[0]
    school_success_rate_values = []
    school_success_rate_imputed_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        success_rate, success_rate_imputed = calculate_school_city_sucess_rate(
            row, df, x)
        # print(f'School city success rate = {success_rate}')
        school_success_rate_values.append(success_rate)
        school_success_rate_imputed_values.append(success_rate_imputed)

    df["School City Success Rate"] = school_success_rate_values
    df["School City Success Rate Imputed"] = school_success_rate_imputed_values

    return df


def calculate_school_success_rate(row_number, df, x):

    success_rate = 0
    success_rate_imputed = 1
    # Store school ID and posted date
    row = df.iloc[row_number, :]
    posted_date_minus_one = row["Project Posted Date"] - \
        DateOffset(months=config.LABEL_PERIOD)
    school_id = row["School ID"]
    posted_date_minus_x = row["Project Posted Date"] - \
        DateOffset(months=x+config.LABEL_PERIOD)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School ID"] == school_id) &
                     (df["Project Posted Date"] < posted_date_minus_one) &
                     (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    # print(df_filtered)
    # print(df_filtered_row_count)
    # print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    if df_filtered_row_count > 0:
        success_rate = success_filtered_count/df_filtered_row_count
        success_rate_imputed = 0

    return success_rate, success_rate_imputed


def create_school_success_rate_feature(df, x):

    # Get length of dataframe
    df_len = df.shape[0]
    school_success_rate_values = []
    school_success_rate_imputed_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        success_rate, success_rate_imputed = calculate_school_success_rate(
            row, df, x)
        # print(f'School success rate = {success_rate}')
        school_success_rate_values.append(success_rate)
        school_success_rate_imputed_values.append(success_rate_imputed)

    df["School Success Rate"] = school_success_rate_values
    df["School Success Rate Imputed"] = school_success_rate_imputed_values

    return df


def calculate_school_county_sucess_rate(row_number, df, x):

    success_rate = 0
    success_rate_imputed = 1
    # Store School County and posted date
    row = df.iloc[row_number, :]
    posted_date_minus_one = row["Project Posted Date"] - \
        DateOffset(months=config.LABEL_PERIOD)
    school_county = row["School County"]
    posted_date_minus_x = row["Project Posted Date"] - \
        DateOffset(months=x+config.LABEL_PERIOD)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School County"] == school_county) &
                     (df["Project Posted Date"] < posted_date_minus_one) &
                     (df["Project Posted Date"] >= posted_date_minus_x)]
    df_filtered_row_count = df_filtered.shape[0]
    # print(df_filtered)
    # print(df_filtered_row_count)
    # print(df_filtered.shape)

    success_filtered_count = df_filtered[df_filtered["Label"] == 0].shape[0]

    if df_filtered_row_count > 0:
        success_rate = success_filtered_count/df_filtered_row_count
        success_rate_imputed = 0

    return success_rate, success_rate_imputed


def create_school_county_success_rate_feature(df, x):

    # Get length of dataframe
    df_len = df.shape[0]
    school_success_rate_values = []
    school_success_rate_imputed_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        success_rate, success_rate_imputed = calculate_school_county_sucess_rate(
            row, df, x)
        # print(f'School county success rate = {success_rate}')
        school_success_rate_values.append(success_rate)
        school_success_rate_imputed_values.append(success_rate_imputed)

    df["School County Success Rate"] = school_success_rate_values
    df["School County Success Rate Imputed"] = school_success_rate_imputed_values

    return df


# for the number of projects in a certain state
def calculate_project_count(row_number, df, x):

    row = df.iloc[row_number, :]
    posted_date_minus_one = row["Project Posted Date"] - \
        DateOffset(months=config.LABEL_PERIOD)
    school_state = row["School State"]
    posted_date_minus_x = row["Project Posted Date"] - \
        DateOffset(months=x+config.LABEL_PERIOD)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School State"] == school_state) &
                     (df["Project Posted Date"] < posted_date_minus_one) &
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

# for the number of projects in a certain city
def calculate_project_count_city(row_number, df, x):

    row = df.iloc[row_number, :]
    posted_date_minus_one = row["Project Posted Date"] - \
        DateOffset(months=config.LABEL_PERIOD)
    school_city = row["School City"]
    posted_date_minus_x = row["Project Posted Date"] - \
        DateOffset(months=x+config.LABEL_PERIOD)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School City"] == school_city) &
                     (df["Project Posted Date"] < posted_date_minus_one) &
                     (df["Project Posted Date"] >= posted_date_minus_x)]

    # Find the number of projects from that city
    df_filtered_row_count = df_filtered.shape[0]

    return df_filtered_row_count

def create_projects_in_a_city_feature(df, x):

    # Get length of dataframe
    df_len = df.shape[0]
    number_of_projects_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        project_count = calculate_project_count_city(row, df, x)
        # print(f'Project count = {project_count}')
        number_of_projects_values.append(project_count)

    df["Project Count in City"] = number_of_projects_values

    return df

# for the number of projects in a certain county
def calculate_project_count_county(row_number, df, x):

    row = df.iloc[row_number, :]
    posted_date_minus_one = row["Project Posted Date"] - \
        DateOffset(months=config.LABEL_PERIOD)
    school_county = row["School County"]
    posted_date_minus_x = row["Project Posted Date"] - \
        DateOffset(months=x+config.LABEL_PERIOD)
    # print(posted_date, posted_date_minus_x)

    # Filter the df
    df_filtered = df[(df["School County"] == school_county) &
                     (df["Project Posted Date"] < posted_date_minus_one) &
                     (df["Project Posted Date"] >= posted_date_minus_x)]

    # Find the number of projects from that county
    df_filtered_row_count = df_filtered.shape[0]

    return df_filtered_row_count

def create_projects_in_a_county_feature(df, x):

    # Get length of dataframe
    df_len = df.shape[0]
    number_of_projects_values = []
    for row in range(df_len):
        # print(f'Row number = {row} --------------------------------------------------------')
        project_count = calculate_project_count_county(row, df, x)
        # print(f'Project count = {project_count}')
        number_of_projects_values.append(project_count)

    df["Project Count in County"] = number_of_projects_values

    return df


# To add the length features
def add_length_features(df):
    df["Project Essay Length"] = np.where(df["Project Essay"].isnull(), 0, df["Project Essay"].str.split().str.len())
    df["Project Need Statement Length"] = np.where(df["Project Need Statement"].isnull(), 0, df["Project Need Statement"].str.split().str.len())
    df["Project Short Description Length"] = np.where(df["Project Short Description"].isnull(), 0, df["Project Short Description"].str.split().str.len())

    return df


# To perform TFIDF
def perform_tfidf(x_train, x_test):
    # Create corpus
    
    vectorizer = TfidfVectorizer()

    return x_train, x_test


# Function to add new features that are not static
def add_new_features(df):

    # Add the teacher success rate column
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Adding teacher success rate column.')
    modified_df_tsr = create_teacher_success_rate_feature(df, 4)
    print("done modified_df_tsr")
    # Add the school city success rate
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Adding school city success rate column.')
    modified_df_scr = create_school_city_success_rate_feature(
        modified_df_tsr, 4)
    print("done modified_df_scr")
    # Add the school id sucess rate
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Adding school success rate column.')
    modified_df_ssr = create_school_success_rate_feature(modified_df_scr, 4)
    print("done modified_df_ssr")
    # Add the school county success rate
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Adding school county success rate column.')
    modified_df_sctr = create_school_county_success_rate_feature(
        modified_df_ssr, 4)
    print("done modified_df_sctr")
    # Add the number of projects in a state for a selected period of time
    log_intermediate_output_to_file(config.INFO_DEST, config.PROGRAM_LOG_FILE,
                                    'Adding number of projects in a state for a selected period of time.')
    modified_df_project_count = create_projects_in_a_state_feature(
        modified_df_sctr, 4)
    print("done modified_df_project_count")
    # Add the number of projects in a city for a selected period of time
    log_intermediate_output_to_file(config.INFO_DEST, config.PROGRAM_LOG_FILE,
                                    'Adding number of projects in a city for a selected period of time.')
    modified_df_project_count_city = create_projects_in_a_city_feature(
        modified_df_project_count, 4)
    print("done modified_df_project_count_city")
    # Add the number of projects in a county for a selected period of time
    log_intermediate_output_to_file(config.INFO_DEST, config.PROGRAM_LOG_FILE,
                                    'Adding number of projects in a county for a selected period of time.')
    modified_df_project_count_county = create_projects_in_a_county_feature(
        modified_df_project_count_city, 4)
    print("done modified_df_project_count_county")

    # Add the essay, need statement, description length columns
    log_intermediate_output_to_file(config.INFO_DEST, config.PROGRAM_LOG_FILE, 
                                    'Adding the Project Essay, Need Statement, Description length')
    modified_df_length_features = add_length_features(modified_df_project_count_county)
    

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Done adding new features.')

    return modified_df_length_features

