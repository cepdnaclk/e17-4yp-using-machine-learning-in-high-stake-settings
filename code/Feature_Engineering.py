#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/DrMadhushan/e17-fyp-xai/blob/master/Feature_Engineering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# get_ipython().system('pip install mlxtend')
import joblib
import sys

# sys.modules['sklearn.externals.joblib'] = joblib


import os
import json
import random 
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm



# Relevant functions
# Function for one hot encoding in a specific time window
def one_hot_encoding_columns(df, cat_cols_with_nan):

  # For the categorical variables with one/more value(s) for each row
  # print("One hot encoding")
  # print("cat_cols_with_nan = ", cat_cols_with_nan)
  # print(df.columns)
  # print("df.columns size = ", len(df.columns))
  # modified_df = df.copy()
  # for col in cat_cols_with_nan:
    # print(f"{str(col)}")
    # print(df[col])
    # print(df[col].str)
    # print(df[col].str.get_dummies(sep=', '))
    # print(df[col].str.get_dummies(sep=', ').add_prefix(col + " "))
    # modified_df = pd.concat(
    #     [df, df[col].str.get_dummies(sep=', ').add_prefix(col + " ")], axis=1
    #      )
    # |  |  |  |  |
  modified_df = pd.get_dummies(df, columns=cat_cols_with_nan)
  # print(modified_df.columns)
  # print("modified_df.columns size = ", len(modified_df.columns))

  # modified_df = modified_df.drop(cat_cols_with_nan, axis=1)  
  # print(modified_df.columns)
  # print("modified_df.columns size = ", len(modified_df.columns))

  return modified_df





# Function for scaling after seperating into train/test
def standardize_data(x_train, x_test, cols_list):

  # Create scaler
  ss = StandardScaler()
  features_train = x_train[cols_list]
  features_test = x_test[cols_list]

  # Fit scaler only for training data
  scaler = ss.fit(features_train)

  # Transform training data
  features_t = scaler.transform(features_train)
  x_train[cols_list] = features_t

  # Transforming test data
  for_test = scaler.transform(features_test)
  x_test[cols_list] = for_test

  return x_train, x_test





# Function for feature selection
def run_sfs(x, y):

  # Define classifier
  knn = KNeighborsClassifier(n_neighbors=4)
  sfs1 = sfs(knn, 
             k_features="best", 
             forward=True, 
             floating=False, 
             verbose=2,
             scoring='accuracy',
             cv=5)
  sfs1 = sfs1.fit(x, y)
  feature_names = list(sfs1.k_feature_names_)
  print(feature_names)
  print(sfs1.k_score_)

  return feature_names




def run_model_with_gridsearch(model, parameters, x_train, y_train, x_test, y_test, scoring):
  # Initiate model
  # model = GridSearchCV(classifier, parameters, cv=5, scoring=scoring)
  # Fit the model on training data
  # print("x_train shape = ", x_train.shape)
  # print("y_train shape = ", y_train.shape)
  model.fit(x_train, y_train)

  # print("Best parameters: ", model.best_params_)
  # print("Scoring: ", model.best_score_)

  # Predict for test data and train data
  y_hat = model.predict(x_test)
  y_pred = model.predict(x_train)
  # print("Classifier params 12 = ", classifier.coef_)

  return y_hat, model, y_pred




# Test pipeline function
def run_pipeline(data, model, parameters, time_period_in_days, fund_ratio_threshold, categorical_variables, variables_to_scale):

  # Initiate lists to store data
  t_current_list = []
  t_current_accuracy = []

  # Create new features
  data["Posted Date to Donation Date"] = data.loc[:,"Donation Received Date"] - data.loc[:,"Project Posted Date"]
  data["Posted Date to Donation Date"] = data["Posted Date to Donation Date"] / np.timedelta64(1, 'D')

  # Initiate variables
  max_t = pd.Timestamp("2013-12-01 00:00:00") #"2018-05-01 00:00:00"
  min_t = pd.Timestamp("2013-01-01 00:00:00")
  time_period = timedelta(days=time_period_in_days)  
  time_posted_window = timedelta(days=120)
  t_current = min_t - time_period + time_posted_window # min-t + three months

  print("t_current = ", t_current)
  print("time_period = ", time_period)
  print("max_t = ", max_t)
  print("min_t = ", min_t)
  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>  ", data.shape)
  # |   |    |    |     |     |

  # Start cohort
  while(t_current < max_t - time_period):

    # Adjust the current time
    t_current = t_current + time_period
    print("Current date: ", t_current)
    print("===============================================================================")
    t_current_list += [t_current]
    t_start = t_current - time_posted_window # t_start = min_t
    t_end = t_current - time_period

    # Filter rows for the relevant time period
    data_window = data[data["Project Posted Date"] < pd.to_datetime(t_current)]
    data_window = data_window[data_window["Project Posted Date"] > pd.to_datetime(t_start)]
    data_window = data_window[data_window["Posted Date to Donation Date"] < time_period_in_days]

    # Create new features
    data_window["Donation to Cost"] = data_window["Donation Amount"] / data_window["Project Cost"]

    # Create new features by aggregating
    data_window["Total Donations"] = data_window.groupby("Project ID")["Donation Amount"].transform("sum")
    data_window["Number of Donors"] = data_window.groupby("Project ID")["Donor ID"].transform("count")
    data_window["Fund Ratio"] = data_window.groupby("Project ID")["Donation to Cost"].transform("sum")

    data_window["Teacher is Donor Coded"] = np.where(data_window["Donor Is Teacher"] == "Yes", 1, 0)
    data_window["Teacher is Donor Count"] = data_window.groupby("Project ID")["Teacher is Donor Coded"].transform("sum")
    data_window["Teacher is Donor Ratio"] = data_window["Teacher is Donor Count"] / data_window["Number of Donors"]

    data_window["Donation is Optional Coded"] = np.where(data_window["Donation Included Optional Donation"] == "Yes", 1, 0)
    data_window["Donation is Optional Count"] = data_window.groupby("Project ID")["Donation is Optional Coded"].transform("sum")
    data_window["Donation is Optional Ratio"] = data_window["Donation is Optional Count"] / data_window["Number of Donors"]

    data_window["Donor State Most Frequent"] = data_window.groupby("Project ID")["Donor State"].transform(lambda x: x.value_counts().idxmax())
    # print("Checkpoint 7")

    # Create label
    data_window["Label"] = data_window.apply(lambda x : 0  if x["Fund Ratio"] < fund_ratio_threshold  else 1, axis=1)
    # print("Checkpoint 8")

    # Filter for columns and remove duplicate rows
    model_vars = ["Project ID", "Total Donations", "Number of Donors", "Fund Ratio", "Project Type", "Project Subject Category Tree", 
                  "Project Subject Subcategory Tree", "Project Grade Level Category", "Project Resource Category", "Donor State Most Frequent", 
                  "School Metro Type", "School Percentage Free Lunch", "School State", "School County", 
                  "Teacher Prefix", "Teacher is Donor Ratio", "Donation is Optional Ratio"]
    cols_to_select = model_vars + ["Label", "Project Posted Date"]
    data_window_final = data_window[cols_to_select].drop_duplicates()
    # print("Checkpoint 9")
    print("data_window_final shape = ", data_window_final.shape)
    # One hot encoding - gets new columns
    data_window_final = one_hot_encoding_columns(data_window_final, categorical_variables)
    # print("Checkpoint 10")
    # print(data_window_final.head())

    # Create training set
    # train_set, test_set = train_test_split(data_window_final, test_size=0.2)
    # data_window_final = data_window_final.drop(["Project ID", "Project Posted Date"], axis=1)

    # print(data_window_final.head())
    # train_set, test_set = split_train_test_set_as_df(data_window_final)

    train_set = data_window_final[data_window_final["Project Posted Date"] < pd.to_datetime(t_end)].drop(["Project ID", "Project Posted Date"], axis=1)
    x_train = train_set.loc[:, train_set.columns != "Label"]
    y_train = train_set.loc[:, ["Label"]]
    # print("Training set columns :", list(train_set.columns.values))

    # Create testing set
    test_set = data_window_final[data_window_final["Project Posted Date"] >= pd.to_datetime(t_end)].drop(["Project ID", "Project Posted Date"], axis=1)
    x_test = test_set.loc[:, test_set.columns != "Label"]
    y_test = test_set.loc[:, ["Label"]]

    # Scaling
    x_train, x_test = standardize_data(x_train, x_test, variables_to_scale)

    # Run SFS

    
    # print("x_train shape = ", x_train.shape)
    # print("y_train shape = ", y_train.shape)
    # Run model
    # print("Classifier params 01 = ", model.coef_)
    # if model_post_run:
    #   model = model_post_run
    y_hat, model, y_pred = run_model_with_gridsearch(
          model,
          parameters,
          x_train,
          y_train.values.ravel(),
          x_test,
          y_test.values.ravel(),
          "accuracy")
    # print("Classifier params 02 = ", model.coef_)
    # print("Classifier params 01 = ", model.intercept_)
    

    # Evaluate
    cm = confusion_matrix(y_test, y_hat)
    sns.heatmap(cm, square=True, annot=True, cbar=False)
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')
    plt.savefig(f"Confusion matrix for {str(t_current)[:10]}")
    plt.clf()
    # plt.savefig("Confusion matrix for " + str(t_current))

    print("Prediction evaluation scores for training: ")
    print(classification_report(y_train, y_pred, output_dict=True))


    print("Prediction evaluation scores for testing: ")
    print(classification_report(y_test, y_hat, output_dict=True))
    print("===============================================================================")

  return t_current_list, t_current_accuracy, model

def split_train_test_set_as_df(data_frame, train_set=0.8, random=True, drop_fields=None):
  """_summary_

  Args:
      data_frame (DataFrame): The dataset as pandas data frame
      train_set (float, optional): Size of the training set. Defaults to 0.8.
      random (bool, optional): Randomize flag. Defaults to True.
      drop_fields (list<str>, optional): Column names to drop. Defaults to None.

  Returns:
      DataFrame, DataFrame: Training set and testing set
  """
  size = data_frame.shape[0]
  test_size = int(size * train_set)
  if random:
    data_frame = data_frame.sample(frac=1)
  train_set = data_frame[:test_size]
  test_set = data_frame[test_size:]
  
  return train_set, test_set

########################################################################################################



data = pd.read_csv('./data/DsDnsPrScTch.csv', nrows=10000) # Set path
print(data.shape)
# print("Checkpoint 1")
# print(data.dtypes)
# Check for duplicates
data[data.duplicated()]
# print(data.columns)
# data.head()
data.isnull().sum()


# Imputing missing values:
# When missing values is from categorical columns (string or numerical) then the missing values can be 
# replaced with the most frequent category. 
# If the number of missing values is very large then it can be replaced with a new category.
# 
# - Donor City (346083)- New category
# - Donor Zip (267808) - New category
# - Project Title (34) - Drop column 
# - project Essay (10) - Drop column
# - Project Short Description (19) - Drop column
# - Project Need Statement (2) - Drop column
# 
# - Project Subject Category Tree (111) - 
# - Project Subject Subcategory Tree (111)
# For these two columns, we can seperate the comma-seperated categories and create new binary columns
# 
# - Project Resource Category (134) - Can use the most frequent item
# - Project Expiration Date (3) - https://www.analyticsvidhya.com/blog/2023/02/impute-missing-dates-not-data-in-python/
# 
# - School Percentage Free Lunch (21894) - Mean/Median
# - School City (30146) - New category
# - School County (37) - Most frequent
# - Teacher Prefix (139) - Most frequent
# 
# 
# New possible features (granularity - Project ID):
# 
# - Number of donations made during the selected time window
# - Number of donors in the selected time window
# - Most frequent donor city/state in the selected time window
# - Donor is teacher yes/no count in the selected time window


########################################################################################################
# print("Checkpoint 2")

# Data preprocessing

# Set data types for datetime
data["Teacher First Project Posted Date"]=pd.to_datetime(data["Teacher First Project Posted Date"])
data["Project Fully Funded Date"]=pd.to_datetime(data["Project Fully Funded Date"])
data["Project Expiration Date"]=pd.to_datetime(data["Project Expiration Date"])
data["Project Posted Date"]=pd.to_datetime(data["Project Posted Date"])
data["Donation Received Date"]=pd.to_datetime(data["Donation Received Date"])
data["Number of dates since posted"]=data.loc[:,"Donation Received Date"] - data.loc[:,"Project Posted Date"]

########################################################################################################
# print("Checkpoint 3")

# Imputing missing data
data["Donor City"] = data["Donor City"].fillna("Unknown")
data["Donor Zip"] = data["Donor Zip"].fillna("Unknown")

data["School Percentage Free Lunch"] = data["School Percentage Free Lunch"].replace(np.NaN, data["School Percentage Free Lunch"].median())

data["School City"] = data["School City"].fillna("Unknown")
data["School County"] = data["School County"].fillna(data["School County"].mode()[0])
data["Teacher Prefix"] = data["Teacher Prefix"].fillna(data["Teacher Prefix"].mode()[0])

# print("Checkpoint 4")

########################################################################################################

# Define models and parameters
classifier_1 = LogisticRegression()
parameters_1 = {"penalty":["l1","l2"]}

classifier_2 = RandomForestClassifier()
parameters_2 = {'max_depth':[2, 3, 4], 
                'n_estimators':[5, 10, 20], 
                'min_samples_split': 2, 
                'min_samples_leaf': 2}

classifier_3 = svm.SVC(kernel='linear')

# print("Checkpoint 5")

########################################################################################################

# Define column lists
categorical_variables = ["Project Type", "Project Subject Category Tree", "Project Subject Subcategory Tree", 
                         "Project Grade Level Category", "Project Resource Category", "Donor State Most Frequent", 
                         "Donor State Most Frequent", "School Metro Type", "School State", "School County", 
                         "Teacher Prefix"]
variables_to_scale = ["Total Donations", "Number of Donors", "Fund Ratio", "School Percentage Free Lunch", 
                      "Teacher is Donor Ratio", "Donation is Optional Ratio"]
# Run and evaluate
t_current_list, t_current_accuracy, trained_model = run_pipeline(data, classifier_1, parameters_1, 
30, 0.6, categorical_variables, variables_to_scale)
