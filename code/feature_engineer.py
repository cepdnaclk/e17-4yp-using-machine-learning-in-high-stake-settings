from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pandas.core.frame import DataFrame
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import language_tool_python
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
import data_processor as dp

lang_tool = language_tool_python.LanguageTool('en-US')
nltk.download('stopwords')


def standardize_data(x_train, x_test, cols_list):
    """Function for scaling after seperating into train/test."""
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


def create_features(data: DataFrame):
    pass


def add_statement_grammertical_error_feature(data: DataFrame):
    # Creates a new feature called text size to error ratio
    data["Statement Error Ratio"] = len(lang_tool.check(
        str(data["Project Need Statement"]))) / len(str(data["Project Need Statement"]).split())
    return data



def add_title_essay_relativity_score(data: DataFrame):
    # Create a new feature that has the relatedness of title and essay
    # using cosine similarity
    stop_words = set(stopwords.words('english'))
    # Preprocess topic and essay
    data["_Title"] = ' '.join(
        [word.lower() for word in str(data["Project Title"]).split() if word.lower() not in stop_words])
    data["_Essay"] = ' '.join(
        [word.lower() for word in str(data["Project Essay"]).split() if word.lower() not in stop_words])

    vectorizer = TfidfVectorizer()

    # Calculate TF-IDF vectors & cosine similarity
    data["Title Essay Relativity"] = cosine_similarity(
        vectorizer.fit_transform([data["_Title"], data["_Essay"]])[0],
        vectorizer.fit_transform([data["_Title"], data["_Essay"]])[1]
    )[0][0]
    data.drop("_Essay")
    data.drop("_Title")
    return data


def add_desc_essay_relativity_score(data: DataFrame):
    # Create a new feature that has the relatedness of description and essay
    # using cosine similarity
    stop_words = set(stopwords.words('english'))
    # Preprocess desc and essay
    data["_Description"] = ' '.join(
        [word.lower() for word in str(data["Project Short Description"]).split() if word.lower() not in stop_words])
    data["_Essay"] = ' '.join(
        [word.lower() for word in str(data["Project Essay"]).split() if word.lower() not in stop_words])

    vectorizer = TfidfVectorizer()

    # Calculate TF-IDF vectors & cosine similarity
    data["Title Essay Relativity"] = cosine_similarity(
        vectorizer.fit_transform([data["_Description"], data["_Essay"]])[0],
        vectorizer.fit_transform([data["_Description"], data["_Essay"]])[1]
    )[0][0]
    data.drop("_Essay")
    data.drop("_Description")
    return data


def label_data_1(data: DataFrame, threshold: float, select_cols: list):
    # Create new features by aggregating
    data["Total Donations"] = data.groupby("Project ID")["Donation Amount"] \
                                                        .transform("sum")

    data["Donation to Cost"] = data["Donation Amount"] / data["Project Cost"]

    data["Fund Ratio"] = data.groupby("Project ID")["Donation to Cost"] \
                                                        .transform("sum")

    data["Label"] = data.apply(
        lambda x : 0  if x["Fund Ratio"] < threshold  else 1, axis=1)
    select_cols = select_cols + ["Label", "Project Posted Date"]
    return data[select_cols].drop_duplicates()


def label_data(data: DataFrame, threshold: float, select_cols: list):
    data["Posted Date to Donation Date"] = data["Donation Received Date"] \
                                                 - data["Project Posted Date"]
    data["Posted Date to Donation Date"] = data["Posted Date to Donation Date"] \
                                                    / np.timedelta64(1, 'D')

    data = data[data["Posted Date to Donation Date"] < config.DONATION_PERIOD]

    data["Total Donations In The Period"] = data.groupby(
                            "Project ID")["Donation Amount"].transform("sum")
    data["Fund Ratio"] = np.where(
        data["Project Cost"] > 0, 
        data["Total Donations In The Period"] / data["Project Cost"], 1)

    data["Label"] = data.apply(
        lambda x : 0  if x["Fund Ratio"] < threshold  else 1, axis=1)
    select_cols = select_cols + ["Label", "Project Posted Date"]

    return data[select_cols].drop_duplicates()


def run_pipeline(data, model):
    # Initiate lists to store data
    t_current_list = []
    t_current_accuracy = []

    # Initiate timing variables
    max_t = pd.Timestamp(config.MAX_TIME)
    min_t = pd.Timestamp(config.MIN_TIME)
    time_period = timedelta(days=config.EVAL_PERIOD_DAYS)  
    training_window = timedelta(days=config.TRAINING_WINDOW)

    t_current = min_t
    print("1================", t_current, max_t, training_window)
    data = dp.encode_data(data, config.CATEGORICAL_COLS)
    print("encoded_data.shape = ", data.shape)


    while(t_current < max_t - training_window):

        t_current_list += [t_current]
        t_start = t_current
        t_end = t_current + training_window
        t_filter = t_end - time_period

        # Filter rows for the relevant time period
        data_window = data[
            data["Project Posted Date"] < pd.to_datetime(t_end)]
        data_window = data_window[
            data_window["Project Posted Date"] > pd.to_datetime(t_start)]
        
        print("final_data.shape = ", data_window.shape)

        x_train, y_train, x_test, y_test = dp.split_time_series_train_test_data(
            data=data_window, filter_date=t_filter)

        # Scaling
        x_train, x_test = standardize_data(x_train, x_test, config.VARIABLES_TO_SCALE)
        print(x_test.shape)
        print(x_train.shape)
        model = model.fit(x_train, y_train.values.ravel())
        y_hat = model.predict(x_test)
        y_pred = model.predict(x_train)

        # Evaluate
        cm = confusion_matrix(y_test, y_hat)
        sns.heatmap(cm, square=True, annot=True, cbar=False)
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        plt.savefig(config.IMAGE_DEST + f"Confusion matrix for {str(t_current)[:10]}")
        plt.clf()

        print("==============================================================================")
        print("Prediction evaluation scores for training: ")
        print(classification_report(y_train, y_pred, output_dict=True))


        print("Prediction evaluation scores for testing: ")
        print(classification_report(y_test, y_hat, output_dict=True))
        print("==============================================================================")
        t_current = t_current + training_window
    
    return model
        

