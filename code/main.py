import pandas as pd
import datetime
import seaborn as sns
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pickle

import feature_engineer as fe
import data_processor as dp
import config

def save_model(file_name, model):
    file_path = config.MODEL_DEST + file_name
    pickle.dump(model, file=open(file_path, "wb"))

def load_model(model_file_path):
    return pickle.load(open(model_file_path, 'rb'))

print("Start data pre processing")
data = dp.load_data_to_df(config.DATA_SOURCE, rows=config.MAX_ROWS)
print("shape 1 = ", data.shape)
data = dp.set_data_types_to_datetime(data, config.DATE_COLS)
print("shape 2 = ", data.shape)
data = dp.impute_data(data)
print("Complete imputing = ", data.shape)
print("shape 3 = ", data.shape)
data = fe.create_features(data)
print("shape 3 = ", data.shape)
print("columns = ", data.columns)

# Define models and parameters
classifier_1 = LogisticRegression()
parameters_1 = {"penalty":["l1","l2"]}

# classifier_2 = RandomForestClassifier(warm_start=True)
# parameters_2 = {'max_depth':[2, 3, 4], 
#                 'n_estimators':[5, 10, 20], 
#                 'min_samples_split': 2, 
#                 'min_samples_leaf': 2}

# classifier_3 = svm.SVC(kernel='linear')

data = fe.label_data(data, config.THRESHOLD_RATIO, config.TRAINING_FEATURES)
print("shape 4 = ", data.shape)
# export labelled data to csv
time = datetime.datetime.now()
file_path = config.DATA_DEST + f"labelled_data - {str(time.strftime('%Y-%m-%d %H:%M:%S'))[:10]}.csv"
dp.export_data_frame(data=data, path=file_path)
trained_model = fe.run_pipeline(data, classifier_1)
save_model(file_name=f'LogReg_{str(time.strftime("%Y-%m-%d %H:%M:%S"))[:10]}.sav', model=trained_model)
