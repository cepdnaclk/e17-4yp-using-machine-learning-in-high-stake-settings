import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import feature_engineer as fe
import data_processor as dp
import config
from helper import save_model, load_model

print("Start data pre processing")
data = dp.load_data_to_df(config.DATA_SOURCE, rows=config.MAX_ROWS)

data = dp.set_data_types_to_datetime(data, config.DATE_COLS)

data = dp.impute_data(data)
print("Complete imputing = ", data.shape)

data = fe.label_data(data, config.THRESHOLD_RATIO)
print("Complete labelling, shape = ", data.shape)

# filter training features
data = data[config.TRAINING_FEATURES + ["Label"]]
print("Filtered training Features, shape = ", data.shape)

# export labelled data to csv
time = datetime.datetime.now()
file_path = config.DATA_DEST + f"labelled_data - {str(time.strftime('%Y-%m-%d %H:%M:%S'))[:10]}.csv"
dp.export_data_frame(data=data, path=file_path)

# Define models and parameters
classifier_1 = RandomForestClassifier(n_estimators=500)
# parameters_1 = {'max_depth':[2, 3, 4], 
#                 'n_estimators':[5, 10, 20], 
#                 'min_samples_split': 2, 
#                 'min_samples_leaf': 2}

classifier_2 = LogisticRegression()
# parameters_2 = {"penalty":["l1","l2"]}


classifier_3 = DecisionTreeClassifier(max_depth=3)
classifier_4 = SVC(kernel='linear', probability=True)


data_1 = dp.encode_data(data, config.CATEGORICAL_COLS)
print("encoded_data.shape = ", data_1.shape)

data_2 = data_1.copy(deep=True)
data_3 = data_1.copy(deep=True)
data_4 = data_1.copy(deep=True)

print("Classifier: Random Forest")
trained_model, eval_metrics, avg_metrics = fe.run_pipeline(data_1, classifier_1, 'random_forest/')
fe.plot_k_fold_evaluation_metrics(eval_metrics, 'random_forest/')
fe.plot_precision_for_fixed_k(eval_metrics, 'random_forest/')
# save_model(config.MODEL_DEST, file_name=f'RandForest_{str(time.strftime("%Y-%m-%d %H:%M:%S"))[:10]}.sav', model=trained_model)


# print("Classifier: Logistic Regression")
# trained_model, eval_metrics, avg_metrics = fe.run_pipeline(data_2, classifier_2, 'log_reg/')
# fe.plot_k_fold_evaluation_metrics(eval_metrics, 'log_reg/')
# fe.plot_precision_for_fixed_k(eval_metrics, 'log_reg/')
# save_model(config.MODEL_DEST, file_name=f'LogReg_{str(time.strftime("%Y-%m-%d %H:%M:%S"))[:10]}.sav', model=trained_model)


# print("Classifier: Decision Tree")
# trained_model, eval_metrics, avg_metrics = fe.run_pipeline(data_3, classifier_3, 'decision_tree/')
# fe.plot_k_fold_evaluation_metrics(eval_metrics, 'decision_tree/')
# fe.plot_precision_for_fixed_k(eval_metrics, 'decision_tree/')
# save_model(config.MODEL_DEST, file_name=f'DecTree_{str(time.strftime("%Y-%m-%d %H:%M:%S"))[:10]}.sav', model=trained_model)

# print("Classifier: SVM")
# trained_model, eval_metrics, avg_metrics = fe.run_pipeline(data_4, classifier_4, 'svm/')
# fe.plot_k_fold_evaluation_metrics(eval_metrics, 'svm/')
# fe.plot_precision_for_fixed_k(eval_metrics, 'svm/')
# save_model(config.MODEL_DEST, file_name=f'SVM_{str(time.strftime("%Y-%m-%d %H:%M:%S"))[:10]}.sav', model=trained_model)
