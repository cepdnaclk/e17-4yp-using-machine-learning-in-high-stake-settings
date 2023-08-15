import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sys

import feature_engineer as fe
import data_processor as dp
import config
from helper import (
    save_model,
    load_model,
    create_dirs,
    create_classification_models,
    create_logistic_regression_parameters,
    create_random_forest_parameters)
import temporal_features as tmpf


# create classifiers including baseline models
rand_for_params = create_random_forest_parameters()
log_reg_params = create_logistic_regression_parameters()
models = create_classification_models(
    random_forest_parameters_list=rand_for_params,
    logistic_regression_parameters_list=log_reg_params
)
# create dirs that not exist
model_names = [model.get("model_name") for model in models]
create_dirs(models=model_names)  # can pass a list of specific model names

print("Start data pre processing")
data = dp.load_data_to_df(config.DATA_SOURCE, rows=config.MAX_ROWS)

data = dp.set_data_types_to_datetime(data, config.DATE_COLS)

data = dp.impute_data(data)
print("Complete imputing = ", data.shape)

data = fe.label_data(data, config.THRESHOLD_RATIO)
print("Complete labelling, shape = ", data.shape)

# filter training features
extra_features_required = ["Teacher ID", "School ID"]
data = data[config.TRAINING_FEATURES + ["Label"] + extra_features_required]
print("Filtered training Features, shape = ", data.shape)

# Adding new features
data = tmpf.add_new_features(data)
print(data.columns)
data = data.drop(extra_features_required, axis=1)
print("After adding new Features, shape = ", data.shape)
print(data.columns)

# export labelled data to csv
time = datetime.datetime.now()
file_path = config.DATA_DEST + \
    f"labelled_data - {str(time.strftime('%Y-%m-%d %H:%M:%S'))[:10]}.csv"
dp.export_data_frame(data=data, path=file_path)

data_1 = dp.encode_data(data, config.CATEGORICAL_COLS)
print("encoded_data.shape = ", data_1.shape)

# load_processed_data = True
# if loa

model_eval_metrics = {}

for model_item in models:
    print(f"Classifier -> {model_item.get('model_name')}")
    trained_model, eval_metrics = fe.run_pipeline(
        data=data_1, model=model_item
    )
    print("k_fixed_precisions = ", eval_metrics.get("k_fixed_precision"))
    fe.plot_precision_for_fixed_k(
        eval_metrics, model_item.get("model_name")+"/")
    model_eval_metrics.update({model_item.get("model_name"): eval_metrics})

# fe.plot_k_fold_evaluation_metrics(eval_metrics, 'random_forest/')
# # save_model(config.MODEL_DEST, file_name=f'RandForest_{str(time.strftime("%Y-%m-%d %H:%M:%S"))[:10]}.sav', model=trained_model)

# print(model_eval_metrics)

fe.plot_precision_for_fixed_k_for_multiple_models(
    model_names, model_eval_metrics)
