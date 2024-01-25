import pickle
import config
import os
import json
from typing import Union
from datetime import datetime as dt
import pandas as pd
from datetime import timedelta

import xgboost
import lightgbm as lgb
import joblib
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Input


def save_model(path, file_name, model, model_type):
    # Create path + file name
    file_path = path + file_name
    # Save
    if model_type == "nn":
        model.save(file_path)
    else:
        #pickle.dump(model, file=open(file_path, "wb"))
        joblib.dump(model, file_path)


def load_model(model_file_path):
    #return pickle.load(open(model_file_path, 'rb'))
    return joblib.load(model_file_path)


def log_intermediate_output_to_file(path, file_name, log_info: Union[list, dict, str]):
    file_path = path + file_name
    json_data = json.dumps(log_info, indent=2)
    time = dt.now()
    with open(file_path, 'a') as file:
        file.write(
            f"\nlog {str(time.strftime('%Y-%m-%d %H:%M:%S'))}\n{json_data}\n")


def create_dirs(models=None):
    if not models:
        models = ['decision_tree', 'log_reg', 'random_forest', 'svm']
    paths = [
        config.ARTIFACTS_PATH+model_dir+'/' for model_dir in models
    ] + [
        config.IMAGE_DEST+model_dir+'/' for model_dir in models
    ] + [
        config.IMAGE_DEST+'k_projects/'+model_dir+'/' for model_dir in models
    ] + [
        config.INFO_DEST+model_dir+'/' for model_dir in models
    ] + [
        config.ROOT+'trained_models/', config.ROOT+'processed_data/'
    ]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    print(f"{len(paths)} directories created...")
    print('Created all directories!')

def create_info_dir():
    if not os.path.exists(config.INFO_DEST):
        os.makedirs(config.INFO_DEST)


def create_random_forest_parameters(
    max_depths=[2, 3],
    n_estimators=[20, 100],
    min_samples_split=2,
    min_samples_leaf=2,
) -> list:
    parameters_list = []
    # create various combinations of the above attributes
    for max_depth in max_depths:
        for n in n_estimators:
            parameters = {
                'criterion': "gini",
                'max_depth': max_depth,
                'n_estimators': n,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }
            parameters_list.append(parameters)
    return parameters_list


def create_logistic_regression_parameters(
        solver="saga",
        max_iters=[100, 200],
        penalties=["l1", "l2"]
) -> list:
    parameters_list = []
    for penalty in penalties:
        for max_iter in max_iters:
            parameters = {
                'penalty': penalty,
                'solver': solver,
                'max_iter': max_iter
            }
            parameters_list.append(parameters)

    return parameters_list


def create_xgb_classifier_parameters(
        n_estimators=[100, 200],
        max_depths=[3, 4],
        learning_rates=[0.1, 0.2]
) -> list:
    parameters_list = []
    for lr in learning_rates:
        for max_depth in max_depths:
            for n in n_estimators:
                parameters = {
                    'learning_rate': lr,
                    'max_depth': max_depth,
                    'n_estimators': n
                }
                parameters_list.append(parameters)

    return parameters_list


def create_svm_parameters(
        kernels=['poly'],
        degrees=[3, 4]
) -> list:
    parameters_list = []
    for kernel in kernels:
        for degree in degrees:
            parameters = {
                'kernel': kernel,
                'degree': degree,
                'class_weight': 'balanced'
            }
            parameters_list.append(parameters)

    return parameters_list


def create_classification_models(
        training_features_count: int,
        random_forest_parameters_list: list = None,
        logistic_regression_parameters_list: list = None,
        xgb_classifier_parameters_list: list = None,
        svm_parameters_list: list = None,
        nn_parameters_list: list = None,
        lightgbm_parameters_list: list = None,
        baseline: bool = True
) -> list:
    models_list = []
    i = 1

    if random_forest_parameters_list != None:
        for parameters in random_forest_parameters_list:
            new_model = RandomForestClassifier(**parameters)
            models_list.append({
                'model_name': f'random_forest_t_{parameters["n_estimators"]}_md_{parameters["max_depth"]}',
                'model': new_model,
                'type': 'non-linear',
                'parameters': parameters,
                'library': 'sklearn',
                'scaling': 'none'
            })
            i += 1

    if logistic_regression_parameters_list != None:
        # Create two models for each scaler: standard and minmax
        i = 1
        for parameters in logistic_regression_parameters_list:
            new_model_std = LogisticRegression(**parameters)
            models_list.append({
                'model_name': f'logistic_regression_mi_{parameters["max_iter"]}_p_{parameters["penalty"]}_standard',
                'model': new_model_std,
                'type': 'linear',
                'parameters': parameters,
                'library': 'sklearn',
                'scaling': 'standard'
            })
            i += 1

        j = 1
        for parameters in logistic_regression_parameters_list:
            new_model_mm = LogisticRegression(**parameters)
            models_list.append({
                'model_name': f'logistic_regression_mi_{parameters["max_iter"]}_p_{parameters["penalty"]}_minmax',
                'model': new_model_mm,
                'type': 'linear',
                'parameters': parameters,
                'library': 'sklearn',
                'scaling': 'minmax'
            })
            j += 1

    if svm_parameters_list != None:
        for parameters in svm_parameters_list:
            new_model = svm.SVC(**parameters)
            models_list.append({
                'model_name': f'svm_k_{parameters["kernel"]}_d_{parameters["degree"]}',
                'model': new_model,
                'type': 'linear',
                'parameters': parameters,
                'library': 'sklearn',
                'scaling': 'standard'
            })

    if xgb_classifier_parameters_list != None:
        i = 1
        for parameters in xgb_classifier_parameters_list:
            new_model = XGBClassifier(**parameters)
            models_list.append({
                'model_name': f'xgb_classifier_t_{parameters["n_estimators"]}_md_{parameters["max_depth"]}_lr_{parameters["learning_rate"]}',
                'model': new_model,
                'type': 'non-linear',
                'parameters': parameters,
                'library': 'xgboost',
                'scaling': 'none'
            })
            i += 1


    if lightgbm_parameters_list != None:
        i = 1
        for parameters in lightgbm_parameters_list:
            new_model = lgb.LGBMClassifier(**parameters)
            models_list.append({
                'model_name': f'lgbm_classifier_numl_{parameters["num_leaves"]}_md_{parameters["max_depth"]}_lr_{parameters["learning_rate"]}',
                'model': new_model,
                'type': 'non-linear',
                'parameters': parameters,
                'library': 'lightgbm',
                'scaling': 'none'
            })
            i += 1

    if nn_parameters_list != None:
        i = 1
        for parameters in nn_parameters_list:
            out_units_layer_1 = parameters['out_units_layer_1']
            out_units_layer_2 = parameters['out_units_layer_2']
            out_units_layer_3 = parameters['out_units_layer_3']
            out_units_layer_4 = parameters['out_units_layer_4']
            learning_rate = parameters['learning_rate']
            activation_fn = parameters['activation_fn']
            loss_fn = parameters['loss_fn']
            epochs = parameters['epochs']

            # Build NN
            new_model = Sequential()
            new_model.add(Input(shape=(training_features_count,)))
            new_model.add(Dense(out_units_layer_1, activation=activation_fn))
            new_model.add(Dense(out_units_layer_2, activation=activation_fn))
            new_model.add(Dense(out_units_layer_3, activation=activation_fn))
            new_model.add(Dense(out_units_layer_4, activation=activation_fn))
            new_model.add(Dense(1, activation='sigmoid'))

            # Compile model
            new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
              loss=loss_fn, 
              metrics=metrics.Precision()
              )
            
            models_list.append({
                'model_name': f'nn_lr_{learning_rate}_loss_{loss_fn}_activation_{activation_fn}_epochs_{epochs}',
                'model': new_model,
                'type': 'nn',
                'parameters': parameters,
                'library': 'keras',
                'scaling': 'standard'
            })

            i = i+1


    cost_sorted_k_baseline_model = {
        'model_name': 'cost_sorted_k_baseline_model',
        'model': None,
        'type': 'baseline',
        'scaling': 'none'
    }
    random_k_baseline_model = {
        'model_name': 'random_k_baseline_model',
        'model': None,
        'type': 'baseline',
        'scaling': 'none'
    }

    if baseline:
        models_list.append(cost_sorted_k_baseline_model)
        models_list.append(random_k_baseline_model)

    return models_list


def filter_dataset_by_date(data, start_date=config.MIN_TIME, end_date=config.MAX_TIME):

    data = data[
        (data["Project Posted Date"] >= pd.to_datetime(
            pd.Timestamp(start_date) - timedelta(days=config.LEAK_OFFSET)))
    ]
    data = data[
        (data["Project Posted Date"] <= pd.to_datetime(
            pd.Timestamp(end_date) - timedelta(days=100)))
    ]
    return data
