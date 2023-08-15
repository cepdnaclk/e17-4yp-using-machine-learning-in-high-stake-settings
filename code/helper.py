import pickle
import config
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def save_model(path, file_name, model):
    file_path = path + file_name
    pickle.dump(model, file=open(file_path, "wb"))


def load_model(model_file_path):
    return pickle.load(open(model_file_path, 'rb'))


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


def create_classification_models(
        random_forest_parameters_list: list,
        logistic_regression_parameters_list: list
) -> list:
    models_list = []
    i = 1
    for parameters in random_forest_parameters_list:
        new_model = RandomForestClassifier(**parameters)
        models_list.append({
            'model_name': f'random_forest_{i}',
            'model': new_model,
            'type': 'non-linear',
            'parameters': parameters
        })
        i += 1

    i = 1
    for parameters in logistic_regression_parameters_list:
        new_model = LogisticRegression(**parameters)
        models_list.append({
            'model_name': f'logistic_regression_{i}',
            'model': new_model,
            'type': 'linear',
            'parameters': parameters
        })
        i += 1
    cost_sorted_k_baseline_model = {
        'model_name': 'cost_sorted_k_baseline_model',
        'model': None,
        'type': 'baseline'
    }
    random_k_baseline_model = {
        'model_name': 'random_k_baseline_model',
        'model': None,
        'type': 'baseline'
    }
    models_list.append(cost_sorted_k_baseline_model)
    models_list.append(random_k_baseline_model)

    return models_list
