import feature_engineer as fe
import data_processor as dp
import config
from helper import (
    create_dirs,
    create_classification_models,
    log_intermediate_output_to_file,
    filter_dataset_by_date)
import temporal_features as tmpf

from model_parameters.logistic_regression import lg_parameters
from model_parameters.random_forest import rf_parameters
from model_parameters.svm import svm_parameters
from model_parameters.xgboost import xgb_parameters
from model_parameters.neural_network import nn_parameters

data_file_path = config.PROCESSED_DATA_PATH
load_processed_data = config.LOAD_PROCESSED_DATA_FLAG


log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'About to load data from csv.')

if load_processed_data:
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Loading preprocessed data.')
    print("Loading already processed data")
    data = dp.load_data_to_df(path=data_file_path, rows=20000)
    data = dp.set_data_types_to_datetime(data, ["Project Posted Date"])
    data = filter_dataset_by_date(data)
else:
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Laoding data and preprocessing.')
    print("Start data pre processing")

    data = dp.load_data_to_df(config.DATA_SOURCE)

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Casting datetime datatype.')
    
    data = dp.set_data_types_to_datetime(data, config.DATE_COLS)
    data = filter_dataset_by_date(data)

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Imputing data.')
    
    data = dp.impute_data(data)
    print("Complete imputing = ", data.shape)

    # filter training features
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Selecting only the required features from df.')
    extra_features_required = ["Teacher ID",
                               "School ID", "School City", "School County", 
                               "Project Title", "Project Essay", "Project Need Statement", 
                               "Project Short Description"]
    data = data[config.TRAINING_FEATURES + ["Label"] + extra_features_required]
    print("Filtered training Features, shape = ", data.shape)

    # Adding new features
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Adding new features.')
    data = tmpf.add_new_features(data)
    print(data.columns)
    data = data.drop(extra_features_required, axis=1)
    print("After adding new Features, shape = ", data.shape)
    print(data.columns)

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE,
        f'After adding new Features, shape = {data.shape}, columns = {data.columns}')

    # export labelled data to csv
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Exporting labelled data to csv.')
    dp.export_data_frame(data=data, path=data_file_path)
    print(f"Saved data as csv at {data_file_path}")

print("label distribution 1:0 = ",
      data["Label"].value_counts()[1] / data["Label"].value_counts()[0])

log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE,
    f"Data {data.shape[0]} rows\nlabel distribution 1:0 = {data['Label'].value_counts()[1] / data['Label'].value_counts()[0]}"
)

log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Encoding data.')

data_1 = dp.encode_data(data, config.CATEGORICAL_COLS)

log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, f'Encoding complete. {data_1.shape}')
print("encoded_data.shape = ", data_1.shape)

data_folds, training_features_count = fe.split_data_folds(data_1)

# create classifiers including baseline models

models = create_classification_models(
    training_features_count,
    # random_forest_parameters_list=rf_parameters,
    # logistic_regression_parameters_list=lg_parameters,
    # svm_parameters_list=svm_parameters,
    # xgb_classifier_parameters_list=xgb_parameters,
    nn_parameters_list=nn_parameters,
    baseline=True)

# create dirs that not exist
model_names = [model.get("model_name") for model in models]
print(model_names)
create_dirs(models=model_names)  # can pass a list of specific model names

log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Creating directories if they do not exist.')

model_eval_metrics = {}
hyper_parameter_performance_table = []

log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Starting loop for each model.')
for model_item in models:

    print(f"Classifier -> {model_item.get('model_name')}")
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, f"Classifier -> {model_item.get('model_name')}")
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Start pipeline for model.')

    trained_model, eval_metrics = fe.run_pipeline(
        data=data_folds, model=model_item, training_features_count=training_features_count
    )

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'End pipeline for model.')

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE,
        'Plotting precision curves and saving hyperparameters.')

    k_fixed_precisions = [x.get("k_fixed_precision", 0)
                          for x in eval_metrics.get("fixed_k_plot_data", [])]
    log_intermediate_output_to_file(config.INFO_DEST, config.PROGRAM_LOG_FILE,
                                    f'k_fixed_precisions = {k_fixed_precisions}')

    fe.plot_precision_for_fixed_k(
        eval_metrics, model_item.get("model_name")+"/")
    model_eval_metrics.update({model_item.get("model_name"): eval_metrics})
    perf_row = {
        "model": model_item.get("model_name"),
        "hyper_paramaters": model_item.get("parameters"),
        "avg_precision": sum(k_fixed_precisions) / len(k_fixed_precisions)
    }
    log_intermediate_output_to_file(
        path=config.INFO_DEST,
        file_name="model_run_log.log",
        log_info=perf_row
    )
    hyper_parameter_performance_table.append(perf_row)


log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Save hyperparameter table for all models.')
file_path = config.INFO_DEST + "hyper_parameter_performance_table.json"
dp.save_json(hyper_parameter_performance_table, path=file_path)

log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Plot precision curves for all models.')
fe.plot_precision_for_fixed_k_for_multiple_models(
    model_names, model_eval_metrics)
