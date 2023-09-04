import feature_engineer as fe
import data_processor as dp
import config
from helper import (
    save_model,
    load_model,
    create_dirs,
    create_classification_models,
    create_logistic_regression_parameters,
    create_random_forest_parameters,
    create_xgb_classifier_parameters,
    log_intermediate_output_to_file,
    filter_dataset_by_date)
import temporal_features as tmpf

data_file_path = config.PROCESSED_DATA_PATH
load_processed_data = config.LOAD_PROCESSED_DATA_FLAG

# create classifiers including baseline models
rand_for_params = create_random_forest_parameters(
    max_depths=[4], n_estimators=[200, 500])
# log_reg_params = create_logistic_regression_parameters(
#     max_iters=[100], penalties=["l1"])
# xgb_classifier_params = create_xgb_classifier_parameters()
models = create_classification_models(
    random_forest_parameters_list=rand_for_params)


# create dirs that not exist
model_names = [model.get("model_name") for model in models]
print(model_names)
create_dirs(models=model_names)  # can pass a list of specific model names

log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Creating directories if they do not exist.')

log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'About to load data from csv.')

if load_processed_data:
    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Loading preprocessed data.')
    print("Loading already processed data")
    data = dp.load_data_to_df(path=data_file_path)
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
    extra_features_required = ["Teacher ID", "School ID"]
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

sample_rows = 300000
data = data.sample(n=sample_rows)
print(f"data sampled with {sample_rows} rows")
print("label distribution 1:0 = ",
      data["Label"].value_counts()[1] / data["Label"].value_counts()[0])
log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE,
    f"Data sampled with {sample_rows} rows\nlabel distribution 1:0 = {data['Label'].value_counts()[1] / data['Label'].value_counts()[0]}"
)


log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Encoding data.')
data_1 = dp.encode_data(data, config.CATEGORICAL_COLS)
log_intermediate_output_to_file(
    config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Encoding complete.')
print("encoded_data.shape = ", data_1.shape)

data_folds = fe.split_data_folds(data_1)

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
        data=data_folds, model=model_item
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
