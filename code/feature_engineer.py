from helper import save_model
import data_processor as dp
import config
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve)
import random
import seaborn as sns
from datetime import timedelta
from pandas.core.frame import DataFrame
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import os
from helper import (log_intermediate_output_to_file)

# Set the backend to a non-GUI backend (e.g., 'Agg' for PNG files)
matplotlib.use('Agg')


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


def label_data(data: DataFrame, threshold: float):
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

    # "Not get funded" is 1
    data["Label"] = data.apply(
        lambda x: 1 if x["Fund Ratio"] < threshold else 0, axis=1)

    return data.drop_duplicates()


def get_best_proba_threshold_prediction(proba_predictions: list, y_test):
    thresholds = list(np.arange(0.3, 0.7, 0.05).round(2))
    best_threshold = None
    best_f1_score = 0.0
    best_precision = 0.0
    best_prediction = None

    for threshold in thresholds:
        # Convert probabilities to binary predictions based on the threshold
        binary_predictions = (proba_predictions[:, 1] >= threshold).astype(int)

        # Calculate F1 score
        f1 = f1_score(y_test, binary_predictions)
        precision = precision_score(y_test, binary_predictions)

        # if f1 > best_f1_score:
        #     best_f1_score = f1
        #     best_threshold = threshold
        #     best_prediction = binary_predictions
        if precision > best_precision:
            best_precision = precision
            best_threshold = threshold
            best_prediction = binary_predictions
    # print("best_f1_score = ", best_f1_score)
    return best_threshold, best_prediction


# Function to observe PR to select the best k
def prk_curve_for_top_k_projects(
        proba_predictions: list,
        k_start: int,
        k_end: int,
        k_gap: int,
        y_test,
        t_current,
        fold: int,
        model_name):
    # Temp: consider 1 for failing projects and 0 for projects getting fully funded in four months
    # Select the probabilities for label 1
    probabilities = proba_predictions[:, 1]
    # Rank the probabilities in descending order
    temp = (-1 * probabilities).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(probabilities))
    total_probabilities = len(proba_predictions)

    # Create new labels based on the k value and plot precision and recall
    precision = []
    recall = []
    k_value = []
    new_labels = []
    precision_recall_smallest_gap = 1.0
    best_k_for_min_dif = None
    best_k_for_max_f1 = None
    best_k_perc = None
    best_labels = None
    precision_for_best_k = None
    max_f1 = 0

    print(
        f"k_start = {k_start}, len(proba_predictions) = {len(proba_predictions)}, k_gap = {k_gap}")
    for k in range(k_start, len(proba_predictions), k_gap):
        k_labels = (ranks <= k).astype(int)
        new_labels.append(k_labels)
        k_value.append(k)
        k_precision = precision_score(y_test, k_labels)
        k_recall = recall_score(y_test, k_labels)
        precision.append(k_precision)
        recall.append(k_recall)

        # Find the smallest gap between precision and recall
        difference = abs(k_precision - k_recall)
        if precision_recall_smallest_gap >= difference:
            precision_recall_smallest_gap = difference
            best_k_for_min_dif = k
            precision_for_best_k = k_precision
            best_k_perc = k/total_probabilities
            best_labels = k_labels

        f1 = f1_score(y_test, k_labels)

        if max_f1 < f1:
            best_k_for_max_f1 = k
            max_f1 = f1

    # Print the k with the minimum difference between P and R
    # print(f"K with the minimum difference between P and R: {best_k}")
    k_value_perc = [val/total_probabilities*100 for val in k_value]

    # Plot the prk curve
    plt.cla()
    plt.plot(k_value_perc, precision, label='precision')
    plt.plot(k_value_perc, recall, label='recall')
    plt.xlabel('Value of k as a percentage (%)')
    plt.title(
        f"Model's Precision and Recall for Varying k ({len(proba_predictions)} test inputs)")
    plt.legend()
    plt.savefig(config.K_PROJECTS_DEST + model_name +
                f"prk_curve_for_fold_{fold}_{str(t_current)[:10]}.png")
    plt.show()
    plt.clf()

    # print("temp = ", list(temp))
    print("best_k_for_min_dif = ", best_k_for_min_dif)
    print("best_k_for_max_f1 = ", best_k_for_max_f1)
    best_k_index = list(temp).index(best_k_for_min_dif)
    best_threshold_by_diff = probabilities[best_k_index]
    best_k_index = list(temp).index(best_k_for_max_f1)
    best_threshold_by_f1 = probabilities[best_k_index]
    print("best_threshold_by_diff = ", best_threshold_by_diff)
    print("best_threshold_by_f1 = ", best_threshold_by_f1)

    prk_results = {
        'k_value': k_value,
        'precision': precision,
        'recall': recall,
        # 'new_labels': new_labels,
        'best_k_for_min_dif': best_k_for_min_dif,
        'best_k_for_max_f1': best_k_for_max_f1,
        'best_threshold_by_diff': best_threshold_by_diff,
        'best_threshold_by_f1': best_threshold_by_f1,
        'precision_for_best_k': precision_for_best_k,
        # 'best_labels': best_labels,
        'best_k_percentage': best_k_perc
    }

    return prk_results


def plot_roc_curve(proba_predictions, y_test, t_current):

    # Get the probabilities for class 1
    probabilities = proba_predictions[:, 1]
    # Find the ROC curve and get the threshold list
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    # Calculate the distance of each point on the ROC curve to the top-left corner
    distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)

    # Find the index of the point with the minimum distance
    best_index = np.argmin(distances)

    # Get the corresponding threshold value
    best_threshold = thresholds[best_index]

    # print(f"Best Threshold: {best_threshold} for {t_current[: 10]}")

    # Find the AUC
    auc = round(roc_auc_score(y_test, probabilities), 4)
    # Plot the curve and save
    plt.cla()
    plt.plot(fpr, tpr, label="AUC="+str(auc))
    plt.title("ROC curve")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(config.ROC_CURVE_DEST +
                f"roc_curve_for_{str(t_current)[: 10]}.png")
    # plt.show()
    plt.clf()

    return best_threshold


def plot_precision_vs_recall_curve(proba_predictions, y_test, t_current):

    # Get the probabilities for class 1
    probabilities = proba_predictions[:, 1]
    # Find the P-R curve and get the threshold list
    precision, recall, thresholds = precision_recall_curve(
        y_test, probabilities)
    # Calculate the distance of each point on the P-R curve to the top-right corner
    distances = np.sqrt((1 - precision) ** 2 + (1 - recall) ** 2)

    # Find the index of the point with the minimum distance
    best_index = np.argmin(distances)

    # Get the corresponding threshold value
    best_threshold = thresholds[best_index]

    # print(f"Best Threshold: {best_threshold} for {t_current[: 10]}")

    # Plot the curve and save
    plt.cla()
    plt.plot(recall, precision)
    plt.title("Precision vs Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    # plt.savefig(config.P_VS_R_CURVE_DEST+ f"precision_vs_recall_curve_for_{str(t_current)[: 10]}")
    # plt.show()
    plt.clf()

    return best_threshold


def create_proba_sorted_k_labels(k, proba_predictions):
    # Select the probabilities for label 1
    probabilities = proba_predictions[:, 1]
    # Rank the probabilities in descending order
    temp = (-1 * probabilities).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(probabilities))

    # Create new labels based on the k value
    k_labels = (ranks <= k).astype(int)

    return k_labels


def create_cost_sorted_k_labels(k, x_test):
    cost_test = x_test["Project Cost"]
    # Rank the project costs in descending order
    temp = (-1 * cost_test).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(cost_test))

    # Create new labels based on the k value
    k_labels = (ranks <= k).astype(int)

    return k_labels


def create_random_k_labels(k: int, test_size: int):
    if test_size < k:
        k = test_size
    ones = [1] * k
    zeros = [0] * (test_size - k)
    k_labels = ones + zeros
    random.shuffle(k_labels)

    return k_labels


def get_k_labels(model_name, model_type, x_test, y_test, y_hat, k=config.FIXED_KVAL) -> list:
    # For a fixed value of k, find the precision for each fold
    print(
        f"get k labels ==============={model_type}===============================")
    if model_type != "baseline":
        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Creating labels for trained model output.')
        k_labels = create_proba_sorted_k_labels(
            k=k, proba_predictions=y_hat)
    else:
        if model_name[:-1] == "random_k_baseline_model":
            print("base_line_rand_k--------------")
            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Creating labels for baseline model; random.')
            k_labels = create_random_k_labels(
                k=k, test_size=len(y_test))
        else:
            print("base_line_cost_sorted---------")
            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Creating labels for baseline model; cost-sorted.')
            k_labels = create_cost_sorted_k_labels(
                k=k, x_test=x_test)
    return k_labels


def get_precision_for_fixed_k(k_labels, y_test):
    k_precision = precision_score(y_test, k_labels)
    return k_precision


def get_positive_percentage(y_train: DataFrame, y_test: DataFrame):
    train_pos = y_train["Label"].value_counts()[1] / len(y_train["Label"])
    test_pos = y_test["Label"].value_counts()[1] / len(y_test["Label"])

    return train_pos, test_pos


def plot_k_fold_evaluation_metrics(model_eval_metrics: dict, model_name: str):

    x_labels = [
        f"Fold {i+1}" for i in range(len(model_eval_metrics.get("accuracy", 0)))]
    x_positions = np.arange(len(x_labels))
    bar_width = 0.2

    # Plot accuracy and f1 score for all the folds
    # print( x_positions, bar_width, len(model_eval_metrics["accuracy"]), len(model_eval_metrics["f1_score"]))

    plt.cla()
    plt.bar(x_positions - bar_width,
            model_eval_metrics["accuracy"], width=bar_width, label='Accuracy')
    plt.bar(x_positions,
            model_eval_metrics["f1_score"], width=bar_width, label='F1 Score')

    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Values')
    plt.title("Model's Accuracy and F1 Score for Each validation fold")
    plt.xticks(x_positions, x_labels, rotation=90)
    plt.legend()
    plt.savefig(config.IMAGE_DEST + model_name + 'cross_validation_plot.png')
    # plt.show()
    plt.clf()

    # Plot the model accuracy for all the folds
    plt.cla()
    plt.plot(x_labels, model_eval_metrics["model_score"])

    plt.xlabel('Fold')
    plt.ylabel('Model Score')
    plt.title("Model Score for each fold")
    plt.xticks(x_positions, x_labels, rotation=90)
    plt.savefig(config.IMAGE_DEST + model_name + 'model_score_plot.png')
    # plt.show()
    plt.clf()


def plot_precision_for_fixed_k(model_eval_metrics: dict, model_name: str):

    x_labels = [
        f"Fold {i+1}" for i in range(len(model_eval_metrics.get("k_fixed_precision", 0)))]
    x_positions = np.arange(len(x_labels))

    # Plot the model precision for all the folds for a fixed value of k
    plt.cla()
    plt.plot(x_labels, model_eval_metrics["k_fixed_precision"])

    plt.xlabel('Fold')
    plt.ylabel('Precision for fixed k')
    plt.title("Precision for each fold for fixed k")
    plt.xticks(x_positions, x_labels, rotation=90)
    plt.savefig(config.IMAGE_DEST + model_name + 'k_fixed_precision_plot.png')
    # plt.show()
    plt.clf()

    return


def plot_precision_for_fixed_k_for_multiple_models(model_names: list, model_eval_metrics: dict):
    ''' A dict of evaluation metrics of each model should be passed.
        fixed_k_plot_data = {
            "fixed_k_value": fixed_k_value,
            "fold_no": folds+1,
            "start_date": start_date,
            "k_fixed_precision": k_fixed_precision
        }
    '''
    sample_plot_data = model_eval_metrics.get(
        model_names[0]).get("fixed_k_plot_data")
    x_labels = [str(entry["start_date"])[:10]
                for entry in sample_plot_data][::-1]
    x_positions = np.arange(len(x_labels))
    print("x_labels = ", x_labels)

    plt.figure(figsize=(18, 10))

    for model_name in model_names:
        plot_data = model_eval_metrics.get(model_name).get("fixed_k_plot_data")
        precision_data = [entry["k_fixed_precision"]
                          for entry in plot_data][::-1]

        plt.plot(x_labels, precision_data, label=model_name)

    plt.ylabel(
        f'Precision for fixed k (k = {config.FIXED_KVAL})')
    plt.xlabel('Fold Start Dates')
    plt.title("Precision for each fold for fixed k for each model")
    plt.xticks(x_positions, x_labels, rotation=20, fontsize=10)
    plt.legend()
    # plt.show()
    plt.savefig(config.IMAGE_DEST +
                'k_fixed_precision_plot_for_all_models.png')


def split_data_folds(data: DataFrame) -> list:
    # Initiate timing variables
    max_t = pd.Timestamp(config.MAX_TIME)
    min_t = pd.Timestamp(config.MIN_TIME)
    shift_period = timedelta(days=config.LEAK_OFFSET)   # 4 months
    fold_period = timedelta(days=config.WINDOW)    # 15 months

    t_current = max_t
    folds = 1

    folded_dataset = []

    while t_current > min_t + fold_period:
        start_date = t_current - fold_period

        x_train, y_train, x_test, y_test = dp.split_temporal_train_test_data(
            data=data,
            start_date=start_date
        )

        # Count the positive labeled percentage in the training set and the test set
        train_pos_perc, test_pos_perc = get_positive_percentage(
            y_train, y_test)

        fold_dataset = {
            "fold_no": folds,
            "start_date": start_date,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "train_pos_perc": train_pos_perc,
            "test_pos_perc": test_pos_perc
        }
        folded_dataset.append(fold_dataset)

        train_end = start_date + timedelta(config.TRAIN_SIZE)
        test_start = train_end + timedelta(config.LEAK_OFFSET)
        test_end = test_start + timedelta(config.TEST_SIZE)

        fold_info = {
            'fold_number': folds,
            'timeline': {
                'train_start': str(start_date)[:10],
                'train_end': str(train_end)[:10],
                'test_start': str(test_start)[:10],
                'test_end': str(test_end)[:10]
            },
            'shape': {
                'training_shape': str(x_train.shape),
                'test_shape': str(x_test.shape),
            },
            'data_distribution': {
                'train_positive_ratio': train_pos_perc,
                'test_positive_ratio': test_pos_perc
            }
        }
        file_name = f"Fold {folds+1} - {str(start_date)[:10]}.json"
        dp.save_json(fold_info, config.INFO_DEST+file_name)

        t_current -= shift_period
        folds += 1

    return folded_dataset


def train_eval_classifier(model, model_type, x_train, y_train, x_test, y_test):
    if model_type == "linear":
        # Scaling
        x_train, x_test = standardize_data(
            x_train, x_test, config.VARIABLES_TO_SCALE)

    # Model Training
    model = model.fit(x_train, y_train.values.ravel())

    # Predicting
    y_hat = model.predict_proba(x_test)

    return model, y_hat


def cross_validate(data, model_item):
    # create a splitted & folded dataset once and pass to each model
    folded_dataset = split_data_folds(data)

    model = model_item.get("model")
    model_name = model_item.get("model_name")+"/"
    model_type = model_item.get("type")

    model_eval_metrics = {
        "accuracy": [],
        "f1_score": [],
        "recall": [],
        "model_score": [],
        "fixed_k_plot_data": []
    }

    # cross validate on each fold
    for fold_data in folded_dataset:
        x_train = fold_data.get("x_train")
        y_train = fold_data.get("x_train")
        x_test = fold_data.get("x_test")
        y_test = fold_data.get("y_test")

        # handle empty dataset situation
        if x_train.shape[0] == 0 or x_test.shape[0] == 0 or y_train.shape[0] == 0 or y_test.shape[0] == 0:
            continue

        if model_type == "linear":
            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Standardizing data.')
            x_train, x_test = standardize_data(
                x_train, x_test, config.VARIABLES_TO_SCALE)

        if model_type != "baseline":
            # train the model
            model, y_hat = train_eval_classifier(
                model=model,
                model_type=model_type,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )

            # Observing the best threshold using different methods
            prk_results = prk_curve_for_top_k_projects(
                proba_predictions=y_hat,
                k_start=10,
                k_end=int(y_hat.shape[0]*0.8),
                k_gap=50,
                y_test=fold_data.get("x_test"),
                t_current=fold_data.get("start_date"),
                fold=fold_data.get("fold_no"),
                model_name=model_name
            )
        else:
            y_hat = None
            prk_results = None

        # k value is by default set in configuration file, can override here.
        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Creating binary k labels based on the model type.')

        # get custom labels for a fixed k value
        k_labels = get_k_labels(
            model_name=model_name,
            model_type=model_type,
            x_test=x_test,
            y_test=y_test,
            y_hat=y_hat
        )

        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Find the precision score.')

        k_fixed_precision = get_precision_for_fixed_k(
            k_labels=k_labels, y_test=y_test)

        fixed_k_plot_data = {
            "fixed_k_value": config.FIXED_KVAL,
            "fold_no": fold_data.get("fold_no"),
            "start_date": fold_data.get("start_date"),
            "k_fixed_precision": k_fixed_precision
        }
        model_eval_metrics["fixed_k_plot_data"].append(
            fixed_k_plot_data)

        # Evaluate the model
        if model_type != "baseline":
            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Get model score for trained model.')
            model_score = model.score(x_test, y_test)
            accuracy = accuracy_score(y_test, y_hat)
            f1 = f1_score(y_test, y_hat)
            recall = recall_score(y_test, y_hat)
            model_eval_metrics["accuracy"].append(accuracy)
            model_eval_metrics["f1_score"].append(f1)
            model_eval_metrics["recall"].append(recall)
            model_eval_metrics["model_score"].append(model_score)

            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Saving model artefacts.')
            art_path = config.ARTIFACTS_PATH+model_name
            if not os.path.exists(art_path):
                os.makedirs(art_path)

            save_model(
                art_path,
                file_name=f'{model_name[:-1]}_fold_{fold_data.get("fold_no")}_{str(fold_data.get("start_date"))[:10]}.pkl',
                model=model
            )

            predicted_probabilities_df = pd.DataFrame(
                y_hat, columns=model.classes_, index=y_test.index)

            test_prediction = pd.concat(
                [data.loc[y_test.index]["Project ID"], predicted_probabilities_df[1]], axis=1)
            test_prediction["Start Date"] = fold_data.get("start_date")

            dp.export_data_frame(
                test_prediction,
                art_path +
                f'test_prediction_fold_{fold_data.get("fold_no")}_{str(fold_data.get("start_date"))[:10]}.csv'
            )

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Exiting fold creation and model training.')

    return model_eval_metrics


def cross_validate_old(data, model_item):

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Starting cross_validate function.')
    model = model_item.get("model")
    model_name = model_item.get("model_name")+"/"
    model_type = model_item.get("type")

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Initiating time variables.')
    # Initiate timing variables
    max_t = pd.Timestamp(config.MAX_TIME)
    min_t = pd.Timestamp(config.MIN_TIME)
    shift_period = timedelta(days=config.LEAK_OFFSET)   # 4 months
    fold_period = timedelta(days=config.WINDOW)    # 15 months

    time_period = timedelta(days=config.DONATION_PERIOD)        # 30 days

    t_current = max_t
    # print("================\n", t_current, max_t, training_window)

    model_eval_metrics = {
        "accuracy": [],
        "precision": [],
        "f1_score": [],
        "model_score": [],
        "k_fixed_precision": [],
        "fixed_k_plot_data": []
    }

    folds = 0

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Starting the loop for folds.')
    while t_current > min_t + fold_period:

        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, f"======================================FOLD==== {folds+1}")
        start_date = t_current - fold_period
        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, f'Starting fold for {start_date}.')

        x_train, y_train, x_test, y_test = dp.split_temporal_train_test_data(
            data=data,
            start_date=start_date
        )

        cost_test = x_test["Project Cost"]

        # Count the positive labeled percentage in the training set and the test set
        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Get positive percentage.')
        train_pos_perc, test_pos_perc = get_positive_percentage(
            y_train, y_test)

        # Scaling
        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Standardizing data.')
        x_train, x_test = standardize_data(
            x_train, x_test, config.VARIABLES_TO_SCALE)

        # handle empty dataset situation
        if x_train.shape[0] == 0 or x_test.shape[0] == 0 or y_train.shape[0] == 0 or y_test.shape[0] == 0:
            continue

        # Model Training
        if model_type != "baseline":
            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Training model.')
            model = model.fit(x_train, y_train.values.ravel())

            # Predicting
            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Predicting after training.')
            y_hat = model.predict_proba(x_test)

            # Observing the best threshold using different methods
            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Plotting PRK curves for varying k.')
            prk_results = prk_curve_for_top_k_projects(
                proba_predictions=y_hat,
                k_start=config.K_START,
                k_end=int(y_hat.shape[0]*0.8),
                k_gap=config.K_STEP,
                y_test=y_test,
                t_current=start_date,
                fold=folds+1,
                model_name=model_name
            )

            best_k = prk_results.get('best_k_for_max_f1')
            best_k_perc = prk_results.get('best_k_percentage')
            best_threshold_by_f1 = prk_results.get('best_threshold_by_f1')
        else:
            y_hat = None
            prk_results = None

        # k value is by default set in configuration file, can override here.
        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Creating binary labels based on the model type.')
        k_labels = get_k_labels(
            model_name=model_name,
            model_type=model_type,
            x_test=x_test,
            y_test=y_test,
            y_hat=y_hat
        )
        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Find the precision score.')
        k_fixed_precision = get_precision_for_fixed_k(
            k_labels=k_labels, y_test=y_test)

        # Evaluate the model
        if model_type != "baseline":
            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Get model score for trained model.')
            model_score = model.score(x_test, y_test)
        else:
            model_score = 0

        model_eval_metrics["model_score"].append(model_score)
        fixed_k_plot_data = {
            "fixed_k_value": config.FIXED_KVAL,
            "fold_no": folds+1,
            "start_date": start_date,
            "k_fixed_precision": k_fixed_precision
        }
        model_eval_metrics["fixed_k_plot_data"].append(fixed_k_plot_data)
        model_eval_metrics["k_fixed_precision"].append(k_fixed_precision)

        print(f"======================================FOLD==== {folds+1}")

        train_end = start_date + timedelta(config.TRAIN_SIZE)
        test_start = train_end + timedelta(config.LEAK_OFFSET)
        test_end = test_start + timedelta(config.TEST_SIZE)

        fold_info = {
            'classifier': model_name[:-1],
            'fold_number': folds+1,
            'timeline': {
                'train_start': str(start_date)[:10],
                'train_end': str(train_end)[:10],
                'test_start': str(test_start)[:10],
                'test_end': str(test_end)[:10]
            },
            'shape': {
                'training_shape': str(x_train.shape),
                'test_shape': str(x_test.shape),
            },
            'data_distribution': {
                'train_positive_ratio': train_pos_perc,
                'test_positive_ratio': test_pos_perc
            },
            'evaluation_metrics': {
                'model_score': model_score
            },
            'prk_results': prk_results
        }

        log_intermediate_output_to_file(
            config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Saving fold info.')
        file_name = f"Fold {folds+1} - {str(start_date)[:10]}.json"
        dp.save_json(fold_info, config.INFO_DEST+model_name+file_name)

        if model_type != "baseline":
            log_intermediate_output_to_file(
                config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Saving model artefacts.')
            art_path = config.ARTIFACTS_PATH+model_name
            if not os.path.exists(art_path):
                os.makedirs(art_path)

            save_model(
                art_path,
                file_name=f'{model_name[:-1]}_fold_{folds+1}_{str(start_date)[:10]}.pkl',
                model=model
            )
            # Combine x_train and y_train into a single DataFrame
            train_df_tmp = pd.concat([x_train, y_train], axis=1)
            train_df = pd.concat(
                [data.loc[train_df_tmp.index]["Project ID"], train_df_tmp], axis=1)
            predicted_probabilities_df = pd.DataFrame(
                y_hat, columns=model.classes_, index=y_test.index)
            # Combine x_test, y_test, and predicted_probabilities into a single DataFrame
            test_df_tmp = pd.concat([x_test, y_test], axis=1)
            test_df = pd.concat(
                [data.loc[test_df_tmp.index]["Project ID"], test_df_tmp], axis=1)

            test_prediction = pd.concat(
                [data.loc[y_test.index]["Project ID"], predicted_probabilities_df[1]], axis=1)
            test_prediction["Start Date"] = start_date

            dp.export_data_frame(
                train_df,
                art_path+f'train_fold_{folds+1}_{str(start_date)[:10]}.csv'
            )
            dp.export_data_frame(
                test_df,
                art_path+f'test_fold_{folds+1}_{str(start_date)[:10]}.csv'
            )

            dp.export_data_frame(
                test_prediction,
                art_path +
                f'test_prediction_fold_{folds+1}_{str(start_date)[:10]}.csv'
            )

        t_current -= shift_period
        folds += 1

    log_intermediate_output_to_file(
        config.INFO_DEST, config.PROGRAM_LOG_FILE, 'Exiting fold creation and model training.')

    return model_eval_metrics


def run_pipeline(data, model):
    model_eval_metrics = cross_validate(data, model)

    return model, model_eval_metrics

