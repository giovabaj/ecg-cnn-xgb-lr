"""Script to train XGB or LR with the same dataset and the same cross-validations folds used for the CNN model. The
script takes as input the folder with the CNN results, from which exam IDs of training and validation are extracted."""
import json
import os
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

from evaluation import MetricsCV


def get_paths():
    """Arguments parsers from terminal"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--model_name', type=str, choices=['xgb', 'lr'])

    # Parsing arguments
    args = parser.parse_args()
    folder = args.folder
    model_name = args.model_name

    # paths to load and save
    path_dl = 'results/cnn/' + folder + '/'
    path_results = 'results/' + model_name + '/' + folder + '/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("\n--- Folder ", folder, " is processed --- ")

    return path_dl, path_results, model_name


def main():
    # DL-folder path to import exids and output path for results
    path_dl, path_results, model_name = get_paths()

    # Importing the config dict from DL folder, extracting only required fields and saving it to the new results folder
    with open(path_dl + '/config.json') as config_file:
        config = json.load(config_file)
    config = {key: value for key, value in config.items() if
              key in ['n_folds', 'n_samples', 'undersampling_flag', 'ratio', 'random_seed', 'path_df']}
    n_folds = config["n_folds"]
    random_seed = config["random_seed"]
    undersampling_flag = config['undersampling_flag']
    ratio = config['ratio']
    with open(path_results + '/config.json', 'w') as outfile:
        json.dump(config, outfile)

    # Importing the dataframe
    path_df = config["path_df"]
    features = ['P_AXIS', 'P_OFFSET', 'P_ONSET', 'PR_INT', 'QRS_AXIS', 'QRS_OFFSET', 'QRS_ONSET', 'QTC_INT',
                'T_AXIS', 'T_OFFSET', 'V_RATE']

    df = pd.read_csv(path_df, encoding='iso-8859-1')[features + ["ExamID", "KEY_ANAGRAFE", "FA"]]

    # Importing the best parameters for the model
    best_params_filename = "best_params.json"

    with open(os.path.dirname(path_results[:-1]) + '/' + best_params_filename) as params_file:
        best_params = json.load(params_file)

    # Initializing variables
    set_names = ['train', 'test']
    exids = {}
    df_dict = {}
    metrics_dict = {}
    for set_name in set_names:
        metrics_dict[set_name] = MetricsCV(path_results, set_name)

    for i in range(n_folds):  # loop over CV-folds
        path_fold_dl = path_dl + "CVFold_" + str(i) + "/"

        # train set obtained merging train and validation sets
        exids_val = np.load(path_fold_dl + "exids_val.npy")
        if undersampling_flag:  # undersampling the validation set if necessary
            labels_val = np.load(path_fold_dl + "labels_val.npy")
            random_under_sampler = RandomUnderSampler(sampling_strategy=ratio / (1 - ratio), random_state=random_seed)
            exids_val, _ = random_under_sampler.fit_resample(exids_val.reshape(-1, 1), labels_val)
            exids_val = exids_val.reshape(-1, )

        exids["train"] = np.append(np.load(path_fold_dl + "exids_train.npy"), exids_val)
        exids["test"] = np.load(path_fold_dl + "exids_test.npy")

        # dictionary with train and test dataframes
        for set_name in set_names:
            df_dict[set_name] = pd.merge(pd.Series(exids[set_name], name="ExamID"), df, on="ExamID")

        # set model parameters and train the model
        if model_name == "lr":
            model = LogisticRegression(max_iter=100000, random_state=random_seed)
        elif model_name == "xgb":
            model = xgb.XGBClassifier(eval_metric='logloss', tree_method='hist')
        model.set_params(**best_params)  # noqa
        model.fit(df_dict["train"][features], df_dict["train"]["FA"])

        # Making predictions and saving results
        path_results_fold = path_results + "CVFold_" + str(i) + "/"
        if not os.path.exists(path_results_fold):
            os.makedirs(path_results_fold)
        pickle.dump(model, open(path_results_fold + "model_trained.pickle", 'wb'))  # save trained model
        for set_name in set_names:
            y_pred = model.predict_proba(df_dict[set_name][features])[:, 1]
            # Save results
            np.save(path_results_fold + 'predictions_' + set_name, y_pred)
            np.save(path_results_fold + 'labels_' + set_name, df_dict[set_name]["FA"].values)
            np.save(path_results_fold + 'exids_' + set_name, df_dict[set_name]["ExamID"].values)
            # Computing metrics
            metrics_dict[set_name].compute_metrics(df_dict[set_name]["FA"], y_pred)

    # Saving metrics results for all the folds
    for set_name in set_names:
        metrics_dict[set_name].save_results()


if __name__ == "__main__":
    main()
