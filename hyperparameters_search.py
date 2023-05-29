"""Script to perform hyperparameters search for XGB or LR models"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from scipy.stats import randint, uniform
import pandas as pd
import sys
import warnings
import os
import json
import argparse
import xgboost as xgb
import pprint


def get_args():
    """Get configuration arguments for the hyperparameters search"""
    parser = argparse.ArgumentParser()  # Create the parser
    parser.add_argument('--model_name', type=str, choices=['xgb', 'lr'])
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=42)

    # Parsing arguments
    args = parser.parse_args()
    model_name = args.model_name
    n_folds = args.n_folds
    n_iter = args.n_iter
    random_seed = args.random_seed

    return model_name, n_folds, n_iter, random_seed


def load_dataset(features, random_seed):
    """Load exams not used for training/test.
    It is fundamental to use the same random_seed used to build the other datasets: in this way all smaller datasets are
    sub-samples of the bigger ones. Thus removing the sample of 150.000 exams from the original dataset allows you to
    remain with the exams not used."""

    df = pd.read_csv('/coorte/ASUGI/ECG_FA_5Y_v06.csv', encoding='iso-8859-1') # Importing tabular data

    n_samples = 150000  # biggest dataset size considered in the study

    # Get the 150.000 exams sample
    frac_1fa = (df["FA"] == 1).sum() / len(df)
    n_1fa = int(frac_1fa * n_samples)
    n_0fa = n_samples - n_1fa
    df_1fa = df[df["FA"] == 1].sample(n_1fa, random_state=random_seed)
    df_0fa = df[df["FA"] == 0].sample(n_0fa, random_state=random_seed)
    df_sample = pd.concat([df_1fa, df_0fa])  # concatenating the two dataframes

    # and remove it from the dataset
    df_rest = df[~df["ExamID"].isin(df_sample["ExamID"])]

    return df_rest[features + ["ExamID", "KEY_ANAGRAFE", "FA"]]


def lr_params_search(x, y, key_anagrafe, n_folds, n_iter, random_seed):
    """Hyperparameters search for logistic regression"""
    params = {
        'solver': ['newton-cg'],  # newton-cg is always chosen
        'penalty': ['l2'],
        'C': uniform(0.01, 100)
    }
    # parameters search for LR
    model = LogisticRegression(max_iter=10000, random_state=random_seed)
    cv = StratifiedGroupKFold(n_splits=n_folds)
    grid_search = RandomizedSearchCV(estimator=model,
                                     n_iter=n_iter,
                                     param_distributions=params,
                                     n_jobs=-1,
                                     cv=cv.split(x, y, key_anagrafe),
                                     scoring='roc_auc',
                                     verbose=1)
    grid_search.fit(x, y)
    return grid_search


def xgb_params_search(x, y, key_anagrafe, n_folds, n_iter, random_seed):
    """Hyperparameters search for XGBoost"""
    params = {
        "n_estimators": randint(100, 300),  # default 100
        "max_depth": randint(3, 10),  # default 3
        "learning_rate": uniform(0.01, 0.2),  # default 0.1
        "min_child_weight": randint(0, 10),
        "gamma": uniform(0, 10),
        "colsample_bytree": uniform(0.7, 0.3),
        "subsample": uniform(0.6, 0.4),
        "max_delta_step": [0, 1, 5, 10]
    }
    # parameters search for LR
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', tree_method='hist', random_state=random_seed)
    cv = StratifiedGroupKFold(n_splits=n_folds)
    rand_search = RandomizedSearchCV(estimator=model,
                                     n_iter=n_iter,
                                     param_distributions=params,
                                     n_jobs=-1,
                                     cv=cv.split(x, y, key_anagrafe),
                                     scoring='roc_auc',
                                     verbose=1)
    rand_search.fit(x, y)
    return rand_search


def main():
    # Silence all warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

    # model name: lr or xgb
    model_name, n_folds, n_iter, random_seed = get_args()

    # get dataset to perform hyperparameters search
    features = ['P_AXIS', 'P_OFFSET', 'P_ONSET', 'PR_INT', 'QRS_AXIS', 'QRS_OFFSET', 'QRS_ONSET', 'QTC_INT',
                'T_AXIS', 'T_OFFSET', 'V_RATE']
    df = load_dataset(features, random_seed)

    # parameters search
    if model_name == "lr":
        params_search = lr_params_search(df[features], df["FA"], df["KEY_ANAGRAFE"], n_folds, n_iter, random_seed)
    elif model_name == "xgb":
        params_search = xgb_params_search(df[features], df["FA"], df["KEY_ANAGRAFE"], n_folds, n_iter, random_seed)

    print("Best iteration: AUC {:f} using".format(params_search.best_score_))
    pprint.pprint(params_search.best_params_)

    # Extracting the parameters of the best model trained
    best_params = params_search.best_params_

    # Saving best parameters to a json file
    path_results = "results/" + model_name + "/"
    filename = "best_params.json"

    if not os.path.exists(path_results):  # create directory if it doesn't exist
        os.makedirs(path_results)

    with open(path_results + filename, 'w') as f:
        json.dump(best_params, f)


if __name__ == "__main__":
    main()
