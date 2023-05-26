import os
import json
import argparse
from datetime import date


def get_args():
    """
    Get arguments from terminal.
    Output: configuration arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=120, help="Maximum number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--dropout_rate', type=float, default=0.3, help="Dropout rate")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience")
    parser.add_argument('--path_df', type=str, default='/coorte/ASUGI/ECG_FA_5Y_v06.csv', help="Dataframe path")
    parser.add_argument('--n_folds', type=int, default=10, help="Number of CV folds")
    parser.add_argument('--n_samples', type=int, default=1000, help="Number of ECG samples")
    parser.add_argument('--undersampling_flag', default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to undersample or not")
    parser.add_argument('--ratio', type=float, default=None, help="Undersampling ratio")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    dropout_rate = args.dropout_rate
    patience = args.patience
    path_df = args.path_df
    n_folds = args.n_folds
    n_samples = args.n_samples
    undersampling_flag = args.undersampling_flag
    ratio = args.ratio
    random_seed = args.random_seed

    path_ecg = "datasets/X_" + str(n_samples) + ".npz"

    # create folder to save results_binary
    path = get_path_results(parent_results_folder="results/cnn", n=n_samples, undersampling=undersampling_flag,
                            ratio=ratio)

    # saving configs to a json file
    config_dict = vars(args)
    config_dict["path_ecg"] = path_ecg
    with open(path + '/config.json', 'w') as outfile:
        json.dump(config_dict, outfile)

    return path, n_epochs, batch_size, lr, dropout_rate, patience, path_ecg, path_df, n_folds,\
        undersampling_flag, ratio, random_seed


def get_path_results(parent_results_folder, n, undersampling=False, ratio=None):
    """
    Make results directory.
    Input:
     - parent_results_folder:
     - n: number of samples
     - undersampling: boolean that indicates whether training set undersampling is performed (Default: False)
     - ratio: undersampling ratio (Default: None)
    Output:
     - results folder path
    """
    # Set directory name based on number of samples and whether undersampling is required
    if undersampling:
        results_folder = parent_results_folder + "/N" + str(n) + "_balanced" + str(ratio).replace(".", "") + "_" + str(
            date.today())
    else:
        results_folder = parent_results_folder + "/N" + str(n) + "_" + str(date.today())

    path_res = os.getcwd() + "/" + results_folder

    # Make directory avoiding over-writing
    if not os.path.exists(path_res):
        path_res = path_res + "/"
        os.makedirs(path_res)  # if dir does not exist, I create it
    else:  # if it exists, I create "dir_1", if already existing I create "dir_2"....
        i = 1
        path_res = path_res + "_" + str(i) + "/"
        while os.path.exists(path_res):
            i = i + 1
            path_res = path_res + "_" + str(i) + "/"
        os.makedirs(path_res)

    return path_res

