"""Script to train CNN models with cross-validation"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchtools.architectures import GoodfellowNet
from torchtools.datasets import EcgDataset
from torchtools.train_functions import train_model, model_predictions
from evaluation import MetricsCV
from torchtools.cv_utils import get_sets_from_cv_folds, create_cv_folds
from torchtools.arg_parser import get_args


def main():
    # Get arguments from terminal
    path_results, n_epochs, batch_size, lr, dropout_rate, patience, path_ecg, path_df, n_folds, \
        undersampling_flag, ratio, random_seed = get_args()

    # Data Loading
    dataset = EcgDataset(path_ecg, path_df)

    # Setting the computation device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(random_seed)

    # create cross-validation folds
    cv_folds = create_cv_folds(dataset.labels, dataset.key_anagrafe, n_folds)

    # Instantiate Metrics class for each set
    set_names = ["train", "val", "test"]
    metrics_dict = {}
    for set_name in set_names:
        metrics_dict[set_name] = MetricsCV(path_results, set_name)

    for i_fold_test in range(n_folds):  # Loop over CV folds

        # Create results folder for the actual cv-fold
        path_results_fold = path_results + "/CVfold_" + str(i_fold_test) + "/"
        os.makedirs(path_results_fold)

        # Save prints to a log file
        old_stdout = sys.stdout
        log_file = open(path_results_fold + "logfile.log", "w")
        sys.stdout = log_file

        # Get dictionary with indices of train/test/val sets. If required, Undersample the training set.
        indices_sets = get_sets_from_cv_folds(i_fold_test, cv_folds, dataset, path_results_fold, ratio, random_seed)

        # Define dataloaders
        dataloaders = {}
        for set_name in ["train", "val"]:
            dataloaders[set_name] = DataLoader(dataset, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(indices_sets[set_name]))

        # Instantiate model, loss function and optimizer
        model = GoodfellowNet(len_input=dataset.X.shape[2])  # Instantiate the model
        loss_fn = torch.nn.CrossEntropyLoss()  # Define the loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # -- TRAIN --
        model = train_model(model, dataloaders["train"], dataloaders["val"], loss_fn,
                            optimizer, n_epochs, device, patience=patience, path_res=path_results_fold)

        # -- TEST --
        for set_name in set_names:  # loop on train, val and test sets
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=indices_sets[set_name],
                                    shuffle=False)
            predictions = model_predictions(model, device, dataloader).cpu().numpy()[:, 1]
            # save results and compute metrics
            np.save(path_results_fold + "predictions_" + set_name, predictions)
            np.save(path_results_fold + "labels_" + set_name, dataset.labels[indices_sets[set_name]])
            metrics_dict[set_name].compute_metrics(dataset.labels[indices_sets[set_name]], predictions)

        # writing logs to the log file
        sys.stdout = old_stdout
        log_file.close()

    # Save overall metrics for all the sets
    for set_name in set_names:
        metrics_dict[set_name].save_results()


if __name__ == "__main__":
    main()
