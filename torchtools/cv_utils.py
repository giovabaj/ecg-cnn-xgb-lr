import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedGroupKFold


def create_cv_folds(labels, key_anagrafe, n_splits):
    """
    Split dataset into folds with non-overlapping groups based on AF outcome. The folds are made by preserving the
    percentage of samples for each class. The ECGs of the same patient will appear only in a single fold.
    Input:
     - labels: AF label (0-1)
     - key_anagrafe: patients id
     - n_splits: number of folds
    Output:
     - folds: array composed of "n_splits" arrays, each with the indexes of a fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    folds = np.zeros(n_splits, dtype=object)
    for i, (train, test) in enumerate(sgkf.split(np.arange(len(labels)), labels, groups=key_anagrafe)):
        folds[i] = test.astype(int)
    return folds


def get_sets_from_cv_folds(i_fold_test, cv_folds, dataset, path, ratio=None, random_seed=None):
    """Assign each fold to train, validation or test set. Training set is under-sampled to the required fraction if
    required.
    Input:
     - i_fold_test: index of the fold that will constitute the test fold. The val set is made by the following
       fold. The rest of the folds will constitute the training set.
     - cv_folds: array of arrays, in which each sub-array contains the index of a single fold
     - dataset: EcgDataset object
     - path: path where to save exam IDs for each set
     - ratio: undersampling ratio of the training set. If "None", no undersampling is applied (Default: None)
     - random_seed: necessary only if undersampling is performed (Default: None)
    Output:
     - indices_fold: dictionary with "train", "val" and "test" as keys and indices of each fold as values
    """
    n_folds = len(cv_folds)
    all_indices_folds = np.arange(n_folds)    # array with the cv-fold indices

    # assign each fold to one of train, test and validation (based on test)
    i_folds = {"test": [i_fold_test], "val": [(i_fold_test + 1) % n_folds]}
    i_folds["train"] = np.delete(all_indices_folds, i_folds["test"] + i_folds["val"])

    # Get all the dataset indexes for each set
    indices_set = {}
    for set_name in ["train", "test", "val"]:
        indices_set[set_name] = np.concatenate(cv_folds[i_folds[set_name]])

    # under sampling the training set to the required ratio
    if ratio is not None:
        print("---- Undersampling the training set to the ratio: ", ratio)
        print("1FA ratio before RUS: ",
              (dataset.labels[indices_set["train"]].sum() / len(indices_set["train"])).item())
        random_under_sampler = RandomUnderSampler(sampling_strategy=ratio / (1 - ratio), random_state=random_seed)
        indices_set["train"], _ = random_under_sampler.fit_resample(indices_set["train"].reshape(-1, 1),
                                                                    dataset.labels[indices_set["train"]])
        indices_set["train"] = indices_set["train"].reshape(-1, )
        print("1FA ratio after RUS: ",
              (dataset.labels[indices_set["train"]].sum() / len(indices_set["train"])).item())

    # Save exam IDs of each set
    for set_name in ["train", "test", "val"]:
        np.save(f"{path}/exids_{set_name}", dataset.exids[indices_set[set_name]])

    return indices_set


def save_and_check_sets(indices_sets, dataset, path):
    """
    Save examIDs of train, validation and test sets obtained combining the cv-folds, and check the sets (FA fraction and
    non-overlapping exmas/patients).
    Input:
     - indices_sets: dictionary with "train", "val" and "test" as keys and the list of indices composing each fold as
                     items
     - dataset: object of EcgDataset class
     - path: path where to save exam IDs
    Output:
     - print info about each set and save the examIDs
    """
    for set_name in ["train", "val", "test"]:
        print("- ", set_name.capitalize())  # Print some info on the folds
        print("ECGs in " + set_name + " fold: ", dataset.X[indices_sets[set_name]].shape[0])
        print("AF fraction in " + set_name + " fold: ", ((dataset.labels[indices_sets[set_name]]==1).sum()/len(indices_sets[set_name])).item())
        indices_other_sets = [indices for other_set, indices in indices_sets.items() if other_set not in [set_name]]  # list of 2 lits
        indices_other_sets = [item for sublist in indices_other_sets for item in sublist]
        print("Patients shared with other folds: ",
          np.isin(dataset.key_anagrafe[indices_sets[set_name]], dataset.key_anagrafe[indices_other_sets]).sum())

        np.save(path + "examIDs_" + set_name, dataset.exids[indices_sets[set_name]])
