# Comparison of discrimination and calibration performance of ECG-based machine learning models for prediction of new-onset atrial fibrillation
Python code for the paper "Comparison of discrimination and calibration performance of ECG-based machine learning models for prediction of new-onset atrial fibrillation
", Baj Giovanni et al., _BMC Medical Research Methodology_ (2023).

## Data organization
Data has to be organized as follows:
- information about ECGs has to be saved in a csv file. Necessary columns are:
  - _ExamID_ key that uniquely identifies the ECG
  - _KEY_ANAGRAFE_ patient's registry key
  - _FA_ binary variable that indicates if the patient developed atrial fibrillation within 5 years after the exam
- ECG signals have to be stored in a single directory as numpy files, and named with their key (_ExamID_ in our case). Each ECG is a 12-lead signal sampled at 1000 Hz for 10s.

## Scripts
`run_all.py` runs all the experiments reported in the paper (dependence on the sample size and dependence on the
balancing ratio).

`plot_results.py` loads the results from previous computations and makes the plots reported in the paper. To use this
script one has to create a json file where directories whose results have to be analyzed are listed (for more details
look at the code).

`make_dataset.py` creates a single dataset that ca be used to train the models. It takes as input the number of ECGs to be
included in the dataset, the path of the csv file where ECGs information is stored, and the path where ECG signals are
stored. Once the dataset is created, the script saves in the "datasets" directory a compressed numpy file (.npz) that
contains two numpy arrays: a matrix of shape (N, 12, 5000) with the ECG signals (where N is the number of ECGs), and a
second array with the corresponding exam IDs.

`cnn_training.py` trains CNN models in a cross-validation setting, for a specified configuration

`xgb_lr_training.py` trains XGB or LR models with the same dataset and cross-validation folds of a previously trained
CNN model.

`hyperparameters_search.py` performs hyperparameters search for XGB and LR models

`xgb_lr_training.py` trains XGB or LR model 

