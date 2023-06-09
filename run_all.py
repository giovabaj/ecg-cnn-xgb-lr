import os


def main():
    path_ecg_info = '/coorte/ASUGI/ECG_FA_5Y_v06.csv'  # path of the csv file with ECG info

    # ------ CNN ------
    # -- Size dependence --
    sizes = [1000, 2000, 5000, 10000, 50000, 100000, 150000]

    # Train the CNN models
    for size in sizes:
        # Make dataset
        os.system(f'python make_dataset.py --n_samples {size}')
        # Train CNN with and without undersampling the training set
        os.system(f'python cnn_training.py --n_samples {size} --path_df {path_ecg_info} --patience 3')
        os.system(f'python cnn_training.py --n_samples {size} --undersampling_flag --ratio 0.5'
                  f'--path_df {path_ecg_info}')

    # -- Undersampling ratio dependence --
    size = 100000           # fixed size
    ratios = [0.25, 0.375]  # undersampling ratios (0.125 and 0.5 already considered)

    for ratio in ratios:
        os.system(f'python cnn_training.py --n_samples {size} --undersampling_flag --ratio {ratio}'
                  f'--path_df {path_ecg_info}')

    # ------ XGB and LR ------
    # Hyperparameters search for XGB and LR
    os.system('python hyperparameters_search.py --model_name xgb --n_iter 100000')
    os.system('python hyperparameters_search.py --model_name lr --n_iter 100000')

    # Train XGB and LR on the same datasets and cv folds used for the CNN (all directories in results/cnn are processed)
    for directory in os.listdir("results/cnn"):
        os.system(f'python xgb_lr_training.py --folder {directory} --model_name xgb')
        os.system(f'python xgb_lr_training.py --folder {directory} --model_name lr')


if __name__ == "__main__":
    main()
