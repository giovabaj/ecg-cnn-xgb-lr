import os

# ------ CNN ------
# -- Size dependence --
sizes = [150, 200, 250]

# Train the CNN models
for size in sizes:
    # Make dataset
    os.system(f'python make_dataset.py --n_samples {size}')
    # Train CNN with and without undersampling in training set
    os.system(f'python cnn_training.py --n_samples {size} --patience 2')
    os.system(f'python cnn_training.py --n_samples {size} --undersampling_flag --ratio 0.5 --patience 2')

# -- Undersampling ratio dependence --
size = 150              # fixed size
ratios = [0.25, 0.375]  # undersampling ratios (0.125 and 0.5 already considered)

for ratio in ratios:
    os.system(f'python cnn_training.py --n_samples {size} --undersampling_flag --ratio {ratio} --patience 2')

# ------ XGB and LR ------
# Hyperparameters search for XGB and LR
os.system('python hyperparameters_search.py --model_name xgb --n_iter 5')
os.system('python hyperparameters_search.py --model_name lr --n_iter 5')

# Train XGB and LR on the same datasets and cv folds used for the CNN
for directory in os.listdir("results/cnn"):
    os.system(f'python xgb_lr_training.py --folder {directory} --model_name xgb')
    os.system(f'python xgb_lr_training.py --folder {directory} --model_name lr')
