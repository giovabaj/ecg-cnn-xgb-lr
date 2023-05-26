import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from torch.nn import AvgPool1d
import torch
import argparse


def main():
    # parsing arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--path_df', type=str, default='/coorte/ASUGI/ECG_FA_5Y_v06.csv')
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    # Importing dataframe
    df = pd.read_csv(args.path_df, encoding="iso-8859-1")
    label = "FA"

    # Subsampling the dataframe to the required size, maintaining the original FA fraction
    frac_1fa = (df[label] == 1).sum() / len(df)  # FA fraction overall
    n_1fa = int(frac_1fa * args.n_samples)       # number of 1-FA exams to subsample
    n_0fa = args.n_samples - n_1fa               # number of 0-FA exams
    df_1fa = df[df[label] == 1].sample(n_1fa, random_state=args.random_seed)  # Sampling classes in separate steps
    df_0fa = df[df[label] == 0].sample(n_0fa, random_state=args.random_seed)
    df_sample = pd.concat([df_1fa, df_0fa])      # concatenating the two dataframes

    waveform_dir = 'D:/WAVEFORMS/'  # folder with ECG signals
    dataset_dir = 'datasets/'       # folder where to save the dataset

    exams = df_sample["ExamID"].values.astype(int)  # Exam IDs to extract
    filenames = np.array([os.path.join(waveform_dir, str(indx) + '.npy') for indx in exams])

    freq_resample = 500  # resampling frequency
    n_points_resample = 10 * freq_resample  # number of time points after resampling
    avg_pool = AvgPool1d(kernel_size=3, stride=2, padding=1)

    ecgs = np.empty([len(exams), 12, n_points_resample], dtype='float32')  # 3D matrix where to store ECGs
    for i, filename in enumerate(filenames):  # Loop over exams
        try:
            if i % 100 == 0:
                print(f'\rProcessing file number {i + 1}', end='\r')
            ecg = np.load(filename).astype('float32').swapaxes(0, 1)
            ecgs[i] = avg_pool(torch.from_numpy(ecg)).numpy().astype('float32')  # Resampling to 500Hz
        except Exception as error:
            print(f'\nerror: {filename} - {error}')

    # saving ECGs and exam_IDs in the same npz file
    filename = os.path.join(dataset_dir, 'X_' + str(args.n_samples))
    np.savez_compressed(filename, ecgs, exams)


if __name__ == "__main__":
    main()
