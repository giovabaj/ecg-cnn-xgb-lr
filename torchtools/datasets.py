import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class EcgDataset(Dataset):
    """Dataset definition for the binary classification problem"""
    def __init__(self, path_ecg, path_df):
        super().__init__()
        self.path_ecg = path_ecg
        self.path_df = path_df

        # define class attributes
        self.X = None
        self.labels = None
        self.exids = None
        self.key_anagrafe = None

        # load ECGs and labels
        self.load_ecgs()
        self.load_labels()

    def __len__(self):
        """Total number of samples"""
        return len(self.exids)

    def __getitem__(self, index):
        """Generates one sample of data+label"""
        ecg_signal = self.X[index]
        label = self.labels[index]
        return ecg_signal, label

    def load_labels(self):
        """Load labels and patient keys"""
        ecg_and_labels = np.load(self.path_ecg)                     # Load npz file (with ECG matrix and exam IDs)
        df_data = pd.read_csv(self.path_df, encoding="iso-8859-1")  # Import csv file with all ECG info
        self.exids = ecg_and_labels[ecg_and_labels.files[1]]        # Extract exam IDs from files
        # Extract labels and patient keys
        exids_series = pd.Series(self.exids, name="ExamID")
        self.labels = torch.Tensor(pd.merge(exids_series, df_data, on="ExamID", how="left")["FA"].values).to(
            torch.int64)
        self.key_anagrafe = pd.merge(exids_series, df_data, on="ExamID", how="left")["KEY_ANAGRAFE"].values

    def load_ecgs(self):
        """Load ECG signals"""
        ecg_and_labels = np.load(self.path_ecg)                 # Load npz file
        self.X = torch.Tensor(ecg_and_labels[ecg_and_labels.files[0]])  # Extract ECG-signal matrix
