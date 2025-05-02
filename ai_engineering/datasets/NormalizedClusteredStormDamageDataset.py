from torch.utils.data.dataset import Dataset
from datasets.ClusteredStormDamageDataset import ClusteredStormDamageDataset
import numpy as np
from numpy import typing as npt
import torch

class NormalizedClusteredStormDamageDataset(Dataset):
    """
    Decorator / Wrapper for ClusteredStormDamageDataset to normalize the features
    """
    def __init__(self, base_dataset: ClusteredStormDamageDataset, mean:npt.NDArray=None, std: npt.NDArray=None):
        self.base_dataset = base_dataset
        self.mean, self.std = mean, std

        if not mean or not std:
            self.mean, self.std = self.get_mean_std()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        features, label = self.base_dataset[idx]
        features = self.normalize_features(features)
        return features, label

    def get_mean_std(self):
        X = []
        for feat, _ in self.base_dataset:
            cur = feat[0].squeeze().numpy()  # current week only (t=0), shape [F]
            X.append(cur)
        X_all = np.stack(X)  # shape [N, F]
        mean = np.mean(X_all, axis=0)
        std = np.std(X_all, axis=0)
        std[std == 0] = 1e-8
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

    def normalize_features(self, features):
        mean = torch.tensor(self.mean, device=features.device)
        std = torch.tensor(self.std,  device=features.device)
        features_normalized = (features - mean) / std
        return features_normalized



