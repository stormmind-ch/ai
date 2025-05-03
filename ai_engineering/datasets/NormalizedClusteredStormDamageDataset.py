from datasets.ClusteredStormDamageDatasetBinaryLabels import ClusteredStormDamageDatasetBinaryLabels
import numpy as np
import torch

class NormalizedClusteredStormDamageDataset(ClusteredStormDamageDatasetBinaryLabels):
    """
    Decorator / Wrapper for ClusteredStormDamageDataset to normalize the features
    """

    def __init__(self, main_data_path: str, weather_data_dir: str, municipality_coordinates_path: str, n_clusters: int,
                 n_sequences: int,
                 split: str = None, test_years: int = 2,
                 damage_distribution: list[float] = [0.90047344, 0.06673681, 0.03278976],
                 damage_weights: dict[int:float] = None, grouping_calendar: str = 'weekly',
                 mean=None, std=None):
        super().__init__(main_data_path, weather_data_dir, municipality_coordinates_path, n_clusters, n_sequences,
                         split, test_years, damage_distribution, damage_weights, grouping_calendar)
        self.mean, self.std = mean, std

        if self.mean is None or self.std is None:
            self.mean, self.std = self.get_mean_std()


    def __getitem__(self, idx):
        features, label = super().__getitem__(idx)
        features = self.normalize_features(features)
        return features, label

    def get_mean_std(self):
        X = []
        y = []
        for i in range(len(self)):
            feat, label = super().__getitem__(i)
            cur = feat[0].squeeze().numpy()  # current week only (t=0), shape [F]
            X.append(cur)
            y.append(label)
        X_all = np.stack(X)  # shape [N, F]
        y_all = np.array(y) # shape [N]
        X_all[:, 5] = y_all # replacing the masked labels with the correct labels.
        mean = np.mean(X_all, axis=0)
        std = np.std(X_all, axis=0)
        std[std == 0] = 1e-8
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

    def normalize_features(self, features):
        mean = torch.tensor(self.mean, device=features.device)
        std = torch.tensor(self.std,  device=features.device)
        features_normalized = (features - mean) / std
        return features_normalized



