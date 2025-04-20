from torch.utils.data import Dataset
import torch
from datetime import date as Date
import numpy as np
from datetime import datetime
from ai_engineering.datasets.utils.dataset_utils import normalize_text, preload_weather_data
from utils.dataset_utils import load_main_dataset_to_numpy, preload_coordinates, date_features_sincos_normalisation, get_weather_features


class StormDamageDataset(Dataset):
    def __init__(self, main_data_path: str, weather_data_dir: str, timespan: int, start_train: str, start_val: str, start_test: str,municipality_coordinates_file : str, downsampling_rate:float=None):
        """
        Args:
            main_data_path (str): Path to the main dataset CSV file.
            weather_data_dir (str): Directory containing weather data files for each municipality.
            timespan (int): Number of past days to consider for weather data.
            start_train (str): start date of the training data as a string
            start_val (str): start date of the val data as a string
            start_test (str): start date of the test data as a string
            downsampling_rate: the ratio which should stay during the down sampling of the majority (no damage) class.
        """
        self.municipalities_coordinates = preload_coordinates(municipality_coordinates_file)
        self.weather_data_dir = weather_data_dir
        self.timespan = timespan
        self.municipalities, self.dates, self.damages = load_main_dataset_to_numpy(main_data_path, downsampling_rate)
        self.total_rows = len(self.municipalities)
        self.weather_cache = preload_weather_data(weather_data_dir)
        date_objs = np.array([datetime.strptime(d, "%Y-%m-%d") for d in self.dates])


        # Store indices based on date ranges
        self.train_indices = np.where((date_objs >= datetime.strptime(start_train, "%Y-%m-%d")) &
                                      (date_objs < datetime.strptime(start_val, "%Y-%m-%d")))[0]

        self.val_indices = np.where((date_objs >= datetime.strptime(start_val, "%Y-%m-%d")) &
                                    (date_objs < datetime.strptime(start_test, "%Y-%m-%d")))[0]

        self.test_indices = np.where(date_objs >= datetime.strptime(start_test, "%Y-%m-%d"))[0]
        self.mean, self.std = self._compute_normalization_stats(self.train_indices)

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        municipality = self.municipalities[idx]
        date_str = self.dates[idx]
        damage = self.damages[idx]

        normalized_municipality = normalize_text(municipality)
        date = Date.fromisoformat(date_str)
        weather_features = self._get_weather_features(normalized_municipality, date)
        coords = np.array(self.municipalities_coordinates[municipality])

        if weather_features is None:
            raise ValueError(f"No weather features loaded for {municipality} and {date}")
        if np.any(np.isnan(weather_features)):
            print(f"[Warning] NaNs found in weather features, replacing with 0: for {municipality}")
            weather_features = np.nan_to_num(weather_features, nan=0.0)

        date_features = date_features_sincos_normalisation(date)
        weather_features = self.weather_features_zscore_normalisation(weather_features)
        feature_vector = np.concatenate([weather_features, date_features, coords])
        feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
        label = torch.tensor(int(damage), dtype=torch.int64)

        return feature_vector, label

    def weather_features_zscore_normalisation(self, features):
        return torch.tensor((features - self.mean) / self.std, dtype=torch.float32)

    def _compute_normalization_stats(self, indices):
        if len(indices) == 0:
            raise ValueError("Indices must have at least 1 value")

        features = []

        for idx in indices:
            municipality = self.municipalities[idx]
            date_str = self.dates[idx]
            date = Date.fromisoformat(date_str)
            normalized_municipality = normalize_text(municipality)
            weather_features = get_weather_features(normalized_municipality, date)

            if weather_features is not None:
                features.append(weather_features)

        feature_matrix = np.stack(features)
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
        return mean, std

