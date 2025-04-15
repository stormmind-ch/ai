from torch.utils.data import Dataset
import torch
import os
import orjson as oj
import polars as pl
from datetime import date as Date
import unicodedata
import numpy as np
from datetime import datetime


def normalize_text(text):
    return unicodedata.normalize("NFKC", text).replace("−", "-").strip().lower()


class StormDamageDataset(Dataset):
    def __init__(self, main_data_path: str, weather_data_dir: str, timespan: int, start_train: str, start_val: str, start_test: str, downsampling_rate:float=None):
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
        self.weather_data_dir = weather_data_dir
        self.timespan = timespan
        self.municipalities, self.dates, self.damages = self._load_main_dataset_to_numpy(main_data_path, downsampling_rate)
        self.total_rows = len(self.municipalities)
        self.weather_cache = self._preload_weather_data()
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

        if weather_features is None:
            raise ValueError(f"No weather features loaded for {municipality} and {date}")
        if np.any(np.isnan(weather_features)):
            raise ValueError(f"NaN values in the loaded weather features: {weather_features}")

        date_features = np.array([date.month])
        feature_vector = np.concatenate([weather_features, date_features])
        feature_vector = self.normalize_features(feature_vector)
        label = torch.tensor(int(damage), dtype=torch.int64)

        return feature_vector, label

    def normalize_features(self, features):
        return torch.tensor((features - self.mean) / self.std, dtype=torch.float32)

    def _load_main_dataset_to_numpy(self, main_data_path, downsampling_majority_ratio=None):
        df = pl.read_csv(main_data_path)

        if downsampling_majority_ratio is not None:
            majority_df = df.filter(pl.col("combination_damage_mainprocess") == 0.0)
            minority_df = df.filter(pl.col("combination_damage_mainprocess") != 0.0)
            sample_size = int(downsampling_majority_ratio * majority_df.height)

            downsampled_majority_df = majority_df.sample(n=sample_size, with_replacement=False, shuffle=True)

            df = pl.concat([downsampled_majority_df, minority_df]).sort("Date")

        # Convert to NumPy
        municipalities = df["Municipality"].to_numpy()
        dates = df["Date"].to_numpy()
        damages = df["combination_damage_mainprocess"].to_numpy()
        df.clear()
        return municipalities, dates, damages

    def _compute_normalization_stats(self, indices):
        features = []

        for idx in indices:
            municipality = self.municipalities[idx]
            date_str = self.dates[idx]
            date = Date.fromisoformat(date_str)
            normalized_municipality = normalize_text(municipality)
            weather_features = self._get_weather_features(normalized_municipality, date)
            date_features = np.array([date.month])

            if weather_features is not None:
                full_feature_vector = np.concatenate([weather_features, date_features])
                features.append(full_feature_vector)

        feature_matrix = np.stack(features)
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
        return mean, std

    def _preload_weather_data(self):
        """
        Preloads all weather data into a dictionary for fast lookup.
        """
        weather_cache = {}

        files = [f for f in os.listdir(self.weather_data_dir) if f.endswith(".json")]
        for file in files:
            if file.endswith(".json"):
                municipality = file.replace(".json", "")
                municipality_normalized = normalize_text(municipality)
                file_path = os.path.join(self.weather_data_dir, file)
                with open(file_path, "rb") as f:
                    try:
                        raw_data = oj.loads(f.read())
                        weather_cache[municipality_normalized] = {
                            "temperature_2m_mean": np.array(raw_data["daily"]["temperature_2m_mean"], dtype=np.float32),
                            "sunshine_duration": np.array(raw_data["daily"]["sunshine_duration"], dtype=np.float32),
                            "rain_sum": np.array(raw_data["daily"]["rain_sum"], dtype=np.float32),
                            "snowfall_sum": np.array(raw_data["daily"]["snowfall_sum"], dtype=np.float32),
                        }
                    except:
                        raise Exception(f"Could not preload data for {municipality_normalized}")
        return weather_cache

    def _get_weather_features(self, municipality: str, date: Date):
        """
        Retrieves weather data from preloaded cache.
        """
        first_date = Date(1972, 1, 1)
        delta = date - first_date
        end_date = delta.days + 1
        start_date = end_date - self.timespan
        municipality_normalized = normalize_text(municipality)

        if municipality_normalized not in self.weather_cache:
            raise Exception(f"{municipality_normalized} does not exist in weather data cache")

        data = self.weather_cache[municipality_normalized]
        temperature_2m_mean = data['temperature_2m_mean'][start_date:end_date]
        sunshine_duration = data['sunshine_duration'][start_date:end_date]
        rain_sum = data['rain_sum'][start_date:end_date]
        snowfall_sum = data['snowfall_sum'][start_date:end_date]

        return np.concatenate([temperature_2m_mean, sunshine_duration, rain_sum, snowfall_sum])