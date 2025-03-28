from torch.utils.data import Dataset
import torch
import os
import orjson as oj
import polars as pl
from datetime import date as Date

class StormDamageDataset(Dataset):
    def __init__(self, main_data_path: str, weather_data_dir: str, timespan: int):
        """
        Args:
            main_data_path (str): Path to the main dataset CSV file.
            weather_data_dir (str): Directory containing weather data files for each municipality.
            timespan (int): Number of past days to consider for weather data.
        """
        self.weather_data_dir = weather_data_dir
        self.timespan = timespan

        # Lazy load data instead of reading into memory
        self.data_df = pl.read_csv(main_data_path).to_dict(as_series=False)
        self.total_rows = len(self.data_df["Municipality"])

        # Preload weather data in memory
        self.weather_cache = self._preload_weather_data()

        # Ensure error log directory exists
        os.makedirs("helper_files", exist_ok=True)

    def __len__(self):
        """Returns the total number of samples."""
        return self.total_rows

    def __getitem__(self, idx):
        """Loads a row, merges with weather data, and returns tensors."""
        municipality = self.data_df["Municipality"][idx]
        date_str = self.data_df["Date"][idx]
        damage = self.data_df["combination_damage_mainprocess"][idx]

        date = Date.fromisoformat(date_str)
        weather_features = self._get_weather_features(municipality, date)

        if not weather_features or any(x is None for x in weather_features):
            self._log_error(municipality, "Invalid or None in weather features")
            return None
            # raise ValueError(f"No weather features loaded for {municipality} and {date}")

        try:
            damage = int(float(damage))
        except ValueError:
            self._log_error(municipality, f"Non-numeric damage value: {damage}")
            return None
            #raise ValueError(f"Non-numeric value found in 'damage': {damage}")

        # Convert to tensors
        feature_vector = torch.tensor(weather_features, dtype=torch.float32)
        label = torch.tensor(damage, dtype=torch.int64)
        return feature_vector, label

    def _preload_weather_data(self):
        """Preloads all weather data into a dictionary for fast lookup."""
        weather_cache = {}
        for file in os.listdir(self.weather_data_dir):
            if file.endswith(".json"):
                municipality = file.replace(".json", "")
                file_path = os.path.join(self.weather_data_dir, file)
                with open(file_path, "rb") as f:
                    try:
                        weather_cache[municipality] = oj.loads(f.read())
                    except:
                        print("FICKEN")
                        self._log_error(municipality, "Could not read weather data")
                        continue
        return weather_cache

    def _get_weather_features(self, municipality: str, date: Date):
        """
        Retrieves weather data from preloaded cache.
        """
        first_date = Date(1972, 1, 1)
        delta = date - first_date
        end_date = delta.days
        start_date = end_date - self.timespan

        if municipality not in self.weather_cache:
            self._log_error(municipality, "Missing weather file")
            return None

        try:
            data = self.weather_cache[municipality]
            temperature_2m_mean = data['daily']['temperature_2m_mean'][start_date:end_date]
            sunshine_duration = data['daily']['sunshine_duration'][start_date:end_date]
            rain_sum = data['daily']['rain_sum'][start_date:end_date]
            snowfall_sum = data['daily']['snowfall_sum'][start_date:end_date]
        except:
            self._log_error(municipality, "Errors in weather data")
            return None

        return temperature_2m_mean + sunshine_duration + rain_sum + snowfall_sum

    def _log_error(self, municipality: str, message: str):
        """Logs errors to a file."""
        with open("helper_files/problem_mun.csv", "a") as file:
            file.write(f"{municipality},{message}\n")
