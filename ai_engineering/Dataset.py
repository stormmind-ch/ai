from torch.utils.data import Dataset
import torch
import os
import orjson as oj
import polars as pl
from datetime import date as Date

class StormDamageDataset(Dataset):
    def __init__(self, main_data_path: str, weather_data_dir: str, timespan: int, chunk_size: int = 1000000):
        """
        Args:
            main_data_path (str): Path to the main dataset CSV file.
            weather_data_dir (str): Directory containing weather data files for each municipality.
            chunk_size (int): Number of rows to process at a time from the main dataset.
        """
        self.main_data_path = main_data_path
        self.weather_data_dir = weather_data_dir
        self.chunk_size = chunk_size
        self.timespan = timespan

        self.data_df = pl.read_csv(self.main_data_path)
        self.total_rows = self.data_df.shape[0]


    def __len__(self):
        """Returns the total number of chunks."""
        return self.total_rows

    def __getitem__(self, idx):
        """Loads a batch of main data, merges with weather data, and returns tensors."""
        row = self.data_df[idx]
        municipality = row["Municipality"][0]
        date_str = row["Date"][0]
        damage = row["combination_damage_mainprocess"][0]

        date = Date.fromisoformat(date_str)
        weather_features = self._load_weather_data(municipality, date)

        if weather_features is None:
            raise ValueError(f"No weather features loaded for {municipality} and {date}")
        try:
            damage = int(float(damage))
        except ValueError:
            raise ValueError(f"Non-numeric value found in 'damage': {damage}")

        # Convert to tensors
        try:
            feature_vector = torch.tensor(weather_features, dtype=torch.float32)
            label = torch.tensor(damage, dtype=torch.int64)
            return feature_vector, label
        except TypeError:
            with open('helper_files/problem_mun.csv', 'wb') as file:
                file.writelines([municipality])



    def _load_weather_data(self, municipality: str, date: Date):
        """
        Loads weather data for a specific municipality.
        """
        first_date = Date(1972, 1,1)
        delta = date - first_date
        end_date = int(delta.days)
        start_date = end_date - self.timespan
        weather_file = os.path.join(self.weather_data_dir, f"{municipality}.json")

        if not os.path.exists(weather_file):
            raise FileNotFoundError(f"There is a missing weather file for {municipality}")

        with open(weather_file, "rb") as f:
            try:
                data = oj.loads(f.read())
            except:
                raise ValueError(f"data could not be read of {municipality}")
        try:
            temperature_2m_mean = data['daily']['temperature_2m_mean'][start_date: end_date]
            sunshine_duration = data['daily']['sunshine_duration'][start_date: end_date]
            rain_sum = data['daily']['rain_sum'][start_date: end_date]
            snowfall_sum = data['daily']['snowfall_sum'][start_date: end_date]
        except:
            raise Exception(f"There are errors in the weather data for {municipality}")

        weather_features = temperature_2m_mean + sunshine_duration + rain_sum + snowfall_sum
        return weather_features