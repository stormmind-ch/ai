import datetime
import numpy as np
import polars as pl
import torch
import wandb
from sympy.polys.numberfields.modules import to_col
from torch.utils.data.dataset import  Dataset
from datasets.utils.dataset_utils import preload_weather_data, merge_clusters_initial, normalize_text, get_weather_features, calculate_agg_weather_features
from datasets.utils.grouping_utils import group_damages
from datasets.utils.clustering_utils import make_clusters
from calendar import monthrange
from datasets.utils.classifing_damages_utils import make_bin_of_classes
from datetime import timedelta, datetime


def _load_main_data(main_data_path: str, municipality_coordinates_path,
                    k, damage_weights, n, grouping_calendar, damage_distribution: list):
    mun_dates_damage = pl.read_csv(main_data_path)
    mun_coordinates = pl.read_csv(municipality_coordinates_path)
    clusters_df, clusters = make_clusters(k, mun_coordinates)
    merged = merge_clusters_initial(clusters_df, mun_dates_damage)
    final = group_damages(merged, damage_weights,n, grouping_calendar)
    final, (low, mid) = make_bin_of_classes(final, damage_distribution)
    if wandb.run is not None:
        wandb.log({
            "low_threshold": low,
            "mid_threshold": mid
        })
    return final, clusters


def split_dataframe_by_time(df: pl.DataFrame, val_years : int,  test_years: int) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = df.sort("end_date")
    last_date = df.select(pl.col("end_date").max()).item()

    test_start = last_date.replace(year=last_date.year - test_years)
    val_start = last_date.replace(year=test_start.year -  val_years)

    train = df.filter(pl.col("end_date") < val_start)
    val = df.filter((pl.col("end_date") >= val_start) & (pl.col("end_date") < test_start))
    test = df.filter(pl.col("end_date") >= test_start)

    return train, val, test


def build_lookup(df:pl.DataFrame):
    lookup = {
        (df["Center_Municipality"][i], df["end_date"][i]): i
        for i in range(df.height)
    }
    return lookup


def get_past_week_dates(base_date, timespan):
    return [base_date - timedelta(weeks=i) for i in range(timespan + 1)]


class ClusteredStormDamageDataset(Dataset):
    def __init__(self, main_data_path : str, weather_data_dir: str, municipality_coordinates_path:str, n_clusters: int, n_sequences:int,
                 split: str = None,val_years: int = 2, test_years: int = 2, damage_distribution:list[float] = [0.90047344, 0.06673681, 0.03278976],
                 damage_weights:dict[int:float]=None, grouping_calendar: str = 'weekly', grouping_daily: int = None):
        """
        Args:
            main_data_path: Path to main data file containing a entry for each day and municipality and the damage
            weather_data_dir: Directory containing all weather information for each municipality in clusters as .json files.
            municipality_coordinates_path: Path to a csv file with all municipalities to their coordinates. Needed Columns: [Municipality, Latitude, Longitude]
            n_sequences: Amount of weeks or months (depending on grouping_calendar) which should be considered for a backward lookup
            n_clusters: Number of clusters which should be created
            split: 'test' or 'train'
            val_years: how many years should be used for the validation
            test_years: how many years should be used for the testing
            damage_distribution: distribution of the damages in the original Dataset.
            damage_weights: each damage (small, medium, large) can here be weighted for the summed damage in CHF
            grouping_calendar: If the data should be grouped weekly or monthly. eg. what one timestamp should include.
        """
        self.dataframe, clusters = _load_main_data(main_data_path, municipality_coordinates_path, n_clusters, damage_weights, n_sequences, grouping_calendar, damage_distribution)
        self.n_sequences = n_sequences
        self.timespan_calendar = grouping_calendar if grouping_calendar else None
        self.timespan_int = grouping_daily if grouping_daily else None
        self.weather_data_cache = preload_weather_data(weather_data_dir, clusters)
        train_df, val_df,  test_df = split_dataframe_by_time(self.dataframe, val_years, test_years)
        if split == 'train':
            self.dataframe = train_df
        elif split == 'val':
            self.dataframe = val_df
        elif split == 'test':
            self.dataframe = test_df
        else:
            self.dataframe = self.dataframe
        self.dataframe.sort("end_date")

        self.lookup = build_lookup(self.dataframe)

    def __len__(self):
        return self.dataframe.height

    def __getitem__(self, idx):
        """
        Returns the data of size:
            features: (sequence_length, 8) in the following order:
                temperature_2m_mean, sun_duration_mean, rain_sun, snow_sum, label (only of last sequence), month, latitude, longitude
            labels: (1)
        """
        if idx >= len(self):
            raise IndexError

        row = self.dataframe.row(idx, named=True)
        end_date = row['end_date']
        municipality = row['Center_Municipality']
        dates = get_past_week_dates(end_date, self.n_sequences)
        damage = row['damage_grouped']
        latitude = row['Cluster_Center_Lat']
        longitude = row['Cluster_Center_Long']

        # dates are from the current back to the one furthest in the past
        features = self.get_feature_sequence(dates, end_date, municipality, latitude, longitude)

        label = torch.tensor(damage, dtype=torch.long)

        return features, label


    def get_feature_sequence(self, dates, current_date, municipality, latitude, longitude):
        feature_sequence = []
        mask_sequence = []

        # dates are from the current back to the one furthest in the past
        for date in dates:
            key = (municipality, date)
            if key in self.lookup:
                row = self.dataframe.row(self.lookup[key], named=True)
                timespan = self.calc_timespan(date)
                t2m, sun, rain, snow = get_weather_features(municipality, date, self.weather_data_cache, timespan)
                t2m, sun, rain, snow = calculate_agg_weather_features(t2m, sun, rain, snow)
                label_old = torch.tensor(row["damage_grouped"], dtype=torch.float32)

                if date == current_date:
                    label_old = torch.tensor(-1.0, dtype=torch.float32) # adding -1 as a "mask" for the current label to predict

                date_tensor = torch.tensor(date.month, dtype=torch.float32)
                lat_tensor = torch.tensor(latitude, dtype=torch.float32)
                long_tensor = torch.tensor(longitude, dtype=torch.float32)

                features = torch.tensor(
                    np.vstack([t2m, sun, rain, snow, date_tensor, label_old, lat_tensor, long_tensor]),
                    dtype=torch.float32).squeeze(-1)

                mask_sequence.append(torch.tensor(1, dtype=torch.float32)) # valid data
            else:
                features = torch.zeros((8,), dtype=torch.float32)
                mask_sequence.append(torch.tensor(0, dtype=torch.float32))  # padding data
                if date.year > 1973:
                    print(f"added padding for unintended year {date.year}")

            feature_sequence.append(features)

        return torch.stack(feature_sequence)




    def calc_timespan(self, end_date:datetime.date):
        if self.timespan_calendar == 'weekly':
            return 7
        elif self.timespan_calendar == 'monthly':
            _, days = monthrange(end_date.year, end_date.month)
            return days