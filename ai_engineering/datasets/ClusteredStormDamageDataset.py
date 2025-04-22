import datetime
import numpy as np
import polars as pl
import torch
from torch.utils.data.dataset import  Dataset
from ai_engineering.datasets.utils.dataset_utils import  preload_weather_data, merge_clusters_initial, normalize_text, get_weather_features
from ai_engineering.datasets.utils.grouping_utils import group_damages
from ai_engineering.datasets.utils.clustering_utils import make_clusters
from calendar import monthrange


def _load_main_data(main_data_path: str, municipality_coordinates_path,
                    k, damage_weights, n, grouping_calendar):

    mun_dates_damage = pl.read_csv(main_data_path)
    mun_coordinates = pl.read_csv(municipality_coordinates_path)
    clusters_df, clusters = make_clusters(k, mun_coordinates)
    merged = merge_clusters_initial(clusters_df, mun_dates_damage)
    final = group_damages(merged, damage_weights,n, grouping_calendar)
    return final, clusters


def split_dataframe_by_time(df: pl.DataFrame, val_years: int, test_years: int) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = df.sort("end_date")
    last_date = df.select(pl.col("end_date").max()).item()

    test_start = last_date.replace(year=last_date.year - test_years)
    val_start = test_start.replace(year=test_start.year - val_years)

    train = df.filter(pl.col("end_date") < val_start)
    val = df.filter((pl.col("end_date") >= val_start) & (pl.col("end_date") < test_start))
    test = df.filter(pl.col("end_date") >= test_start)

    return train, val, test


class ClusteredStormDamageDataset(Dataset):
    def __init__(self, main_data_path : str, weather_data_dir: str, municipality_coordinates_path:str,
                 agg_method: str, k: int, split: str, val_years:int = 2, test_years: int = 2,
                 damage_weights:dict[int:float]=None, n:int = None, grouping_calendar: str = None):
        """
        Args:
            main_data_path: Path to main data file containing a entry for each day and municipality and the damage
            weather_data_dir: Directory containing all weather information for each municipality in clusters as .json files.
            timespan: Amount of days which should be considered for one entry
            agg_method: How the weather data over the timespan should be aggregated to one value. E.g. ['mean', 'median']
            k: Number of clusters which should be created
        """
        self.dataframe, clusters = _load_main_data(main_data_path, municipality_coordinates_path, k, damage_weights, n, grouping_calendar)
        self.timespan_int = n if n else None
        self.timespan_calendar = grouping_calendar if grouping_calendar else None
        self.agg_method = agg_method
        self.weather_data_cache = preload_weather_data(weather_data_dir, clusters)

        train_df, val_df, test_df = split_dataframe_by_time(self.dataframe, val_years, test_years)
        if split == 'train':
            self.dataframe = train_df
        elif split == 'val':
            self.dataframe = val_df
        elif split == 'test':
            self.dataframe = test_df
        else:
            raise ValueError(f"Unknown split type: {split}")

    def __len__(self):
        return self.dataframe.height

    def __getitem__(self, idx):
        row = self.dataframe.row(idx, named=True)
        end_date = row['end_date']
        municipality = row['Center_Municipality']
        damage_grouped = row['damage_grouped']

        timespan = self.calc_timespan(end_date)
        weather_features = get_weather_features(municipality, end_date, self.weather_data_cache, timespan)

        if self.agg_method == 'median':
            agg_weather_features = np.median(weather_features, axis=1)
        elif self.agg_method == 'sum':
            agg_weather_features = np.sum(weather_features, axis=1)
        else:
            agg_weather_features = np.mean(weather_features, axis=1)

        features = torch.tensor(agg_weather_features)
        label = torch.tensor(damage_grouped)

        return features, label


    def calc_timespan(self, end_date:datetime.date):
        if self.timespan_int:
            return self.timespan_int
        elif self.timespan_calendar == 'weekly':
            return 7
        elif self.timespan_calendar == 'monthly':
            _, days = monthrange(end_date.year, end_date.month)
            return days