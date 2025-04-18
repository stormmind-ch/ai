import numpy as np
import polars as pl
import torch
from torch.utils.data.dataset import  Dataset
from dataset_utils import  preload_weather_data


class ClusteredStormDamageDataset(Dataset):
    def __init__(self, main_dataframe : pl.DataFrame, weather_data_dir: str, clusters:list[str], timespan: int, agg_method: str):
        """
        Args:
            main_dataframe: Polars dataframe containing the following columns ['index', 'end_date', 'cluster','damage_grouped]. Returned by feature_utils.group_damages
            weather_data_dir: Directory containing all weather information for each municipality in clusters as .json files.
            clusters: List of all municipalities which are the centroids of a cluster
            timespan: Amount of days which should be considered for one entry
            agg_method: How the weather data over the timespan should be aggregated to one value. E.g. ['mean', 'median']
        """
        self.main_dataframe = main_dataframe
        self.weather_data_cache = preload_weather_data(weather_data_dir,clusters)
        pass

