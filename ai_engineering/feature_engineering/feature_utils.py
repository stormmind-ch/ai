import polars as pl
import numpy as np
from sklearn.cluster import KMeans


def make_clusters(k: int, mun_coord_df: pl.DataFrame)->pl.DataFrame:
    """
    Args:
        k: Number of clusters which should be generated.
        mun_coord_df: Polars Dataframe which lists each municipality to its latitude and longitude. Must have columns ['Municipality', 'Latitude', 'Longitude']
    returns: A polars dataframe with columns: ['Municipality', 'Latitude', 'Longitude', 'Cluster_ID', 'Cluster_Center_Lat', 'Cluster_Center_Long', 'Cluster_Center_Municipality'
    """
    pass


def group_damages(n:int, date_cluster_damage: pl.DataFrame, damage_weights: list[float]= None)-> pl.DataFrame:
    """
    Args:
        n: Number of days in the dataframe which should be grouped to one timeframe. Starting from the smallest date.
        date_cluster_damage: Dataframe containing a date, cluster (name of the centroid municipality), damage (0: no damage, 1: small damage, 2: medium damage, 3: large damage)
        damage_weights: Weights of the damages on how they should be multiplied. Default: None and the damages are just summed for each timeframe.
    returns: A polars dataframe with the following columns: ['index', 'end_date', 'cluster','damage_grouped]
            'damage_grouped' refers to the total damages which occurred in the given timeframe (end_date-n). If weights were given, they are multiplied by the corresponding weight before summing.
    """
    pass
