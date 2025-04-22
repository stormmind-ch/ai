import polars as pl
import numpy as np
import numpy.typing as npt
from datasets.utils.feature_utils import df_column_checker
from sklearn.cluster import KMeans


def make_clusters(k: int, mun_coord_df: pl.DataFrame)->tuple[pl.DataFrame, list[str]]:
    """
    Args:
        k: Number of clusters which should be generated.
        mun_coord_df: Polars Dataframe which lists each municipality to its latitude and longitude. Must have columns ['Municipality', 'Latitude', 'Longitude']
    Returns: A polars dataframe with columns: ['Municipality', 'Latitude', 'Longitude', 'Cluster_ID', 'Cluster_Center_Lat', 'Cluster_Center_Long', 'Cluster_Center_Municipality'
    """
    df_column_checker(mun_coord_df, {'Municipality', 'Latitude', 'Longitude'})

    coords = mun_coord_df.select(['Latitude', 'Longitude']).to_numpy()
    clusters, centers, center_indices = kmeans_clustering(k, coords)

    mun_coord_df = mun_coord_df.with_columns([
        pl.Series('Cluster_ID', clusters)
    ])

    center_coords = centers[clusters]
    mun_coord_df = mun_coord_df.with_columns([
        pl.Series('Cluster_Center_Lat', center_coords[:, 0]),
        pl.Series('Cluster_Center_Long', center_coords[:, 1])
    ])

    municipality_names = mun_coord_df['Municipality'].to_numpy()
    cluster_center_municipality = [municipality_names[i] for i in center_indices]
    center_name_column = [cluster_center_municipality[c] for c in clusters]

    mun_coord_df = mun_coord_df.with_columns(
        pl.Series('Center_Municipality', center_name_column)
    )

    return mun_coord_df, cluster_center_municipality


def kmeans_clustering(k: int, data: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, list[int]]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_

    center_indices = []
    for center in centers:
        dists = np.linalg.norm(data - center, axis=1)
        center_idx = np.argmin(dists)
        center_indices.append(center_idx)

    return clusters, centers, center_indices