from importlib.metadata import distribution

import polars as pl
import numpy as np
import numpy.typing as npt

def make_bin_of_classes(dataframe: pl.DataFrame, distribution: list[float]) -> tuple[pl.DataFrame, tuple[float, float]]:
    low_percentage = distribution[0]
    mid_percentage = distribution[1]

    damages = dataframe['damage_grouped']
    non_zero_damages = damages.filter(damages != 0.0)

    low_thresh = np.percentile(non_zero_damages.to_numpy(), low_percentage * 100)
    mid_thresh = np.percentile(non_zero_damages.to_numpy(), (low_percentage + mid_percentage) * 100)

    dataframe = dataframe.with_columns(
        pl.when(pl.col('damage_grouped') == 0.0).then(0)
        .when(pl.col('damage_grouped') <= low_thresh).then(1)
        .when(pl.col('damage_grouped') <= mid_thresh).then(2)
        .otherwise(3)
        .alias('damage_grouped')
    )
    return dataframe, (low_thresh, mid_thresh)
