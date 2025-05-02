import polars as pl
from datetime import timedelta
from datasets.utils.feature_utils import df_column_checker

def group_damages(date_cluster_damage: pl.DataFrame, damage_weights: dict[int, float] = None, n: int = None, grouping_calendar: str = None) -> pl.DataFrame:
    df_column_checker(date_cluster_damage, {'Municipality', 'Date', 'Center_Municipality'})

    if date_cluster_damage['Date'].dtype != pl.Date:
        date_cluster_damage = date_cluster_damage.with_columns(
            pl.col('Date').str.strptime(pl.Date, "%Y-%m-%d")
        )

    if damage_weights:
        date_cluster_damage = date_cluster_damage.with_columns(
            pl.col('extent_of_damage').map_elements(lambda x: damage_weights.get(int(x), 0.0), return_dtype=pl.Float64)
        )
    else:
        date_cluster_damage = date_cluster_damage.with_columns(
            pl.when(pl.col('extent_of_damage') != 0.0)
            .then(1.0)
            .otherwise(0.0)
            .alias('extent_of_damage')
        )

    if grouping_calendar == 'monthly':
        grouped = _group_monthly(date_cluster_damage)
    elif grouping_calendar == 'weekly':
        grouped = _group_weekly(date_cluster_damage)
    elif n is not None:
        grouped = _group_daily(date_cluster_damage, n)
    else:
        raise ValueError("At least n or grouping_calendar must be given to the function")

    grouped = grouped.sort(['end_date', 'Center_Municipality']).with_columns(
        pl.arange(0, grouped.height).alias('index')
    )

    return grouped.select(['index', 'end_date', 'Center_Municipality', 'Cluster_Center_Lat', 'Cluster_Center_Long', 'damage_grouped'])

def _group_weekly(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("Date").map_elements(lambda d: d + timedelta(days=(6 - d.weekday())), return_dtype=pl.Date).alias("end_date")
    ])
    return df.group_by(['end_date', 'Center_Municipality', 'Cluster_Center_Lat', 'Cluster_Center_Long']).agg(
        pl.col('extent_of_damage').sum().alias('damage_grouped')
    )

def _group_monthly(df: pl.DataFrame) -> pl.DataFrame:
    def last_day_of_month(date):
        next_month = (date.replace(day=28) + timedelta(days=4)).replace(day=1)
        return next_month - timedelta(days=1)

    df = df.with_columns([
        pl.col("Date").map_elements(last_day_of_month, return_dtype=pl.Date).alias("end_date")
    ])
    return df.group_by(['end_date', 'Center_Municipality']).agg(
        pl.col('extent_of_damage').sum().alias('damage_grouped')
    )

def _group_daily(df: pl.DataFrame, n: int) -> pl.DataFrame:
    df = df.sort('Date')
    min_date = df.select(pl.col("Date").min()).item()
    df = df.with_columns([
        ((pl.col("Date") - min_date).dt.total_days() // n).alias("window")
    ])
    grouped = df.group_by(['window', 'Center_Municipality']).agg([
        pl.col('extent_of_damage').sum().alias('damage_grouped'),
        pl.col('Date').max().alias('end_date')
    ])
    return grouped.drop('window')
