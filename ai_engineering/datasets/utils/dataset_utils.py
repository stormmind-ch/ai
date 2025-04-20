import unicodedata
import numpy as np
import os
import orjson as oj
import polars as pl
import datetime

def normalize_text(text):
    return unicodedata.normalize("NFKC", text).replace("−", "-").strip().lower()


def preload_weather_data(weather_data_dir: str, clusters: list[str] = None) -> dict[str, dict[str, np.typing.NDArray]]:
    """
    Preloads all weather data into a dictionary for fast lookup.
    """
    weather_cache = {}
    if clusters is not None:
        files = [f"{cluster}.json" for cluster in clusters]
    else:
        files = [f for f in os.listdir(weather_data_dir) if f.endswith(".json")]
    for file in files:
        if file.endswith(".json"):
            municipality = file.replace(".json", "")
            municipality_normalized = normalize_text(municipality)
            file_path = os.path.join(weather_data_dir, file)
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

def merge_clusters_initial(cluster_df:pl.DataFrame, main_df:pl.DataFrame)->pl.DataFrame:
    merged = main_df.join(
        cluster_df,
        on='Municipality',
        how='left'
    )
    return merged


def date_features_sincos_normalisation(date: datetime.date):
    month = date.month
    month_sin = np.sin(2 * np.pi * month) / 12.0
    month_cos = np.cos(2 * np.pi * month) / 12.0
    return np.array([month_sin, month_cos])

def preload_coordinates(municipality_coordinates_file: str):
    df = pl.read_csv(municipality_coordinates_file)
    min_lat = df.select(pl.min("Latitude")).item()
    max_lat = df.select(pl.max("Latitude")).item()
    min_long = df.select(pl.min("Longitude")).item()
    max_long = df.select(pl.max("Longitude")).item()
    dic = {}
    for row in df.iter_rows():
        lat_norm = (row[1] - min_lat) / (max_lat - min_lat)
        long_norm = (row[2] - min_long) / (max_long - min_long)
        dic[row[0]] = (lat_norm, long_norm)
    df.clear()
    return dic



def load_main_dataset_to_numpy(main_data_path, downsampling_majority_ratio=None):
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


def get_weather_features(municipality: str, date: datetime.date, weather_data_cache, timespan: int):
    """
    Retrieves weather data from preloaded cache.
    """
    first_date = datetime.date(1972, 1, 1)
    delta = date - first_date
    end_date = delta.days + 1
    start_date = end_date - timespan
    municipality_normalized = normalize_text(municipality)

    if municipality_normalized not in weather_data_cache:
        raise Exception(f"{municipality_normalized} does not exist in weather data cache")

    data = weather_data_cache[municipality_normalized]
    temperature_2m_mean = data['temperature_2m_mean'][start_date:end_date]
    sunshine_duration = data['sunshine_duration'][start_date:end_date]
    rain_sum = data['rain_sum'][start_date:end_date]
    snowfall_sum = data['snowfall_sum'][start_date:end_date]

    return np.vstack([temperature_2m_mean, sunshine_duration, rain_sum, snowfall_sum])
