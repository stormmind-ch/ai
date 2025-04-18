import unicodedata
import numpy as np
import os
import orjson as oj

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