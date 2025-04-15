import pandas as pd
import unicodedata
import torch
from ai_engineering.StormDamageDataset import StormDamageDataset
import numpy as np

def normalize_text(text):
    return unicodedata.normalize("NFKC", text).replace("−", "-").strip().lower()

MAIN_DATA_PATH = "main_data_combined_test.csv"
WEATHER_DATA_DIR  = "../../Ressources/weather_data4"
TIMESPAN = 3


def test_preload_weather_data():
    """
    Tests if weather data is correctly preloaded into the cache.
    """
    dataset = StormDamageDataset(MAIN_DATA_PATH, WEATHER_DATA_DIR, TIMESPAN)
    unique_municipalities = '/ai_engineering/test/unique_mun.csv'
    municipalities = pd.read_csv(unique_municipalities)['Municipality'].tolist()
    for i, mun in enumerate(municipalities, start=1):
        if i == 0:
            continue
        mun_norm = normalize_text(mun)
        assert mun_norm in dataset.weather_cache.keys(), f"Missing {mun} in cache"
        assert len(dataset.weather_cache[mun_norm]["daily"]["temperature_2m_mean"]) == 18994, f"Not correct length of temperature list: {mun_norm}"
        assert len(dataset.weather_cache[mun_norm]["daily"]["sunshine_duration"]) == 18994, f"Not correct length of sunshine list : {mun_norm}"
        assert len(dataset.weather_cache[mun_norm]["daily"]["rain_sum"]) == 18994, f"Not correct length of rain list : {mun_norm}"
        assert len(dataset.weather_cache[mun_norm]["daily"]["snowfall_sum"]) == 18994, f"Not correct length of snowfall list : {mun_norm}"

def test_train():
    """
    Tests if each row of the main dataset can be retrieved. Note that in case of an Error, the Dataset throws an Exception.
    """
    dataset = StormDamageDataset(MAIN_DATA_PATH, WEATHER_DATA_DIR, TIMESPAN, '1972-01-01', '2002-01-01', '2012-01-01')
    max_idx = 52_231_987
    expected_feature_size = 4 * TIMESPAN

    for i in range(49978607,max_idx):
        features, label = dataset.__getitem__(i)
        assert features.size() == torch.Size([expected_feature_size]), f"Feature error on index {i}"

def test_dataset_length():
    length_dataset = len(pd.read_csv(MAIN_DATA_PATH))
    dataset = StormDamageDataset(MAIN_DATA_PATH, WEATHER_DATA_DIR, TIMESPAN, '1972-01-01', '2002-01-01', '2012-01-01')
    assert len(dataset.damages) == length_dataset
    assert len(dataset.municipalities) == length_dataset
    assert len(dataset.damages) == length_dataset

def test_weather_data_accuracy():
    dataset = StormDamageDataset(MAIN_DATA_PATH, WEATHER_DATA_DIR, TIMESPAN, '1972-01-01', '2002-01-01', '2012-01-01')
    data, _ = dataset.__getitem__(0)
    data = data[:TIMESPAN * 4]
    goal = [8.5,9.3,9.8, 25730.62,22622.33,23916.93, 0.00,0.00,0.30, 0.00,0.00,0.00]
    goal = zscore(np.array(goal), dataset.mean, dataset.std)
    assert np.allclose(data, goal, rtol=0.1)


def zscore(x, mean, std):
    return (x - mean) / std