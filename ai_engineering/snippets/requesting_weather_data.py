import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry



# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)


def get_params(latitude, longitude, start_date="1972-01-01", end_date="2023-12-31", daily_parameters=None, timezone="Europe/Berlin"):
    if daily_parameters is None:
        daily_parameters = ["temperature_2m_mean", "sunshine_duration", "rain_sum", "snowfall_sum"]
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_parameters,
        "timezone": timezone
    }

def call(latitude, longitude, daily_parameters=None) -> pd.DataFrame:
    if daily_parameters is None:
        daily_parameters = ["temperature_2m_mean", "sunshine_duration", "rain_sum", "snowfall_sum"]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = get_params(latitude, longitude, daily_parameters=daily_parameters)
    response = openmeteo.weather_api(url, params=params)[0]
    daily = response.Daily()

    data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}

    for i, parameter in enumerate(daily_parameters):
        data[parameter] = daily.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data)


