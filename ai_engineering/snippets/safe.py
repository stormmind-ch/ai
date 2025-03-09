import numpy as np
import pandas as pd


def safe_weather_data(data: pd.DataFrame, name: str, location: str) -> None:
    data.to_csv(f'{location}/{name}.csv', index=False)