from datetime import date, timedelta
import torch
from datasets.NormalizedClusteredStormDamageDataset import NormalizedClusteredStormDamageDataset

class ClusteredStormDamageDatasetIncludesPreviousYear(NormalizedClusteredStormDamageDataset):
    """
    Decorator / Wrapper for ClusteredStormDamageDataset to also return values from one year before.
    """
    def __init__(self, main_data_path: str, weather_data_dir: str, municipality_coordinates_path: str, n_clusters: int,
                 n_sequences: int,
                 split: str = None, test_years: int = 2,
                 damage_distribution: list[float] = [0.90047344, 0.06673681, 0.03278976],
                 damage_weights: dict[int:float] = None, grouping_calendar: str = 'weekly',
                 mean=None, std=None):
        super().__init__(main_data_path, weather_data_dir, municipality_coordinates_path, n_clusters, n_sequences,
                         split, test_years, damage_distribution, damage_weights, grouping_calendar, mean, std)

    def __getitem__(self, idx):
        """
        Returns the data of size:
            features: (sequence_length, 8) in the following order:
                temperature_2m_mean, sun_duration_mean, rain_sun, snow_sum, label (only of last sequence), month, latitude, longitude
            labels: (1)
            features_last_year: (sequence_length, 8):
                same order as features
        """
        if idx >= len(self):
            raise IndexError

        row = self.dataframe.row(idx, named=True)
        end_date = row['end_date']
        municipality = row['Center_Municipality']
        dates = get_past_week_dates_year(end_date, self.n_sequences)
        damage = row['damage_grouped']
        latitude = row['Cluster_Center_Lat']
        longitude = row['Cluster_Center_Long']

        # dates are from the current back to the one furthest in the past
        all_features = self.get_feature_sequence(dates, end_date, municipality, latitude, longitude)
        current_features = all_features[:self.n_sequences]
        last_year_features = all_features[self.n_sequences:]

        damage = 1 if damage > 0 else 0
        label = torch.tensor(damage, dtype=torch.long)
        return current_features, label, last_year_features





def get_past_week_dates_year(base_date: date, timespan: int):
    # Step 1: Get the current year's Sundays (backward)
    current_dates = [base_date - timedelta(weeks=i) for i in range(timespan + 1)]

    # Step 2: Get previous year's corresponding Sundays + 1 week
    previous_year_dates = []
    for d in current_dates:
        try:
            one_year_earlier = d.replace(year=d.year - 1)
        except ValueError:
            # handle Feb 29 → Feb 28
            one_year_earlier = d - timedelta(days=365)
        shifted = one_year_earlier + timedelta(weeks=1)
        # Ensure it’s still a Sunday
        if shifted.weekday() != 6:  # Sunday = 6
            # Adjust to next Sunday
            shifted += timedelta(days=(6 - shifted.weekday()))
        previous_year_dates.append(shifted)

    return current_dates + previous_year_dates