from datasets.ClusteredStormDamageDataset import ClusteredStormDamageDataset

class ClusteredStormDamageDatasetBinaryLabels(ClusteredStormDamageDataset):
    """
    Decorator / Wrapper for ClusteredStormDamageDataset to just return labels in binary [0,1]
    0: No damage happened
    1: Damage happened
    """
    def __init__(self, main_data_path : str, weather_data_dir: str, municipality_coordinates_path:str, n_clusters: int, n_sequences:int,
                 split: str = None,val_years : int = 2, test_years: int = 2,
                 damage_distribution:list[float] = [0.90047344, 0.06673681, 0.03278976],
                 damage_weights:dict[int:float]=None, grouping_calendar: str = 'weekly', grouping_daily: int= None):
        super().__init__(main_data_path, weather_data_dir, municipality_coordinates_path,
                         n_clusters, n_sequences, split,
                         val_years, test_years, damage_distribution,
                         damage_weights, grouping_calendar, grouping_daily)


    def __getitem__(self, idx):
        features, label = super().__getitem__(idx)
        label = 1 if label > 0 else 0
        return features, label




