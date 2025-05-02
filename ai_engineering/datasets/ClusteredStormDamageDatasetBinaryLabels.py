from torch.utils.data.dataset import Dataset
from datasets.ClusteredStormDamageDataset import ClusteredStormDamageDataset

class ClusteredStormDamageDatasetBinaryLabels(Dataset):
    """
    Decorator / Wrapper for ClusteredStormDamageDataset to just return labels in binary [0,1]
    0: No damage happened
    1: Damage happened
    """
    def __init__(self, base_dataset: ClusteredStormDamageDataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        features, label = self.base_dataset[idx]
        label = 1 if label > 0 else 0
        return features, label




