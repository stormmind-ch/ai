from torch.utils.data import Subset
from Faster_Dataset import StormDamageDataset
import random

class RandomDownsampler():
    def __init__(self, dataset: StormDamageDataset):
        self.dataset = dataset

    def downsample_majority_class(self, ratio: float):
        train_indices = self.dataset.train_indices
        labels = self.dataset.damages

        majority_indices = []
        minority_indices = []
        for i in train_indices:
            if labels[i] == 0.0:
                majority_indices.append(i)
            else:
                minority_indices.append(i)

        sample_size = int(ratio * len(majority_indices))
        downsampled_majority_indices = random.sample(majority_indices, sample_size)

        final_train_indices = downsampled_majority_indices + minority_indices
        random.shuffle(final_train_indices)

        train_data = Subset(self.dataset, final_train_indices)
        return train_data
