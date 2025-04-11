from torch.utils.data import Subset
from Faster_Dataset import StormDamageDataset
import random

class RandomDownsampler():
    def __init__(self, dataset: StormDamageDataset):
        self.dataset = dataset

    def downsample_majority_class(self, ratio: float):
        train_indices = self.dataset.train_indices
        train_labels = [self.dataset[i][1] for i in train_indices]


        majority_indices = [idx for idx, label in zip(train_indices, train_labels) if label == 0.0]
        minority_indices = [idx for idx, label in zip(train_indices, train_labels) if label != 0.0]

        downsampled_majority_indices = random.sample(majority_indices, int(ratio * len(majority_indices)))

        final_train_indices = downsampled_majority_indices + minority_indices
        random.shuffle(final_train_indices)

        train_data = Subset(self.dataset, final_train_indices)
        return train_data
