import numpy as np
import wandb
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Dataset
from torch import Tensor
from torch.optim import Adam, SGD
from typing import Iterable
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import os
import torch

def get_optimizer(optimizer: str, model_params: Iterable[Tensor], lr: float):
    if str.lower(optimizer) == 'adam':
        return Adam(model_params, lr)
    if str.lower(optimizer) == 'sgd':
        return SGD(model_params, lr)

def get_criterion(criterion: str, weights):
    if str.lower(criterion) == 'mseloss':
        return MSELoss()
    elif str.lower(criterion) == 'l1loss':
        return L1Loss()
    elif str.lower(criterion) == 'crossentropyloss':
        return CrossEntropyLoss(weight=weights)
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")


def create_splits(dataset: Dataset, n_splits: int, test_data: bool = False):
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tss.split(dataset))

    if not test_data:
        return splits[:-1]
    else:
        return splits[-1]


def save_model(model):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = f"trained_models/model_fold_{timestamp}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    artifact = wandb.Artifact(name=model_path, type="model")
    artifact.add_file(local_path=model_path, name="model")
    artifact.save()

def calculate_class_weights(dataset: torch.utils.data.Dataset):
    y = []
    for _, label in dataset:
        y.append(label)
    y = np.array(y)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    return torch.tensor(weights)



