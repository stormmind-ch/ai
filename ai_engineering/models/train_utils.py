from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Dataset
from torch import Tensor
from torch.optim import Adam, SGD
from typing import Iterable
from torch.nn import L1Loss, MSELoss
from datetime import datetime
import os
import torch

def get_optimizer(optimizer: str, model_params: Iterable[Tensor], lr: float):
    if str.lower(optimizer) == 'adam':
        return Adam(model_params, lr)
    if str.lower(optimizer) == 'sgd':
        return SGD(model_params, lr)

def get_criterion(criterion: str):
    if str.lower(criterion) == 'mseloss':
        return MSELoss()
    elif str.lower(criterion) == 'l1loss':
        return L1Loss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")


def create_splits(dataset: Dataset, n_splits: int, test_data: bool = False):
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tss.split(dataset))

    if not test_data:
        return splits[:-1]
    else:
        return splits[-1]

# Create a timestamp
def save_model(model, fold):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = f"trained_models/model_fold_{fold + 1}_{timestamp}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    return model_path