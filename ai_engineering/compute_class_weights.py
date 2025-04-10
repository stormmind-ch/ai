import torch
import numpy as np
from sklearn.utils.class_weight import  compute_class_weight


def compute_class_weights(labels:torch.Tensor):
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(weights, dtype=torch.float32)
