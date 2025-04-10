import torch
import numpy as np
from sklearn.utils.class_weight import  compute_class_weight


def compute_class_weights(labels:torch.Tensor):
    return torch.from_numpy(compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels))
