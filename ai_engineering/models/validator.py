import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import numpy as np
import sys


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", unit="batch", file=sys.stdout, dynamic_ncols=True):
            labels = torch.log1p(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, (h0, c0) = model(inputs)

            # Squeeze outputs/labels if needed
            if outputs.dim() == 2 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            if labels.dim() == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        all_preds_real = np.expm1(all_preds)
        all_labels_real = np.expm1(all_labels)

    avg_loss = running_loss / len(dataloader)
    mse = mean_squared_error(all_labels_real, all_preds_real)
    mae = mean_absolute_error(all_labels_real, all_preds_real)
    r2 = r2_score(all_labels_real, all_preds_real)

    return avg_loss, mse, mae, r2, all_labels_real, all_preds_real
