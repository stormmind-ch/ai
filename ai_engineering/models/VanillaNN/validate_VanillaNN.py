import sys
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error,mean_absolute_error, r2_score
import torch
import numpy as np

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", unit="batch", file=sys.stdout, dynamic_ncols=True):
            labels =  torch.log1p(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds_real = np.expm1(all_preds)
        all_labels_real = np.expm1(all_labels)
        
    avg_loss = running_loss / len(dataloader)
    mse = mean_squared_error(all_labels_real, all_preds_real)
    mae = mean_absolute_error(all_labels_real, all_preds_real)
    r2 = r2_score(all_labels_real, all_preds_real)

    return avg_loss, mse, mae, r2, all_labels_real, all_preds_real

def validate_classification(model, dataloader, criterion, device):
    pass
