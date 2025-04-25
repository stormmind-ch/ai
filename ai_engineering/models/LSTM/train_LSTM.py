import sys
import torch
import wandb
from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from models.LSTM.validate_LSTM import validate

# ---------- Train One Split ----------
def train_one_split(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Training on single Split", unit="batch", file=sys.stdout, dynamic_ncols=True):
        labels = torch.log1p(labels)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, h0, c0 = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    # Flatten lists of arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Avoid overflow, then the max value is 1e13. Should be enought
    all_preds = np.clip(all_preds, a_min=None, a_max=30)
    all_labels = np.clip(all_labels, a_min=None, a_max=30)

    # Convert back to original scale
    all_preds_real = np.expm1(all_preds)
    all_labels_real = np.expm1(all_labels)

    mae = mean_absolute_error(all_labels_real, all_preds_real)

    return running_loss / len(dataloader), mae

def create_splits(dataset : Dataset, n_splits, test_data=False):
    tss = TimeSeriesSplit(n_splits=n_splits)
    if not test_data:
        data = tss.split(dataset)
        data = data[:-1]
        return data
    else:
        data = tss.split(dataset)
        return data[-1]

def train_one_epoch (model, dataset: Dataset, criterion, optimizer, device, n_splits, batch_size):
    train_losses = []
    train_maes = []
    val_losses = []
    val_mses = []
    val_maes = []
    val_r2s = []
    val_labels = []
    val_predictions = []
    for fold, (train_indices, test_indices) in tqdm(enumerate(tss.split(dataset)), desc="Training", unit="Split", file=sys.stdout, dynamic_ncols=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size)
        test_loader = DataLoader(test_dataset, batch_size)

        #Train
        train_loss, train_mae = train_one_split(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_maes.append(train_mae)

        #Validate
        avg_loss, val_mse, val_mae, val_r2, all_labels_real, all_preds_real = validate(model, test_loader, criterion, device)
        val_losses.append(avg_loss)
        val_mses.append(val_mse)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
        val_labels.append(all_labels_real)
        val_predictions.append(all_preds_real)

    return np.array(train_losses), np.array(train_maes), np.array(val_losses), np.array(val_mses), np.array(val_maes), np.array(val_r2s), np.array(val_labels), np.array(val_predictions)


# ---------- Training Function ----------
def train_and_validate(model, dataset, criterion, optimizer, epochs, device, n_splits, batch_size=1):
    for epoch in trange(epochs, desc="Epochs", file=sys.stdout, dynamic_ncols=True):
        train_losses, train_maes, val_losses, val_mses, val_maes, val_r2s, val_labels, val_predictions = train_one_epoch(model, dataset, criterion, optimizer, device, n_splits, batch_size)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_losses.mean(),
            "train_mae": train_maes.mean(),
            "val_avg_loss": val_losses.mean(),
            "val_mse": val_mses.mean(),
            "val_mae" : val_maes.mean(),
            "val_r2" : val_r2s.mean(),
            "val_labeels" : val_labels,
            "val_predictions" : val_predictions
        })
        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_losses.mean():.4f}, Train MAE: {train_maes.mean():.2f} -"
              f"Val Avg Loss: {val_losses.mean():.4f}, Val mse: {val_mses.mean():.2f}, Val mae: {val_maes.mean():.2f}, Val r2: {val_r2s.mean():.2f}")


