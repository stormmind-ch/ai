import sys
import torch
import wandb
from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error
import numpy as np

from validate import validate_regression

# ---------- Train One Epoch ----------
def train_one_epoch_regression(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Training", unit="batch", file=sys.stdout, dynamic_ncols=True):
        labels = torch.log1p(labels)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).view(-1)  # flatten safely
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





# ---------- Training Function ----------
def train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in trange(epochs, desc="Epochs", file=sys.stdout, dynamic_ncols=True):
        train_loss, train_mae = train_one_epoch_regression(model, train_loader, criterion, optimizer, device)
        avg_loss, val_mse, val_mae, val_r2, all_labels_real, all_preds_real = validate_regression(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_avg_loss": avg_loss,
            "val_mse": val_mse,
            "val_mae" : val_mae,
            "val_r2" : val_r2
        })


        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_mae:.2f} -"
              f"Val Avg Loss: {avg_loss:.4f}, Val mse: {val_mse:.2f}, Val mae: {val_mae:.2f}, Val r2: {val_r2:.2f}")


