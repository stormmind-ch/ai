from sklearn.metrics import mean_absolute_error
from models.validator import validate
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm, trange
from models.init_model import init_model
import numpy as np
import sys
from models.train_utils import get_optimizer, get_criterion, create_splits, save_model
import wandb
from models.LSTMModel import LSTM
from models.VanillaNNModel import VanillaNN


def _train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    h0, c0 = None, None
    for inputs, labels in tqdm(dataloader, desc="Training", unit="batch", file=sys.stdout, dynamic_ncols=True):
        labels = torch.log1p(labels)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        if isinstance(model, LSTM):
            if h0 is None and c0 is None:
                outputs, (h0, c0) = model(inputs)
            else:
                outputs, (h0, c0) = model(inputs, hc=(h0, c0))
            h0 = h0.detach()
            c0 = c0.detach()
        else:
            outputs = model(inputs)

        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pred_array = outputs.detach().cpu().numpy()
        label_array = labels.cpu().numpy()

        if pred_array.ndim == 0:
            pred_array = np.expand_dims(pred_array, axis=0)

        if label_array.ndim == 0:
            label_array = np.expand_dims(label_array, axis=0)

        all_preds.append(pred_array)
        all_labels.append(label_array)

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

def _train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in trange(epochs, desc="Epochs", file=sys.stdout, dynamic_ncols=True):
        train_loss, train_mae = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss" : train_loss,
            "train_mae": train_mae
        })

def train_and_validate(dataset: Dataset, config,  device):
    splits = create_splits(dataset, config.n_splits, False)

    model_paths = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        model = init_model(config.model, config.input_size, config.hidden_size, config.output_size)
        model.to(device)

        optimizer = get_optimizer(config.optimizer, model.parameters(), config.learning_rate)
        criterion = get_criterion(config.criterion)
        _train(model, train_loader, val_loader, criterion, optimizer, config.epochs, device)
        avg_loss, mse, mae, r2, all_labels_real, all_preds_real = validate(model, val_loader, criterion, device)
        wandb.log({
            f"fold_{fold+1}_avg_loss": avg_loss,
            f"fold_{fold+1}_mse": mse,
            f"fold_{fold+1}_mae": mae,
            f"fold_{fold+1}_r2": r2
        })
        path = save_model(model, fold)
        model_paths.append(path)

    return model_paths
