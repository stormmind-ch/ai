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

    h0, c0 = None, None
    for inputs, labels in tqdm(dataloader, desc="Training", unit="batch", file=sys.stdout, dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if isinstance(model, LSTM):
            outputs, _ = model(inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def _train(model, train_loader, criterion, optimizer, epochs, device):
    for epoch in trange(epochs, desc="Epochs", file=sys.stdout, dynamic_ncols=True):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss" : train_loss
        })

def train_and_validate(dataset: Dataset, config,  device):
    splits = create_splits(dataset, config.n_splits, False)

    model_paths = []
    fold_table = wandb.Table(columns=["fold", "avg_loss", "accuracy", "precision", "specificity", "f1"])
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        model = init_model(config.model, config.input_size, config.hidden_size, config.output_size)
        model.to(device)

        optimizer = get_optimizer(config.optimizer, model.parameters(), config.learning_rate)
        criterion = get_criterion(config.criterion)
        _train(model, train_loader, criterion, optimizer, config.epochs, device)
        avg_loss, accuracy, precision, specificity, f1, labels, predictions = validate(model, val_loader, criterion, device)

        fold_table.add_data(fold + 1, avg_loss, accuracy, precision, specificity, f1)
        path = save_model(model, fold)
        model_paths.append(path)

    wandb.log({"fold_metrics": fold_table})

    accuracy_col = fold_table.get_column("accuracy")
    mean_accuracy = sum(accuracy_col) / len(accuracy_col)
    wandb.log({"mean_accuracy": mean_accuracy})

    precision_col = fold_table.get_column("precision")
    mean_precision = sum(precision_col) / len(precision_col)
    wandb.log({"mean_precision": mean_precision})

    specificity_col = fold_table.get_column("specificity")
    mean_specificity = sum(specificity_col) / len(specificity_col)
    wandb.log({"mean_specificity": mean_specificity})

    f1_col = fold_table.get_column("f1")
    mean_f1 = sum(f1_col) / len(f1_col)
    wandb.log({"mean_f1": mean_f1})

    return model_paths
