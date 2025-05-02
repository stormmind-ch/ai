from sklearn.metrics import mean_absolute_error
from models.validator import validate
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm, trange
from models.init_model import init_model, get_seq2seq
import numpy as np
import sys
from models.train_utils import get_optimizer, get_criterion, create_splits, save_model
import wandb
from models.LSTMModel import LSTM
from models.VanillaNNModel import VanillaNN


def _train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels, previousyear in tqdm(dataloader, desc="Training", unit="batch", file=sys.stdout, dynamic_ncols=True):
        inputs, labels, previousyear = inputs.to(device), labels.to(device), previousyear.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, previousyear)

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

def train_and_validate(train_dataset: Dataset, test_dataset: Dataset, config, device):

    model_paths = []


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = get_seq2seq(config.hidden_size, config.num_layers, config.p)
    model.to(device)

    optimizer = get_optimizer(config.optimizer, model.parameters(), config.learning_rate)
    criterion = get_criterion(config.criterion)
    _train(model, train_loader, criterion, optimizer, config.epochs, device)
    avg_loss, accuracy, precision, specificity, f1, labels, predictions = validate(model, val_loader, criterion, device)


    path = save_model(model, 1)
    model_paths.append(path)
    wandb.log({
        "avg_loss" : avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "specificity" : specificity,
        "f1": f1
    })

    return model_paths
