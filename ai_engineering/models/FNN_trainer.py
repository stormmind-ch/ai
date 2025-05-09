from models.FNN_validator import validate
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm, trange
from models.init_model import get_seq2seq, get_fnn
from torch.optim.adam import Adam
import sys
from models.train_utils import save_model, calculate_class_weights
from torch.nn import CrossEntropyLoss
import wandb


def _train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=0.5, total_iters=30)
    for inputs, labels in tqdm(dataloader, desc="Training", unit="batch", file=sys.stdout, dynamic_ncols=True):
        inputs = inputs[:, :, :3]
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()
    return running_loss / len(dataloader)

def _train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in trange(epochs, desc="Epochs", file=sys.stdout, dynamic_ncols=True):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss" : train_loss
        })
        val_loss, accuracy, precision, specificity, val_f1, _, _ = validate(model,val_loader, criterion,device)
        wandb.log({
            "val_loss" : val_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_specificy": specificity,
            "val_f1": val_f1
        })

def train(train_dataset: Dataset, test_dataset: Dataset, config, device):
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = get_fnn()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    class_weights = calculate_class_weights(train_dataset).to(device).float()
    criterion = CrossEntropyLoss(weight=class_weights)
    _train(model, train_loader, val_loader, criterion, optimizer, config.epochs, device)
    return model, criterion
