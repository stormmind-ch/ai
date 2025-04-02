import sys
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm, trange
from model_1 import Model
from Faster_Dataset import StormDamageDataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------- WandB Initialization ----------
wandb.init(project="storm_damage_nn", config={
    "input_size": 28,
    "hidden_size": 128,
    "output_size": 16,
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 10
})

# Load hyperparameters from WandB
config = wandb.config

# ---------- Initialize Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Model(input_size=config.input_size, hidden_size=config.hidden_size, output_size=config.output_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


# ---------- Train One Epoch ----------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training", unit="batch", file=sys.stdout, dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100 * correct / total


# ---------- Validate One Epoch ----------
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", unit="batch", file=sys.stdout, dynamic_ncols=True):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100 * correct / total

def validate_with_metrics(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", unit="batch", file=sys.stdout, dynamic_ncols=True):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1

# ---------- Training Function ----------
def train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in trange(epochs, desc="Epochs", file=sys.stdout, dynamic_ncols=True):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        val_loss, acc, prec, rec, f1 = validate_with_metrics(model, val_loader, criterion, device)
        print(f"Loss: {val_loss:.4f} | Acc: {acc:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")


# ---------- Main Function ----------
def main():
    dataset = StormDamageDataset('../Ressources/main_data_combined.csv',
                                 '../Ressources/weather_data2', 7)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)

    train(model, train_loader, val_loader, criterion, optimizer, config.epochs, device)


if __name__ == "__main__":
    main()
