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
from compute_class_weights import compute_class_weights
from downsampler import RandomDownsampler
# ---------- WandB Initialization ----------
wandb.init(project="stormmind.ai")

# Load hyperparameters from WandB
config = wandb.config

# ---------- Initialize Model ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"RUNNING ON: {DEVICE}")
model = Model(input_size=config.timespan*4 +1, hidden_size=config.hidden_size, output_size=config.output_size).to(DEVICE)



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
        #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config.gradient_clipping)
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
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", unit="batch", file=sys.stdout, dynamic_ncols=True):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds

# ---------- Training Function ----------
def train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in trange(epochs, desc="Epochs", file=sys.stdout, dynamic_ncols=True):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, prec, rec, f1, all_labels, all_preds = validate(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                        y_true=all_labels, preds=all_preds,
                        class_names=[i for i in range(config.output_size)])

        })

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        print(f"Loss: {val_loss:.4f} | Acc: {val_accuracy:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")


# ---------- Main Function ----------
def main():
    dataset = StormDamageDataset('../Ressources/main_data_combined.csv',
                                 '../Ressources/weather_data2', config.timespan, '1972-01-01', '2002-01-01', '2012-01-01', config.downsample_ratio)

    train_data =  torch.utils.data.Subset(dataset, dataset.train_indices)
    val_data = torch.utils.data.Subset(dataset, dataset.val_indices)
    test_data = torch.utils.data.Subset(dataset, dataset.test_indices)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, pin_memory=True, num_workers=4)

    # Train the model
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(dataset.damages).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    train(model, train_loader, val_loader, criterion, optimizer, config.epochs, DEVICE)
    test_loss, test_accuracy, prec, rec, f1, all_labels, all_preds = validate(model, test_loader, criterion, DEVICE)
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision" : rec,
        "test_f1" : f1,
        "test_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                        y_true=all_labels, preds=all_preds,
                                                        class_names=[i for i in range(config.output_size)])

    })
    myPath = f"models/hidden_size_{config.hidden_size}_batch_size_{config.batch_size}_learning_rate_{config.learning_rate}.pth"
    torch.save(model.state_dict(), myPath)


if __name__ == "__main__":
    main()
