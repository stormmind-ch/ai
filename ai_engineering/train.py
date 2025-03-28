import time
import sys
from tqdm import tqdm
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from model_1 import Model
from Faster_Dataset import StormDamageDataset

# ---------- Hyperparameters ----------
input_size = 28
hidden_size = 128
output_size = 16  # [0-15]
batch_size = 64
learning_rate = 0.001
epochs = 10
# ----------------------------------------

# ---------- Initialize Model ----------
model = Model(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
# ----------------------------------------


# ---------- Loss function and optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# ----------------------------------------

# ---------- Setup WandB for hyperparameter tuning ----------

# ----------------------------------------

# ---------- Train one epoch ----------

def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    #print("Train one epoch started")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    time_one = time.time()
    time_two = time.time()


    for i, batch in enumerate(tqdm(dataloader, desc="Training", unit="batch", file=sys.stdout, dynamic_ncols=True, leave=True)):
        if batch is None:
            continue
        (inputs, labels) = batch
        try:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #if i % 100 == 0:
                #time_two = time_one
                #time_one = time.time()
                ##duration = (time_one - time_two)
                #print(f"100 inputs done in {duration:.2f} seconds")
        except Exception:
            continue

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy



# Define function to train the model for multiple epochs
def train(model, train_dataset, batch_size=1, learning_rate=0.001, epochs=1, device="cuda"):
    # Initialize WandB


    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=safe_collate)
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to the selected device (GPU or CPU)
    model.to(device)

    # Training loop
    for epoch in trange(epochs, desc="Epochs", file=sys.stdout, dynamic_ncols=True, leave=True):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Log metrics to WandB
        #wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_accuracy})

        #print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

def safe_collate(batch):
    # Filter out any None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # or raise an error if this is unexpected
    return torch.utils.data._utils.collate.default_collate(batch)

if __name__ == "__main__":

    train_data = StormDamageDataset('../StormMindData/main_data_combined.csv', '../StormMindData/weather_data1', 7)


    # Train the model
    train(model, train_data, batch_size=batch_size, learning_rate=0.001, epochs=1,
          device="cuda" if torch.cuda.is_available() else "cpu")

