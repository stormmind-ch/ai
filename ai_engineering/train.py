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
wandb.init(project="pytorch_classifier", config={
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "hidden_size": hidden_size,
})
# ----------------------------------------

# ---------- Train one epoch ----------

def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    print("Train one epoch started")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(dataloader):
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
            if i % 100 == 0:
                print('100 inputs done')
        except Exception:
            continue

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy



# Define function to train the model for multiple epochs
def train(model, train_dataset, batch_size=1, learning_rate=0.001, epochs=1, device="cuda"):
    # Initialize WandB
    wandb.init(project="pytorch_classifier", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_size": 128,
    })

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to the selected device (GPU or CPU)
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Log metrics to WandB
        #wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_accuracy})

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")



if __name__ == "__main__":

    train_data = StormDamageDataset('/Users/nilsgamperli/Documents/StormMindData/main_data_combined.csv', '/Users/nilsgamperli/Documents/StormMindData/weather_data', 7)


    # Train the model
    train(model, train_data, batch_size=batch_size, learning_rate=0.001, epochs=1,
          device="cuda" if torch.cuda.is_available() else "cpu")

