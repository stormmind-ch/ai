from models.LSTM.train_LSTM import train_and_validate
from datasets.ClusteredStormDamageDataset import ClusteredStormDamageDataset
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.optim
from torch.utils.data.dataloader import DataLoader
import wandb
from models.init_model import init_model


def init_wandb():
    wandb.init(project="stormmind.ai")
    return wandb.config

def init_device():
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RUNNING ON: {device}")
    return device


def main():
    config = init_wandb()
    device = init_device()

    dataset = ClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                        '../Ressources/municipalities_coordinates_newest.csv',
                                        k=12,
                                        grouping_calendar='weekly',
                                        damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3})


    model = init_model('LSTM', 4, 64, 1)
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=0.001)
    train_and_validate(model, dataset, criterion, optimizer, 1, device, 10)


if __name__ == "__main__":
    main()