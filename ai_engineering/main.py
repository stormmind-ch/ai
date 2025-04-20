from train import train, validate
from datasets.ClusteredStormDamageDataset import ClusteredStormDamageDataset
import torch
from torch.optim import Adam
import torch.nn as nn
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




    dataset = ClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv', '../Ressources/weather_data4',
                                          '../Ressources/municipalities_coordinates_newest.csv', 'mean', 6,grouping_calendar='monthly')

    pattern = dataset[0]
    #TODO init model and split dataset to train, val, test


if __name__ == "__main__":
    main()