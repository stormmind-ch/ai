from train import train, validate_regression
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

    train_dataset = ClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                          '../Ressources/municipalities_coordinates_newest.csv',
                                        'mean',6, 'train', 4,4,
                                        grouping_calendar='weekly')
    val_dataset = ClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                          '../Ressources/municipalities_coordinates_newest.csv',
                                        'mean',6, 'val', 4,4,
                                        grouping_calendar='weekly')

    test_dataset = ClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                          '../Ressources/municipalities_coordinates_newest.csv',
                                        'mean',6, 'test', 4,4,
                                        grouping_calendar='weekly')

    model = init_model('VanillaNN', 4, 128, 1)

    train_loader, val_loader, test_loader = DataLoader(train_dataset), DataLoader(val_dataset), DataLoader(test_dataset)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=0.001)
    train(model, train_loader, val_loader, criterion, optimizer,10, device)
    avg_loss_test, rmse_test, mae_test, r2_test, all_labels_real_test, all_preds_real_test = validate_regression(model, test_loader, criterion, device)

    wandb.log({
        "avg_loss_test": avg_loss_test,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test
    })

if __name__ == "__main__":
    main()