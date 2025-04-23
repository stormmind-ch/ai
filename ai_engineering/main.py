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
                                        config.agg_method,config.clusters, 'train', 4,4,
                                        grouping_calendar='weekly',
                                        damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3})
    val_dataset = ClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                          '../Ressources/municipalities_coordinates_newest.csv',
                                        config.agg_method, config.clusters, 'val', 4,4,
                                        grouping_calendar='weekly',
                                        damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3})

    test_dataset = ClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                          '../Ressources/municipalities_coordinates_newest.csv',
                                        config.agg_method,config.clusters, 'test', 4,4,
                                        grouping_calendar='weekly',
                                        damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3})


    model = init_model('VanillaNN', 4, config.hidden_size, 1)
    model.to(device)
    train_loader, val_loader, test_loader = DataLoader(train_dataset), DataLoader(val_dataset), DataLoader(test_dataset)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    train(model, train_loader, val_loader, criterion, optimizer,config.epochs, device)
    avg_loss_test, rmse_test, mae_test, r2_test, all_labels_real_test, all_preds_real_test = validate_regression(model, test_loader, criterion, device)

    wandb.log({
        "avg_loss_test": avg_loss_test,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test,
        "test_true_labels": all_labels_real_test,
        "test_pred_labels": all_preds_real_test
    })

if __name__ == "__main__":
    main()