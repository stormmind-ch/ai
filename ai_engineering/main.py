from training.train import train
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim
import wandb
from datasets.NormalizedClusteredStormDamageDataset import NormalizedClusteredStormDamageDataset
from validation.validation import validate

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

    train_dataset = NormalizedClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                        '../Ressources/municipalities_coordinates_newest.csv',
                                                  n_clusters=config.clusters,
                                                  n_sequences=config.n_sequences, test_years=10,
                                                  grouping_calendar='weekly', split='train',
                                                  damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3})
    mean, std = train_dataset.mean, train_dataset.std

    wandb.log({
        "mean": mean,
        "std": std
    })

    val_dataset = NormalizedClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                        '../Ressources/municipalities_coordinates_newest.csv',
                                                  n_clusters=config.clusters,
                                                  n_sequences=config.n_sequences, test_years=10,
                                                  grouping_calendar='weekly', split='val',
                                                  damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3},
                                                         mean=mean, std=std)
    test_dataset = NormalizedClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                        '../Ressources/municipalities_coordinates_newest.csv',
                                                  n_clusters=config.clusters,
                                                  n_sequences=config.n_sequences, test_years=10,
                                                  grouping_calendar='weekly', split='test',
                                                  damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3},
                                                         mean=mean, std=std)


    model, criterion = train(train_dataset, val_dataset, config, device)
    test_loader = DataLoader(test_dataset )
    avg_loss, accuracy, precision, specificity, f1, all_labels, all_preds = validate(model, test_loader, criterion, config.threshold, device, testing=True)
    wandb.log({
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_specificity": specificity,
        "test_f1": f1,
    })


if __name__ == "__main__":
    main()