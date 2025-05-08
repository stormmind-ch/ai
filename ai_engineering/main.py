from models.trainer import train_and_validate
from datasets.ClusteredStormDamageDatasetIncludesPreviousYear import ClusteredStormDamageDatasetIncludesPreviousYear
import torch
import torch.optim
import wandb
from datasets.NormalizedClusteredStormDamageDataset import NormalizedClusteredStormDamageDataset


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

    test_dataset = NormalizedClusteredStormDamageDataset('../Ressources/main_data_1972_2023.csv',
                                        '../Ressources/weather_data4',
                                        '../Ressources/municipalities_coordinates_newest.csv',
                                                  n_clusters=config.clusters,
                                                  n_sequences=config.n_sequences, test_years=10,
                                                  grouping_calendar='weekly', split='test',
                                                  damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3},
                                                         mean=mean, std=std)


    model_paths = train_and_validate(train_dataset, test_dataset, config, device)





if __name__ == "__main__":
    main()