from models.trainer import train_and_validate
from datasets.ClusteredStormDamageDataset import ClusteredStormDamageDataset
import torch
import torch.optim
from models.tester import test_on_final_split
import wandb



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
                                        k=config.clusters,
                                        n=7,
                                        damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3})

    model_paths = train_and_validate(dataset, config, device)
    test_on_final_split(dataset, config, device, model_paths)



if __name__ == "__main__":
    main()