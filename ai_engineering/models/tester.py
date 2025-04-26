from models.init_model import init_model
from models.validator import validate
from torch.utils.data import Dataset
from models.train_utils import create_splits, get_criterion
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
import torch
import wandb

def test_on_final_split(dataset: Dataset, config, device, model_paths):
    test_split = create_splits(dataset, config.n_splits, test_data=True)
    train_idx, test_idx = test_split

    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    for fold, model_path in enumerate(model_paths):
        model = init_model(config.model, config.input_size, config.hidden_size, config.output_size)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        criterion = get_criterion(config.criterion)
        avg_loss, mse, mae, r2, _, _ = validate(model, test_loader, criterion, device)

        wandb.log({
            f"test_fold_{fold+1}_avg_loss": avg_loss,
            f"test_fold_{fold+1}_mse": mse,
            f"test_fold_{fold+1}_mae": mae,
            f"test_fold_{fold+1}_r2": r2
        })

        print(f"Fold {fold+1} tested on final split: MAE = {mae:.4f}, R2 = {r2:.4f}")
