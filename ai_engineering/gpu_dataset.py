import torch

class StormDamageDatasetGPU(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        self.features = torch.stack([x[0].clone().detach().to(torch.float32) for x in dataset]).to(device, non_blocking=True)
        self.labels = torch.stack([x[1].clone().detach().to(torch.float32) for x in dataset]).to(device, non_blocking=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
