from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ai_engineering.Faster_Dataset import StormDamageDataset
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    dataset = StormDamageDataset('../Ressources/main_data_combined.csv',
                                  '../Ressources/weather_data2', 7)

    dataloader = DataLoader(
        dataset,
        num_workers=1,  # can stay 1 or even 0 for debugging
        shuffle=False,
    )

    X_list = []
    y_list = []

    for X, y in dataloader:
        X_list.append(X)
        y_list.append(y)

    X_all = torch.cat(X_list, dim=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all.numpy())

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)

    print("\nTop 3 components:")
    top3 = pca.components_[:3]
    print(top3)
