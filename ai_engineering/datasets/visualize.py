import numpy as np
from ClusteredStormDamageDataset import ClusteredStormDamageDataset

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

def plot_scatter_matrix(dataset, feature_names=None, max_samples=500):
    # Sample data
    X, y = [], []
    for i in range(min(len(dataset), max_samples)):
        features, label = dataset[i]
        X.append(features.numpy())
        y.append(label.item())
    X = np.array(X)
    y = np.array(y)

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'feat_{i}' for i in range(n_features)]

    # Create grid with extra column for colorbar
    fig = plt.figure(figsize=(13, 13))
    gs = gridspec.GridSpec(n_features, n_features + 1, width_ratios=[1]*n_features + [0.05], wspace=0.1, hspace=0.1)

    axes = np.empty((n_features, n_features), dtype=object)
    scatter = None
    for i, j in itertools.product(range(n_features), repeat=2):
        ax = fig.add_subplot(gs[i, j])
        axes[i, j] = ax
        if i == j:
            ax.hist(X[:, i], bins=20, alpha=0.7)
        else:
            scatter = ax.scatter(X[:, j], X[:, i], c=y, cmap='viridis_r', s=10, alpha=0.7)
        if i == n_features - 1:
            ax.set_xlabel(feature_names[j], fontsize=8)
        else:
            ax.set_xticks([])
        if j == 0:
            ax.set_ylabel(feature_names[i], fontsize=8)
        else:
            ax.set_yticks([])

    # Add colorbar in last column
    cbar_ax = fig.add_subplot(gs[:, -1])
    fig.colorbar(scatter, cax=cbar_ax, label="Damage")

    fig.suptitle("Scatterplot Matrix of Weather Features vs. Damage", fontsize=14)
    plt.show()



# Instantiate your dataset
test_dataset = ClusteredStormDamageDataset('../../Ressources/main_data_1972_2023.csv',
                                           '../../Ressources/weather_data4',
                                           '../../Ressources/municipalities_coordinates_newest.csv',
                                           'mean', 6, 'test', 4, 4,
                                           grouping_calendar='weekly',
                                           damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3})

plot_scatter_matrix(test_dataset, feature_names=['Temperature', 'Sunshine', 'Rain', 'Snow'])

