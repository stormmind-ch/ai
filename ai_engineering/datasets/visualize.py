import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import itertools
from ClusteredStormDamageDataset import ClusteredStormDamageDataset

def plot_scatter_matrix(dataset, feature_names=None, num_clusters = None):
    # Sample data
    X, y = [], []
    for i in range(len(dataset)):
        features, label = dataset[i]
        X.append(features.numpy())
        y.append(label.item())
    X = np.array(X)
    y = np.array(y)

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'feat_{i}' for i in range(n_features)]

    # Setup figure with colorbar space
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(n_features, n_features + 1, width_ratios=[1]*n_features + [0.05], wspace=0.1, hspace=0.1)

    axes = np.empty((n_features, n_features), dtype=object)
    scatter = None

    mask_damage = y >= 1
    mask_nodamage = y < 1

    for i, j in itertools.product(range(n_features), repeat=2):
        ax = fig.add_subplot(gs[i, j])
        axes[i, j] = ax

        if i == j:
            ax.hist(X[:, i], bins=20, alpha=0.7, color='gray')
        else:
            # Plot no-damage points
            ax.scatter(
                X[mask_nodamage][:, j],
                X[mask_nodamage][:, i],
                color='lightgray',
                s=5,
                alpha=0.3,
                label='No Damage'
            )

            # Plot damage points
            scatter = ax.scatter(
                X[mask_damage][:, j],
                X[mask_damage][:, i],
                c=y[mask_damage],
                cmap='viridis_r',
                s=20,
                alpha=0.9,
                edgecolors='black',
                linewidth=0.3,
                label='Damage ≥ 1'
            )

        # Axis labels
        if i == n_features - 1:
            ax.set_xlabel(feature_names[j], fontsize=8)
        else:
            ax.set_xticks([])
        if j == 0:
            ax.set_ylabel(feature_names[i], fontsize=8)
        else:
            ax.set_yticks([])

        # Add legend only once
        if i == n_features - 1 and j == n_features - 1:
            ax.legend(fontsize=6, loc='upper right')

    # Add shared colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    fig.colorbar(scatter, cax=cbar_ax, label="Damage Severity")

    fig.suptitle(f"Scatterplot Matrix of Weather Features vs. Damage, grouping: {dataset.timespan_calendar}, clusters: {num_clusters}", fontsize=10)
    plt.show()




dataset = ClusteredStormDamageDataset('../../Ressources/main_data_1972_2023.csv',
                                           '../../Ressources/weather_data4',
                                           '../../Ressources/municipalities_coordinates_newest.csv',
                                           'mean', 20,
                                           grouping_calendar='weekly',
                                           damage_weights={0: 0, 1: 0.06, 2: 0.8, 3: 11.3})

plot_scatter_matrix(dataset, feature_names=['Temperature', 'Sunshine', 'Rain', 'Snow'], num_clusters=20 )

