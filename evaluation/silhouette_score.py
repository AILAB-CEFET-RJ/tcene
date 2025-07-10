import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from examples.empenhos_df import EMPENHOS
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import time
import matplotlib.pyplot as plt
from sklearn.utils import resample


# # Open the configuration file and load the different arguments
# with open('config.yaml') as f:
#     config = yaml.safe_load(f)
    
# # Load the DataFrame from a Parquet file
# df = pd.read_parquet('examples/tce.parquet')

torch.cuda.set_device(1)

ds_train = EMPENHOS(
    train=True, testing_mode=False
)  # training dataset


static_dataloader = DataLoader(
    ds_train,
    batch_size=256,
    shuffle=False, # os lotes serão na mesma ordem que no original
)



data_iterator = tqdm(
    static_dataloader,
    leave=True,
    unit="batch",
    disable=False,
)


features = []
for index, batch in enumerate(data_iterator):
    batch = batch.cuda(non_blocking=True)
    features.append(batch.detach().cpu())  # Apenas adiciona o batch, sem passar pelo encoder


features_np = torch.cat(features).numpy()
print(features_np.shape) 




silhouettes = []


from concurrent.futures import ProcessPoolExecutor, as_completed

def evaluate_k(k, features_np, n_rounds, n_samples):
    print(f'[Process] Starting k={k}')
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init='auto', random_state=42) 
    predicted = kmeans.fit_predict(features_np)

    silhouette_per_round = []
    for i in range(n_rounds):
        _, X_sample, y_sample = resample(
            np.arange(len(features_np)),
            features_np,
            predicted,
            stratify=predicted,
            n_samples=n_samples,
            random_state=42 + i
        )
        score = silhouette_score(X_sample, y_sample)
        silhouette_per_round.append(score)

    silhouette_mean = np.mean(silhouette_per_round)
    return k, silhouette_mean

# --- Parallel Execution ---
def parallel_kmeans_silhouette(k_values, features_np, n_rounds, n_samples):
    silhouettes = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_k, k, features_np, n_rounds, n_samples) for k in k_values]
        
        for future in as_completed(futures):
            k, silhouette_mean = future.result()
            print(f'[Result] k={k}, silhouette={silhouette_mean:.4f}')
            silhouettes.append((k, silhouette_mean))

    # Sort by k to maintain order
    silhouettes.sort()
    silhouette_values = [s for _, s in silhouettes]
    return silhouette_values

# --- Usage ---
k_values = range(47, 58)
n_rounds = 3
n_samples = 100000

silhouettes = parallel_kmeans_silhouette(k_values, features_np, n_rounds, n_samples)
print(silhouettes)



# Plot silhouette score vs K
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouettes, marker='o', color='green')
plt.xticks(list(k_values))
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.savefig('silhouette_plot.png')
plt.tight_layout()
plt.show()


"""
The value of the silhouette coefﬁcient is between [-1, 1]. 
A score of 1 denotes the best meaning that the data point o is very compact 
within the cluster to which it belongs and far away from the other clusters. 
The worst value is -1. Values near 0 denote overlapping clusters.
"""