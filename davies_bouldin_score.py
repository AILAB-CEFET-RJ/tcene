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
from sklearn.metrics import davies_bouldin_score
import time
import matplotlib.pyplot as plt



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


features_tensor = torch.cat(features).numpy()
print(features_tensor.shape) 

dbi_scores = []
k_values = range(20, 71)

# https://towardsdatascience.com/davies-bouldin-index-for-k-means-clustering-evaluation-in-python-57f66da15cd/

for k in k_values:
    print(f'k = {k}')
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(features_tensor)
    dbi = davies_bouldin_score(features_tensor, labels)
    dbi_scores.append(dbi)
    print(f"dbi score: {dbi}")

print(dbi_scores)

k_optimal_idx = np.argmin(dbi_scores)
k_optimal = k_values[k_optimal_idx]

plt.figure(figsize=(15, 6))
plt.plot(k_values, dbi_scores, marker='o')
plt.axvline(x=k_optimal, color='r', linestyle='--', label=f'k ótimo = {k_optimal}')
plt.xticks(list(k_values))
plt.xlabel('Número de clusters (k)', fontsize=7)
plt.ylabel('Davies-Bouldin Index')
plt.title('DBI para diferentes valores de k')
plt.legend()
plt.savefig('davies_bouldin_score.png')
plt.grid(True)

"""
O menor valor de DBI indica o número de clusters mais apropriado.
DBI é mais estável com MiniBatchKMeans em grandes conjuntos.
"""