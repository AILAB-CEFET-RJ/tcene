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



dataset = EMPENHOS(
    train=False, val=False, testing_mode=False
)


autoencoder = torch.load('outputs/models/autoencoder_full.pt', map_location=torch.device('cpu'), weights_only=False) 

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


# Encode all data
encoded_outputs = []

autoencoder.eval()  # Set to evaluation mode
with torch.no_grad():
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch

        # inputs = inputs.to(device)  # Move to GPU if needed
        encoded = autoencoder.encoder(inputs)
        encoded_outputs.append(encoded.cpu())

# Concatenate all batches
features_tensor = torch.cat(encoded_outputs, dim=0).numpy()


print(features_tensor.shape) 

dbi_scores = []
k_values = range(64, 129)

# https://towardsdatascience.com/davies-bouldin-index-for-k-means-clustering-evaluation-in-python-57f66da15cd/

for k in k_values:
    print(f'k = {k}')
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(features_tensor)
    dbi = davies_bouldin_score(features_tensor, labels)
    dbi_scores.append(dbi)
    print(f"dbi score: {dbi}")



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