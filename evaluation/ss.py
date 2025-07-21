import os
import numpy as np
from sklearn.cluster import KMeans
from ptdec.model import train, predict
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from data.empenhos_df import EMPENHOS
from ptdec.dec import DEC
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans


dataset = EMPENHOS(
    train=False, val=False, testing_mode=False
)

# aplicação do elbow method na prática:
# https://medium.com/aimonks/knee-plot-algorithms-standardizing-the-trade-off-dilemma-72f53afd6452

# alternativas ao elbow method:
# https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6/

print("Loading model...")
autoencoder = torch.load('outputs/models/autoencoder_full.pt', map_location=torch.device('cpu'), weights_only=False) 
print("Model loaded!")

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
array = torch.cat(encoded_outputs, dim=0).numpy()
print(f'Testing k_values for {array.shape}')


k_values = range(390, 392) 

ss = []  # Store the Silhouette Scores for each k
for k in k_values:
    mb_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init=10)
    clusters = mb_kmeans.fit_predict(array)

    score = silhouette_score(array, clusters)
    ss.append(score)
    print(f'ss = {score} with k = {k}')

# save the png file
plt.figure(figsize=(10, 10))
plt.plot(list(k_values), ss, marker='o')
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig(f'outputs/pngs/elbow_plot_ss.png')

best_k = k_values[np.argmax(ss)]
print(f'Best_K: {best_k}')
max_ss = max(ss)
print(f'Maximum Silhouette Score: {max_ss}')


