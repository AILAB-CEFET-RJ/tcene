import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import MiniBatchKMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Open the configuration file and load the different arguments
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    
# Load the DataFrame from a Parquet file
df = pd.read_parquet('examples/tce.parquet')

directory = config['output_embeddings']
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]



all_embeddings = []
for file_path in files: # for each batch saved ..
    embeddings = np.load(file_path)
    
    all_embeddings.extend(embeddings)


# Convert the list of embeddings into a large NumPy array
X = np.vstack(all_embeddings)

unidades = df['Unidade']
elemdespesatce = df['ElemDespesaTCE']    
credor = df['Credor']

# Frequency encoding
frequency_unidades = unidades.value_counts(normalize=True)
frequency_elemdespesa = elemdespesatce.value_counts(normalize=True)
frequency_credor = credor.value_counts(normalize=True)  


# Map frequencies to original data
freq_uni = unidades.map(frequency_unidades).fillna(0).values.reshape(-1, 1)
freq_elem = elemdespesatce.map(frequency_elemdespesa).fillna(0).values.reshape(-1, 1)
freq_credor = credor.map(frequency_credor).fillna(0).values.reshape(-1, 1)


# Apply StandardScaler to each variable
scaler = StandardScaler()
freq_uni = scaler.fit_transform(freq_uni).astype(np.float32)
freq_elem = scaler.fit_transform(freq_elem).astype(np.float32)
freq_credor = scaler.fit_transform(freq_credor).astype(np.float32)


# hstack: Used to add features (columns) to existing rows.
X_new = np.hstack([X, freq_uni, freq_elem, freq_credor])

print(X_new.shape)



# Step 2: Run k-means for different values of k and compute SSE (Sum of Squared Errors)

sse = []  # Store the SSE for each k
silhouettes = [] # Store silhouette score for each k
k_values = range(55, 65)

for k in k_values:
    mb_kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, random_state=42)
    clusters = mb_kmeans.fit_predict(X_new)  # Fit the model to the data
    ss = silhouette_score(X_new, clusters)
    silhouettes.append(ss)


# Plot silhouette score vs K
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouettes, marker='o', color='green')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.tight_layout()
plt.show()


"""
The value of the silhouette coefﬁcient is between [-1, 1]. 
A score of 1 denotes the best meaning that the data point o is very compact 
within the cluster to which it belongs and far away from the other clusters. 
The worst value is -1. Values near 0 denote overlapping clusters.
"""