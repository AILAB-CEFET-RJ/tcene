import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import MiniBatchKMeans


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


# Train/Val split (90%/10%)
split_idx = int(0.90 * len(X_new))

X_new = X_new[split_idx:]    

print('generating clusters')
mb_kmeans = MiniBatchKMeans(n_clusters=40, batch_size=1024, n_init=20, random_state=42)
clusters = mb_kmeans.fit_predict(X_new)  # Fit the model to the data

# Combine X_new and clusters into a DataFrame
df_out = pd.DataFrame(np.hstack([X_new, clusters.reshape(-1, 1)]))

# Save to Excel file
df_out.to_excel('output.xlsx', index=False)