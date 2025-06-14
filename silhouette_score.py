import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from examples.empenhos_df import EMPENHOS
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
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
    postfix={
        "epo": -1,
        "acc": "%.4f" % 0.0,
        "lss": "%.8f" % 0.0,
        "dlb": "%.4f" % -1,
    },
    disable=False,
)
features = []

# Redução de dimensionalidade com o encoder (do AE)
model = torch.load('saved_models/dec_model_full.pt')
model.cuda()


for index, batch in enumerate(data_iterator):
    batch = batch.cuda(non_blocking=True)
    features.append(model.encoder(batch).detach().cpu()) # detach.cpu é importante para não haver CUDA OUT OF MEMORY

# verificação do shape depois de aplicar o encoder
features_tensor = torch.cat(features).numpy()
print(features_tensor.shape) # Dimensionalidade: 56 (reduziu de 387 para 56)

silhouettes = []
k_values = range(59, 61)

for k in k_values:
    print(f'cluster: {k}')
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init='auto', random_state=42) 
    predicted = kmeans.fit_predict(features_tensor)
    print(f'KMeans done for k={k}')
    

    start_time = time.time()
    ss = silhouette_score(features_tensor, predicted)
    elapsed_time = time.time() - start_time
    print(f'Silhouette score done for k={k} in {elapsed_time:.2f} seconds')
    silhouettes.append(ss)

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