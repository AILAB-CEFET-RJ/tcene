import torch
import os
from keras_sdec import DeepEmbeddingClustering
import yaml
import numpy as np
import pandas as pd
from processing_utils import transformDataInCategory

# Open the configuration file and load the different arguments
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    
# Load the DataFrame from a Parquet file
df = pd.read_parquet('tce.parquet')

directory = config['output_embeddings']
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]

all_embeddings = []
id_contratos = df['IdContrato'].astype(str).tolist()  # Ensure it's a list of strings

for file_path in files: # for each batch saved ..
    # Load the stored tuples of (index, embedding, contract_id)
    embeddings = np.load(file_path)
    
    all_embeddings.extend(embeddings)
    
# Convert the list of embeddings into a large NumPy array
X = np.vstack(all_embeddings)
    
unidades = df['Unidades']
elemdespesatce = df['ElemDespesaTCE']

# unidades_one_hot is going to return a table with 1484918 rows and 771 columns
# there are 771 unidades, so for each row, it's going to return true for its corresponding unidade
# and false for all the others
unidades_one_hot = transformDataInCategory(unidades)

# same thing goes for the elemdespesatce column
elemdespesatce_one_hot = transformDataInCategory(elemdespesatce)


# Next steps: concatenate unidades_one_hot and elemdespesa_one_hot to 'all_embeddings'

y = df['IdContrato'].astype(str).tolist()  # Ensure it's a list of strings

# empty IdContrato values are '0'


# Initialize the DeepEmbeddingClustering model
n_clusters = 10  # Assuming 10 clusters
input_dim = X  # Your embedding dimension


c = DeepEmbeddingClustering(n_clusters=n_clusters, input_dim=input_dim)


# Initialize the model with the embeddings
c.initialize(X, finetune_iters=100, layerwise_pretrain_iters=50)  # Adjust iterations as necessary

# Perform clustering
#c.cluster(X, y=Y, iter_max=10)
