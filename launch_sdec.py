import os
from keras_sdec import DeepEmbeddingClustering
import yaml
import numpy as np
import pandas as pd
from processing_utils import transformDataInCategory, divideDataset


# Open the configuration file and load the different arguments
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    
# Load the DataFrame from a Parquet file
df = pd.read_parquet('tce.parquet')

directory = config['output_embeddings']
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]

all_embeddings = []
for file_path in files: # for each batch saved ..
    # Load the stored tuples of (embedding, contract_id)
    embeddings = np.load(file_path)
    
    all_embeddings.extend(embeddings)
    

    
# Convert the list of embeddings into a large NumPy array
X = np.vstack(all_embeddings)



"""
Adding unidades and elemdespesatce
"""
unidades = df['Unidade']
elemdespesatce = df['ElemDespesaTCE']

# unidades_one_hot is going to return a table with 1484918 rows and 771 columns
# there are 771 unidades, so for each row, it's going to return true for its corresponding unidade
# and false for all the others
unidades_one_hot = transformDataInCategory(unidades)

# same thing goes for the elemdespesatce column
elemdespesatce_one_hot = transformDataInCategory(elemdespesatce)


# Next steps: concatenate unidades_one_hot and elemdespesa_one_hot to 'all_embeddings'


# Labels dataset
Y = np.array(df['IdContrato'].astype(str).tolist())  # # empty IdContrato values are '0'


# note that if X.shape and Y.shape are not the same, it will error.
# make sure to have all the embeddings
X_non_zero, X_zeros, Y_non_zero, Y_zeros = divideDataset(X, Y)


n_clusters = 20  # Assuming 10 clusters
input_dim = 384  # Your embedding dimension


print(X_non_zero.shape)
print(Y_non_zero.shape)


# Set the seed for reproducibility
np.random.seed(42)

# Determine how many samples to select (half of the dataset)
sample_size = X_non_zero.shape[0] // 10

# Randomly sample indices
indices = np.random.choice(X_non_zero.shape[0], sample_size, replace=False)

# Use the sampled indices to create a new subset of the data
X_non_zero_reduced = X_non_zero[indices]
Y_non_zero_reduced = Y_non_zero[indices]


print(X_non_zero_reduced.shape)
print(Y_non_zero_reduced.shape)



# Initialize the DeepEmbeddingClustering model
c = DeepEmbeddingClustering(n_clusters=n_clusters, input_dim=input_dim)


# Initialize the model with the embeddings
c.initialize(X_non_zero_reduced, finetune_iters=100, layerwise_pretrain_iters=50)  # Adjust iterations as necessary

# Perform clustering
c.cluster(X_non_zero_reduced, y=Y_non_zero_reduced, iter_max=10)
