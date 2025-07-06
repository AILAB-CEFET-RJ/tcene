import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class EMPENHOS(Dataset):
    def __init__(self, train=True, val=False, testing_mode=False):
        self.testing_mode = testing_mode

        # Load config
        with open('config.yaml') as f:
            config = yaml.safe_load(f)

        # Load parquet DataFrame (can be used later for metadata or filtering)
        self.df = pd.read_parquet(config['parquet_path'])

        # Load all .npy embeddings
        directory = config['output_embeddings']
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]

        all_embeddings = []
        for file_path in files:
            embeddings = np.load(file_path)
            all_embeddings.append(torch.tensor(embeddings, dtype=torch.float16))

        # Stack into one big tensor
        all_X  = torch.vstack(all_embeddings) # 384 features
        
        
        # Adding all categories
        unidades = self.df['Unidade']
        elemdespesatce = self.df['ElemDespesaTCE']    
        credor = self.df['Credor']
        # idcontrato = self.df['IdContrato']

        # Frequency encoding
        frequency_unidades = unidades.value_counts(normalize=True)
        frequency_elemdespesa = elemdespesatce.value_counts(normalize=True)
        frequency_credor = credor.value_counts(normalize=True)  
        
        # Map frequencies to original data
        freq_uni = unidades.map(frequency_unidades).fillna(0).values.reshape(-1, 1)
        freq_elem = elemdespesatce.map(frequency_elemdespesa).fillna(0).values.reshape(-1, 1)
        freq_credor = credor.map(frequency_credor).fillna(0).values.reshape(-1, 1)


        # Apply StandardScaler to each variable
        scaler_uni = StandardScaler()
        scaler_elem = StandardScaler()
        scaler_credor = StandardScaler()
        freq_uni = scaler_uni.fit_transform(freq_uni).astype(np.float32)
        freq_elem = scaler_elem.fit_transform(freq_elem).astype(np.float32)
        freq_credor = scaler_credor.fit_transform(freq_credor).astype(np.float32)
        
        
        # Concatenate all columns as torch tensors
        freq_uni_tensor = torch.tensor(freq_uni, dtype=torch.float32)
        freq_elem_tensor = torch.tensor(freq_elem, dtype=torch.float32)
        freq_credor_tensor = torch.tensor(freq_credor, dtype=torch.float32)
        X = torch.cat([all_X, freq_uni_tensor, freq_elem_tensor, freq_credor_tensor], dim=1)

        # Limit to 50000 samples for testing
        if self.testing_mode:
            X = X[:50000]
            
        # Train/Val split (90%/10%)
        split_idx = int(0.90 * len(X))
        if train:
            self.X = X[:split_idx]
        elif val:
            self.X = X[split_idx:]
        else:
            self.X = X  
        
    
    def X_by_elem(self, index):
        elemdespesatce = self.df['ElemDespesaTCE']   
        elems = pd.unique(elemdespesatce)
        mask = (elemdespesatce == elems[index]).values
        X_grouped = self.X[mask]
        return X_grouped, elems[index]
    
    def lenElemDespesaTCE(self):
        uniqueElemDespesa = np.unique(self.df['ElemDespesaTCE'])
        return len(uniqueElemDespesa)
    

    def __getitem__(self, index):
        return self.X[index]  # No label, unsupervised

    def __len__(self):
        return self.X.shape[0]





