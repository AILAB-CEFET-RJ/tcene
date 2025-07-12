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



dataset = EMPENHOS(
    train=False, val=False, testing_mode=False
)

# aplicação do elbow method na prática:
# https://medium.com/aimonks/knee-plot-algorithms-standardizing-the-trade-off-dilemma-72f53afd6452

# alternativas ao elbow method:
# https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6/

all = 129
threshold_silhouette = 0.25
optimal_k = []
        

autoencoder = torch.load('outputs/models/autoencoder_3d.pt', map_location=torch.device('cpu'), weights_only=False) 
for index in range(all):
    sub_ds, _ = dataset.X_by_elem(index)
    
    if len(sub_ds) == 1:
        optimal_k.append(1)
        print(f'Optimal k= 1')
    else:
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
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
        print(f'array shape: {array.shape}')
            
        
        
        if len(array) == 2:
            k_values = [2] # nao necessariamente 2 clusters é o ideal (pode ser que não existam clusters)
        elif 11 > len(array) > 2:
            k_values = [2,3] # nao necessariamente 2 clusters é o ideal (pode ser que não existam clusters)
        elif 40 > len(array) >= 11:
            k_values = range(2,10)
        elif 1000 > len(array) >=40:
            k_values = range(2,16)
        elif 2000 > len(array) >=1000:
            k_values = range(2,31)
        elif 10000 > len(array) >= 2000:
            k_values = range(2,51)
        else:
            k_values = range(2,61)
        
        
        ss = []  # Store the Silhouette Scores for each k
        for k in k_values:
            
            mb_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            
            
            clusters = mb_kmeans.fit_predict(array)
            
            if len(np.unique(clusters)) < 2 or len(np.unique(clusters)) >= len(array):
                ss.append(-1)  # Placeholder for invalid silhouette scores
                continue
            
            score = silhouette_score(array, clusters)
            ss.append(score)
            
            if len(np.unique(clusters)) < 2 or len(np.unique(clusters)) >= len(array):
                ss.append(-1)  # Placeholder for invalid k
                continue
        
            
        if np.max(ss) < threshold_silhouette:
            optimal_k.append(1) # weak silhouette score
            print(f'index = {index}, Optimal k= 1. ss = {np.max(ss):.2f}, size = {len(array)}')
        else:
            best_k = k_values[np.argmax(ss)]
            print(f'index = {index}, Optimal k= {best_k}. ss = {np.max(ss):.2f}')
            optimal_k.append(best_k)
        
            # save the png file
            plt.figure(figsize=(10, 10))
            plt.plot(list(k_values), ss, marker='o')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title(f'Silhouette Score Index = {index}')
            plt.xticks(rotation=90, fontsize=6)
            plt.tight_layout()
            plt.savefig(f'outputs/pngs/elbow_plot_{index}.png')


