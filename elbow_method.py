import os
import numpy as np
from sklearn.cluster import KMeans
from ptdec.model import train, predict
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from data.empenhos_df import EMPENHOS
from ptdec.dec import DEC
import torch



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

autoencoder = torch.load(f'/outputs/models/autoencoder_full.pt', map_location=torch.device('cpu'), weights_only=False) 
for index in range(all):
    sub_ds, _ = dataset.X_by_elem(index)
    if len(sub_ds) == 1:
        optimal_k.append(1)
        print(f'Optimal k= 1')
    else:
        if len(sub_ds) == 2:
            k_values = [2] # nao necessariamente 2 clusters é o ideal (pode ser que não existam clusters)
        elif 11 > len(sub_ds) > 2:
            k_values = [2,3] # nao necessariamente 2 clusters é o ideal (pode ser que não existam clusters)
        elif 40 > len(sub_ds) >= 11:
            k_values = range(2,10)
        elif 1000 > len(sub_ds) >=40:
            k_values = range(2,14)
        elif 2000 > len(sub_ds) >=1000:
            k_values = range(2,25)
        elif 10000 > len(sub_ds) >= 2000:
            k_values = range(2,56)
        else:
            k_values = range(2,71)
        
        
        ss = []  # Store the Silhouette Scores for each k
        for k in k_values:
            
            model = DEC(cluster_number=k, hidden_dimension=10, encoder=autoencoder.encoder)
            dec_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0004, weight_decay=1e-5)
            train(
                dataset=sub_ds,
                model=model,
                epochs=20, # 20 epocas para testar o num optimal de clusters
                lr=0.0004,
                batch_size=256,
                optimizer=dec_optimizer,
                stopping_delta=0.000001,
                cuda=False,
                evaluate_batch_size=512,
                silent=True # treina baaixo
            )
            
            X, predicted = predict( # we don't have actual values, so False
                    sub_ds, model, batch_size=512, silent=True, return_actual=False, cuda=False
            )
            
            if len(np.unique(predicted)) < 2 or len(np.unique(predicted)) >= len(sub_ds):
                ss.append(-1)  # Placeholder for invalid k
                continue
        

            score = silhouette_score(sub_ds, predicted)
            ss.append(score)
            
        if np.max(ss) < threshold_silhouette:
            optimal_k.append(1) # weak silhouette score
            print(f'index = {index}, Optimal k= 1. ss = {np.max(ss):.2f}, size = {len(sub_ds)}')
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
            plt.savefig(f'elbow_plot_{index}.png')


