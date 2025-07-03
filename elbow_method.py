import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from data.empenhos_df import EMPENHOS



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
            k_values = range(2,6)
        elif 1000 > len(sub_ds) >=40:
            k_values = range(2,14)
        elif 2000 > len(sub_ds) >=1000:
            k_values = range(2,15)
        elif 10000 > len(sub_ds) >= 2000:
            k_values = range(3,18)
        else:
            k_values = range(4,28)
        
        ss = []  # Store the Silhouette Scores for each k
        for k in k_values:
            mb_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = mb_kmeans.fit_predict(sub_ds)
            
            if len(np.unique(clusters)) < 2 or len(np.unique(clusters)) >= len(sub_ds):
                ss.append(-1)  # Placeholder for invalid silhouette score
                continue
            
            score = silhouette_score(sub_ds, clusters)
            ss.append(score)
            
        if np.max(ss) < threshold_silhouette:
            optimal_k.append(1) # weak silhouette score
            print(f'Optimal k= 1. ss = {np.max(ss)}')
        else:
            best_k = k_values[np.argmax(ss)]
            print(f'Optimal k= {best_k}. ss = {np.max(ss):.2f}')
            optimal_k.append(best_k)
        


    # # Plot SSE vs K
    # plt.figure(figsize=(8, 5))
    # plt.plot(list(k_values), sse, marker='o')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Sum of Squared Errors (SSE)')
    # plt.title('SSE vs Number of Clusters')
    # # plt.axvline(optimal_k, color='red', linestyle='--', label=f'Elbow at k={optimal_k}')
    # plt.xticks(list(k_values))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('elbow_method/elbow_plot.png')
    # plt.show()

