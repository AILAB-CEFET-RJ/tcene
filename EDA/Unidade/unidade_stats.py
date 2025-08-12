import yaml
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'

# Load the DataFrame from a Parquet file
df = pd.read_parquet(file_path)

# Ensure 'unidade' column is a string
df['Unidade'] = df['Unidade'].astype(str)

# Get all unique "Unidades"
unique_unidades = df['Unidade'].unique()

# Count occurrences of each "Unidade"
unidade_counts = Counter(df['Unidade'])

top_N = 10
top_N_most_frequent = unidade_counts.most_common(top_N)

# Print statistics
print(f"Total number of 'Unidade': {len(df['Unidade'])}")
print(f"Number of unique 'Unidade': {len(unique_unidades)}")

# Print Top 5 most frequent "Unidade"
print(f"Top {top_N} most frequent 'Unidade':")
for idx, (unidade, count) in enumerate(top_N_most_frequent, start=1):
    percentage = count / len(df['Unidade'])
    print(f"{idx}. {unidade} ({count} occurrences - {percentage})")
    


print('\n\n')
frequencies = list(unidade_counts.values())

# Define bin width
min_val = 0 
max_val = max(frequencies) 
bin_width_first = 100  # First X bins width --> until bin X
bin_width_second = 500  # From bin X to Y, new width
bin_width_rest= 1000 # from bin Y onwards, new width
until_bin_X = 16
until_bin_Y = 32


# Create bins
bins_first = np.arange(min_val, min_val + bin_width_first * until_bin_X, bin_width_first)  
bins_second = np.arange(min_val + bin_width_first * until_bin_X, 
                        min_val + bin_width_first * until_bin_X + bin_width_second * (until_bin_Y - until_bin_X), 
                        bin_width_second)
bins_rest = np.arange(min_val + bin_width_first * until_bin_X + bin_width_second * (until_bin_Y - until_bin_X), 
                      max_val + bin_width_rest, 
                      bin_width_rest)

# Combine all bins
bins = np.concatenate([bins_first, bins_second, bins_rest])  


# Calcula a distribuição das frequências nos bins
bin_frequencies, _ = np.histogram(frequencies, bins)
aggregated_bins = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]


# Define colors based on bin sections
colors = []
for i in range(len(bin_frequencies)):
    if i < until_bin_X:
        colors.append('#82e0aa')  # Light green
    elif i < until_bin_Y:
        colors.append('#f9e79f') 
    else:
        colors.append('#e74c3c') 
        
        
legend_labels = {
    '#82e0aa': f'Agregação de {bin_width_first} em {bin_width_first}',
    '#f9e79f': f'Agregação de 500 em 500',
    '#e74c3c': f'Agregação de 1000 em 1000',
}


# Criando o gráfico de barras (histograma)
plt.figure(figsize=(14, 7))
bars = plt.bar(aggregated_bins, bin_frequencies, color=colors, edgecolor='black', alpha=0.7, log=True)

# Add legend
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_labels.keys()]
#plt.legend(handles, legend_labels.values(), title="Legenda", loc="upper right")

# Show every second label on the x-axis
plt.xticks(aggregated_bins[::2], rotation=45, fontsize=9)

# Mostrar o gráfico
plt.show()


