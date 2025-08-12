import yaml
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'

# Load the DataFrame from a Parquet file
df = pd.read_parquet(file_path)

# Ensure 'elemdespesatce' column is a string
df['ElemDespesaTCE'] = df['ElemDespesaTCE'].astype(str)

# Get all unique "elemdespesa"
unique_elemdespesa = df['ElemDespesaTCE'].unique()

# Count occurrences of each "elemdespesa"
elemdespesa_counts = Counter(df['ElemDespesaTCE'])

top_N = 10
top_N_most_frequent = elemdespesa_counts.most_common(top_N)

# Print statistics
print(f"Total number of 'ElemDespesaTCE': {len(df['ElemDespesaTCE'])}")
print(f"Number of unique 'ElemDespesaTCE': {len(unique_elemdespesa)}")

# Print Top n most frequent "ELEMDESPESA"
print(f"Top {top_N} most frequent 'ElemDespesaTCE':")
for idx, (elemdespesa, count) in enumerate(top_N_most_frequent, start=1):
    percentage = count / len(df['ElemDespesaTCE'])
    print(f"{idx}. {elemdespesa} ({count} occurrences - {percentage})")
    


print('\n\n')
frequencies = list(elemdespesa_counts.values())

# Define bin width
min_val = 0 
max_val = max(frequencies) 
bin_width_first = 100  # First bin_width --> until bin X
bin_width_second = 1000 # Second bin_width --> from bin X to bin Y
bin_width_third = 5000 # from bin Y to Z
bin_width_rest = 10000  # From bin Z onwards, new bin_width
until_bin_X = 10
until_bin_Y = 15
until_bin_Z = 20

# Create bins
bins_first = np.arange(min_val, min_val + bin_width_first * until_bin_X, bin_width_first)  
bins_second = np.arange(min_val + bin_width_first * until_bin_X, 
                        min_val + bin_width_first * until_bin_X + bin_width_second * (until_bin_Y - until_bin_X), 
                        bin_width_second)
bins_third = np.arange(min_val + bin_width_first * until_bin_X + bin_width_second * (until_bin_Y - until_bin_X), 
                       min_val + bin_width_first * until_bin_X + bin_width_second * (until_bin_Y - until_bin_X) + bin_width_third * (until_bin_Z - until_bin_Y), 
                       bin_width_third)
bins_rest = np.arange(min_val + bin_width_first * until_bin_X + bin_width_second * (until_bin_Y - until_bin_X) + bin_width_third * (until_bin_Z - until_bin_Y), 
                      max_val + bin_width_rest, 
                      bin_width_rest)

# Combine all bins
bins = np.concatenate([bins_first, bins_second, bins_third, bins_rest])  


# Calcula a distribuição das frequências nos bins
bin_frequencies, _ = np.histogram(frequencies, bins)
aggregated_bins = [f"{bins[i+1]}" for i in range(len(bins) - 1)]


# Define colors based on bin sections
colors = []
for i in range(len(bin_frequencies)):
    if i < until_bin_X:
        colors.append('#82e0aa') 
    elif i < until_bin_Y:
        colors.append('#f9e79f') 
    elif i < until_bin_Z:
        colors.append('#f5b041') 
    else:
        colors.append('#e74c3c') 

# Definindo legendas
legend_labels = {
    '#82e0aa': 'Agregação de 100 em 100',
    '#f9e79f': 'Agregação de 1000 em 1000',
    '#f5b041': 'Agregação de 5000 em 5000',
    '#e74c3c': 'Agregação de 10000 em 10000'
}

# Criando o gráfico de barras (histograma)
plt.figure(figsize=(14, 9))
bars = plt.bar(aggregated_bins, bin_frequencies, color=colors, edgecolor='black', alpha=0.7)

# Add legend
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_labels.keys()]
plt.legend(handles, legend_labels.values(), title="Legenda", loc="upper right")

# Show every second label on the x-axis
plt.xticks(aggregated_bins, rotation=45, fontsize=8)

# Mostrar o gráfico
plt.show()


