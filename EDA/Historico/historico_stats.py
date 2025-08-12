import yaml
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# Load the DataFrame from a Parquet file
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'
df = pd.read_parquet(file_path)

# Ensure 'Historico' column is a string
df['Historico'] = df['Historico'].astype(str)

# Calculate the number of words in each 'Historico'
df['word_count'] = df['Historico'].apply(lambda x: len(x.split()))

# Calculate mean and standard deviation of word counts
mean_word_count = df['word_count'].mean()
std_word_count = df['word_count'].std()

print(f"Mean number of words in 'Historico': {mean_word_count}")
print(f"Standard deviation of words in 'Historico': {std_word_count}")

# Get unique values
unique_historico = df['Historico'].unique() # NUMERO DE CATEGORIAS
num_unique_historico = len(unique_historico)

# Count occurrences of each unique value # NUMERO DE CATEGORIAS
historico_counts = Counter(df['Historico'])

# Get the most frequent values (top 10 most repeated)
top_N = 10
top_N_historico = historico_counts.most_common(top_N)

# Print statistics
print(f"Total number of 'Historico': {len(df['Historico'])}")
print(f"Number of unique 'Historico': {len(unique_historico)}")

# Print Top n most frequent "Historico"
"""print(f"Top {top_N} most frequent 'Historico':")

for idx, (historico, count) in enumerate(top_N_historico, start=1):
    percentage = count / len(df['Historico'])
    print(f"{idx}. {historico} ({count} occurrences - {percentage*100}%)")"""
    

frequencies = list(historico_counts.values())
#frequencies = df['Historico'].map(historico_counts)

# Define bin width
min_val = 0 
max_val = max(frequencies) 
bin_width_first = 1  # First bin_width --> until bin X
bin_width_second = 10 # Second bin_width --> from bin X to bin Y
bin_width_third = 50 # from bin Y to Z
bin_width_rest = 1000  # From bin Z onwards, new bin_width
until_bin_X = 20
until_bin_Y = 28
until_bin_Z = 30

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
aggregated_bins = [f"{bins[i]}" if i < until_bin_X else f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

print(bin_frequencies[:11])
print(bin_frequencies[:11]/len(df['Historico']))
print("ATENÇÃO - somatório de bin_frequencies != len(df['Historico]): ",bin_frequencies.sum())


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
bars = plt.bar(aggregated_bins, bin_frequencies, color=colors, edgecolor='black', alpha=0.7, log=True)

# Add legend
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_labels.keys()]
#plt.legend(handles, legend_labels.values(), title="Legenda", loc="upper right")

# Show every second label on the x-axis
plt.xticks(aggregated_bins, rotation=45, fontsize=8)

# Mostrar o gráfico
plt.show()



