import yaml
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np



file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'
# Load the DataFrame from a Parquet file
df = pd.read_parquet(file_path)

idcontrato = df['IdContrato'].astype(str).tolist()  # Contract IDs (as strings)

valid_idcontrato = [item for item in idcontrato if item != '0' and item.strip() != '' and item.lower() != 'nan']

# Count occurrences of each IdContrato
idcontrato_counts = Counter(valid_idcontrato)

# Create the idcontratos_per_item list with (idcontrato, num_times_it_repeats)
idcontratos_per_item = [(idcontrato, count) for idcontrato, count in idcontrato_counts.items()]

unique_idcontratos = [item for item in idcontratos_per_item if item[1]== 1]

# Get the top 5 most common contract IDs
top_N = 10
top_N_most_frequent = idcontrato_counts.most_common(top_N)

# Count the total number of distinct IdContrato values
total_distinct_idcontratos = len(set(valid_idcontrato))  # Or use df['IdContrato'].nunique()

print(f"Total number of distinct IdContratos: {total_distinct_idcontratos}")
print(f"number of valid contract Ids: {len(valid_idcontrato)}")
print(f"number of total contract Ids: {len(idcontrato)}")
print(f"Percentage of valid contract Ids: {100*len(valid_idcontrato)/len(idcontrato):2f}%")
print(f"number of unique id contracts: {len(unique_idcontratos)}")
print(f"Percentage of unique id contracts: {100*len(unique_idcontratos)/len(idcontrato):2f}%")
print(f"\nTop {top_N} most frequent contract IDs:")
for idx, (contract_id, count) in enumerate(top_N_most_frequent, start=1):
    percentage = count/total_distinct_idcontratos
    print(f"{idx}. {contract_id} ({count} occurrences - {percentage*100}%)")





# Get the frequency of each contract ID
frequencies = list(idcontrato_counts.values())


# Define bin width
min_val = 0 
max_val = max(frequencies) 
bin_width_first = 1  # First X bins width --> until bin X
bin_width_rest = 10  # From bin X and then, new width
until_bin_X = 10

# Create bins
bins_first = np.arange(min_val, min_val + bin_width_first * until_bin_X, bin_width_first)  # First 10 bins with width 1
bins_rest = np.arange(min_val + bin_width_first * until_bin_X, max_val + bin_width_rest, bin_width_rest)  # Remaining bins with width 200
bins = np.concatenate([bins_first, bins_rest])  # Combine the two ranges


# Calcula a distribuição das frequências nos bins
bin_frequencies, _ = np.histogram(frequencies, bins)
aggregated_bins = [f"{bins[i]}" if i<10 else f"{bins[i]} - {bins[i+1]}" for i in range(len(bins) - 1)]


print(bin_frequencies[:11]/total_distinct_idcontratos)

# Define colors: lighter for smaller bins, stronger for larger bins
colors = ['royalblue' if i < until_bin_X else 'midnightblue' for i in range(len(bin_frequencies))]




# Criando o gráfico de barras (histograma)
plt.figure(figsize=(14, 7))
plt.bar(aggregated_bins, bin_frequencies, color=colors, edgecolor='black', alpha=0.7, log=True)

# Show every second label on the x-axis
plt.xticks(aggregated_bins, rotation=45, fontsize = 8)

plt.show()


