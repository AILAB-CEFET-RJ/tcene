import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the DataFrame from a Parquet file
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'
df = pd.read_parquet(file_path)


df['Credor'] = df['Credor'].astype(str)

# Get all unique values in 'Credor'
categorias_credor = df['Credor'].unique()

# Count occurrences of each 'Credor'
credor_counts = Counter(df['Credor'])

# Get the top 10 most common 'Credor'
top_10_most_frequent = credor_counts.most_common(10)


# Print statistics
print(f"Total number of 'Credor': {len(df['Credor'])}")
print(f"Number of categorias of 'Credor': {len(categorias_credor)}")

# Print Top 10 most frequent 'Credor'
print("Top 10 most frequent 'Credor':")
for idx, (elem, count) in enumerate(top_10_most_frequent, start=1):
    percentage = count / len(df['Credor'])
    print(f"{idx}. {elem} ({count} occurrences - {percentage*100:.2f}%)")
    
    
# Verify if there are missing values
empty_credor = (df['Credor'] == "").sum()
    


# Additional stats for inconsistencies
total_count = len(df['Credor'])
unique_count = len(categorias_credor)
single_occurrence_count = sum(1 for count in credor_counts.values() if count == 1)
max_occurrence = max(credor_counts.values())
min_occurrence = min(credor_counts.values())
mean_occurrence = sum(credor_counts.values()) / unique_count

# Print additional statistics
print("\nAdditional Statistics:")
print(f"Total number of occurrences: {total_count}")
print(f"Number of MISSING VALUES: {empty_credor}")
print(f"Number of unique values: {unique_count}")
print(f"Number of values appearing only once: {single_occurrence_count}")
print(f"Percentage of values appearing only once: {100 * single_occurrence_count / unique_count:.2f}%")
print(f"Maximum occurrences of a single value: {max_occurrence}")
print(f"Minimum occurrences (excluding single occurrences): {min_occurrence if min_occurrence > 1 else 'N/A'}")
print(f"Mean occurrences per unique value: {mean_occurrence:.2f}")


import numpy as np
# Get the frequency of each contract ID
frequencies = list(credor_counts.values())

# Print the number of 'Credor' that appear only once
for i in range(0,11):
    credores_appearance = sum(1 for freq in frequencies if freq == i)
    percentage_appearance = (credores_appearance / len(frequencies)) * 100
    print(f"{i} > {credores_appearance} ({percentage_appearance:.2f}%)")







# Define bin width
min_val = 0 
max_val = max(frequencies) 
bin_width_first = 1  # First bin_width --> until bin X
bin_width_second = 2 # Second bin_width --> from bin X to bin Y
bin_width_third = 5 # from bin Y to Z
bin_width_forth = 25  # From bin Z to W
bin_width_rest = 8000  # From bin W onwards, new bin_width
until_bin_A = 10
until_bin_B = 20
until_bin_C = 25
until_bin_D = 32

# Create bins
bins_first = np.arange(min_val, min_val + bin_width_first * until_bin_A, bin_width_first)  
bins_second = np.arange(min_val + bin_width_first * until_bin_A, 
                        min_val + bin_width_first * until_bin_A + bin_width_second * (until_bin_B - until_bin_A), 
                        bin_width_second)
bins_third = np.arange(min_val + bin_width_first * until_bin_A + bin_width_second * (until_bin_B - until_bin_A), 
                       min_val + bin_width_first * until_bin_A + bin_width_second * (until_bin_B - until_bin_A) + bin_width_third * (until_bin_C - until_bin_B), 
                       bin_width_third)
bins_forth = np.arange(min_val + bin_width_first * until_bin_A + bin_width_second * (until_bin_B - until_bin_A) + bin_width_third * (until_bin_C - until_bin_B), 
                       min_val + bin_width_first * until_bin_A + bin_width_second * (until_bin_B - until_bin_A) + bin_width_third * (until_bin_C - until_bin_B) + bin_width_forth * (until_bin_D - until_bin_C), 
                       bin_width_forth)
bins_rest = np.arange(min_val + bin_width_first * until_bin_A + bin_width_second * (until_bin_B - until_bin_A) + bin_width_third * (until_bin_C - until_bin_B) + bin_width_forth * (until_bin_D - until_bin_C), 
                      max_val + bin_width_rest, 
                      bin_width_rest)

# Combine all bins
bins = np.concatenate([bins_first, bins_second, bins_third, bins_forth, bins_rest])  


# Calcula a distribuição das frequências nos bins
bin_frequencies, _ = np.histogram(frequencies, bins)
aggregated_bins = [f"{bins[i]}" if i<until_bin_A else f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]




# Define colors based on bin sections
colors = []
for i in range(len(bin_frequencies)):
    if i < until_bin_A:
        colors.append('#ADD8E6') 
    elif i < until_bin_B:
        colors.append('#82e0aa') 
    elif i < until_bin_C:
        colors.append('#f9e79f') 
    elif i < until_bin_D:
        colors.append('#f5b041') 
    else:
        colors.append('#e74c3c')  # Additional color for bins beyond until_bin_D


# Criando o gráfico de barras (histograma)
plt.figure(figsize=(14, 9))
bars = plt.bar(aggregated_bins, bin_frequencies, color=colors, edgecolor='black', alpha=0.7, log=True)

# Add grid lines for the y-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show every second label on the x-axis
plt.xticks(aggregated_bins, rotation=45, fontsize=7)

# Mostrar o gráfico
plt.show()




