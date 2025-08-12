import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the file path
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'

# Load the DataFrame from a Parquet file
df = pd.read_parquet(file_path)

# Gerando estatísticas gerais
stats = df['Vlr_Empenhado'].describe()


"""
    Definir os bins com base no intervalo desejado
"""
bin_width = df['Vlr_Empenhado'].std()  # Tamanho do intervalo
min_val = 0 #df['Vlr_Empenhado'].min()  # AO COLOCAR MIN_VAL = 0, OS 'VLR_EMPENHO' NEGATIVOS SÃO DESPREZADOS
max_val = df['Vlr_Empenhado'].max()

# Criar os bins de acordo com o intervalo especificado
bins = np.arange(min_val, max_val + bin_width, bin_width)

# Compute histogram frequencies
hist, bin_edges = np.histogram(df['Vlr_Empenhado'], bins=bins)

# Process bins: keep first 20, then aggregate every 20 bins
num_bins = len(hist)
aggregated_freq = []  # List to store final frequencies
aggregated_bins = []  # List to store final bin edges

# Keep the first 20 bins as they are
if num_bins <= 20:
    aggregated_freq = list(hist)
    aggregated_bins = list(bin_edges)
else:
    aggregated_freq = list(hist[:20])
    aggregated_bins = list(bin_edges[:21])  # Need 1 more edge than frequencies

    # Aggregate remaining bins in groups of 20
    for i in range(20, num_bins, 20):
        end_idx = min(i + 20, num_bins)  # Ensure not exceeding length
        aggregated_freq.append(sum(hist[i:end_idx]))  # Sum frequencies
        aggregated_bins.append(bin_edges[end_idx])  # Store new bin edge

# Plot the frequency histogram
plt.figure(figsize=(12, 8))
plt.bar(range(len(aggregated_freq)), aggregated_freq, width=0.8, align='center', alpha=0.7, color='b', log=True)

# Set x-ticks with bin ranges
tick_labels = [f"{aggregated_bins[i] / 1e7:.1f}e7 - {aggregated_bins[i+1] / 1e7:.1f}e7" for i in range(len(aggregated_freq))]
plt.xticks(range(len(aggregated_freq)), tick_labels, rotation=45, ha='right', fontsize=8)


# Labels and title
plt.xlabel("Vlr_Empenhado Ranges")
plt.ylabel("Frequency")
plt.title("Histogram of Vlr_Empenhado")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.show()
        