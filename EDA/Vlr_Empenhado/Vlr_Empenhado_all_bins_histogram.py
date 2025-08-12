import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the file path
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'

# Load the DataFrame from a Parquet file
df = pd.read_parquet(file_path)

# Define bin width based on standard deviation
bin_width = df['Vlr_Empenhado'].std()
min_val = 0  # Ignoring negative values
max_val = df['Vlr_Empenhado'].max()

# Create bins
bins = np.arange(min_val, max_val + bin_width, bin_width)

# Assign bins to the data
df['bin'] = pd.cut(df['Vlr_Empenhado'], bins=bins, right=False)

# Compute total amount in each bin
bin_totals = df.groupby('bin')['Vlr_Empenhado'].sum().reset_index() 
# dataframe com 2 colunas: 
# [bin_range, qtd_total]
# 175 rows usando bin_width == std deviation

     
# Processando as bins
# Deixar as primeiras A bins, depois agregar de A em A da bin Y à Z e depois agregar de B em B à partir da bin Z
bin_A = 5
bin_B = 16
bin_C = 27
bin_D = 43
bin_E = 78
agregate_in_A = 2  # Aggregation step from bin "bin_A" to "bin_B"
agregate_in_B = 3  # Aggregation step from bin "bin_B" to "bin_C"
agregate_in_C = 5  # Aggregation step from bin "bin_C" to "bin_D"
agregate_in_D = 10 # Aggregation step from bin "bin_D" to "bin_E"
agregate_in_E = 23 # Aggregation step from bin "bin_E" onwards
num_bins = len(bin_totals)
aggregated_totals = []  # List to store final total values
aggregated_bins = []  # List to store final bin edges
bar_colors = []
aggregation_rules = [
    (bin_B, agregate_in_A, '#d4f4dd'),  # very light green
    (bin_C, agregate_in_B, '#82e0aa'),  # light-medium green
    (bin_D, agregate_in_C, '#f9e79f'),  # yellowish
    (bin_E, agregate_in_D, '#f5b041'),  # orange
    (num_bins, agregate_in_E, '#e74c3c')  # strong red
]


# Keep the first X bins as they are
if num_bins <= bin_A:
    
    aggregated_totals = list(bin_totals['Vlr_Empenhado'])

    aggregated_bins = [f"{interval.left / 1e6:.1f}M - {interval.right / 1e6:.1f}M" for interval in bin_totals['bin']]

    bar_colors = ['#a6c8ff'] * len(aggregated_totals) # light blue
else:
    aggregated_totals = list(bin_totals['Vlr_Empenhado'][:bin_A])
    aggregated_bins = [f"{interval.left / 1e6:.1f}M - {interval.right / 1e6:.1f}M" for interval in bin_totals['bin'][:bin_A]]
    bar_colors = ['#a6c8ff'] * bin_A  # default color for first bins = light blue
    
    # Aggregate bins according to 'aggregation rules'
    i = bin_A
    for bin_limit, group_size, color in aggregation_rules:
        while i < min(bin_limit, num_bins):
            end_idx = min(i + group_size, num_bins)
            aggregated_totals.append(bin_totals['Vlr_Empenhado'][i:end_idx].sum())
            aggregated_bins.append(f"{bin_totals['bin'][i ].left / 1e6:.1f}M - {bin_totals['bin'][end_idx - 1].right / 1e6:.1f}M")
            i = end_idx
            bar_colors.append(color)
        
        
        

# Plot total amount per bin
plt.figure(figsize=(12, 7))
plt.bar(range(len(aggregated_totals)), aggregated_totals, width=0.8, align='center', alpha=0.7, color=bar_colors, edgecolor='black', log=True)

# Set x-ticks with bin ranges

plt.xticks(range(len(aggregated_totals)), aggregated_bins ,fontsize=7, rotation=45)

# Labels and title
#plt.xlabel("Vlr_Empenhado Intervalos (em milhões de Reais - R$)")
#plt.ylabel("Valor Total em Reais (R$)")
#plt.title("Valor Total de Vlr_Empenhado por Intervalo")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.show()
