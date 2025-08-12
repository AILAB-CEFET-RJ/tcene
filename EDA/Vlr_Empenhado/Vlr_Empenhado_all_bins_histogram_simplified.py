import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the DataFrame
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'
df = pd.read_parquet(file_path)

# Ignore negative values
df = df[df['Vlr_Empenhado'] >= 0]

# Get min and max
min_val = 0
max_val = df['Vlr_Empenhado'].max()

# Define 10 bins
bins = np.linspace(min_val, max_val, 11)  # 11 edges = 10 bins
labels = [f'{int(bins[i]):,} - {int(bins[i+1]):,}' for i in range(10)]

# Assign each row to a bin
df['Range'] = pd.cut(df['Vlr_Empenhado'], bins=bins, labels=labels, include_lowest=True)

# Group by bin and sum
bin_sums = df.groupby('Range')['Vlr_Empenhado'].sum()

# Plot
plt.figure(figsize=(12, 6))
bin_sums.plot(kind='bar', log=True)
plt.title('Total Vlr_Empenhado por Intervalor de Valor')
plt.ylabel('Total Vlr_Empenhado')
plt.xlabel('Intervalo em Valor total (BRL)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
# Set y-axis to start at 10^7
plt.ylim(1e8, bin_sums.max() * 1.1)  # small margin on top for aesthetics

plt.tight_layout()
plt.show()
