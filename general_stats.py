import yaml
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Open the configuration file and load the different arguments
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    
# Load the DataFrame from a Parquet file
df = pd.read_parquet('tce.parquet')

idcontrato = df['IdContrato'].astype(str).tolist()  # Contract IDs (as strings)

valid_idcontrato = [item for item in idcontrato if item!= '0']

# Count occurrences of each IdContrato
idcontrato_counts = Counter(valid_idcontrato)

# Create the idcontratos_per_item list with (idcontrato, num_times_it_repeats)
idcontratos_per_item = [(idcontrato, count) for idcontrato, count in idcontrato_counts.items()]

unique_idcontratos = [item for item in idcontratos_per_item if item[1]== 1]

# Get the top 5 most common contract IDs
top_5_most_frequent = idcontrato_counts.most_common(5)

print(f"number of valid contract Ids: {len(valid_idcontrato)}")
print(f"number of total contract Ids: {len(idcontrato)}")
print(f"Percentage of valid contract Ids: {100*len(valid_idcontrato)/len(idcontrato):2f}%")
print(f"number of unique id contracts: {len(unique_idcontratos)}")
print(f"Percentage of unique id contracts: {100*len(unique_idcontratos)/len(idcontrato):2f}%")
print("\nTop 5 most frequent contract IDs:")
for idx, (contract_id, count) in enumerate(top_5_most_frequent, start=1):
    print(f"{idx}. {contract_id} ({count} occurrences)")



# Get the frequency of each contract ID
frequencies = list(idcontrato_counts.values())

# Plot histogram of frequencies
plt.hist(frequencies, bins=range(1, max(frequencies) + 1))
plt.xlabel('Frequency of contract ID')
plt.ylabel('Number of contract IDs')
plt.title('Distribuição de frequências para o IdContrato')
# Apply log scale to the y-axis
plt.yscale('log')
plt.show()
