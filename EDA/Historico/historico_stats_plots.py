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

# Calculate the total number of words in each 'Historico'
df['word_count'] = df['Historico'].apply(lambda x: len(x.split()))

# Count occurrences of each unique word count
word_count_counts = Counter(df['word_count'])

# Separate values < 54 and group values >= 54 in bins of 10
aggregated_counts = {}
for count, freq in word_count_counts.items():
    if count < 54:
        aggregated_counts[count] = freq  # Keep values as they are
    else:
        bin_start = (count // 10) * 10  # Group into bins of 10
        aggregated_counts[bin_start] = aggregated_counts.get(bin_start, 0) + freq  # Aggregate

# Sort the results
sorted_counts = dict(sorted(aggregated_counts.items()))

# Prepare data for plotting
word_counts = list(sorted_counts.keys())
frequencies = list(sorted_counts.values())

# Convert numeric bins to string labels for clarity
bin_labels = [str(wc) if wc < 54 else f"{wc}-{wc+9}" for wc in word_counts]

# Plot the frequency per total number of words
plt.figure(figsize=(12, 6))
plt.bar(bin_labels, frequencies, color='skyblue', edgecolor='black', alpha=0.7)
#plt.xlabel('Total Number of Words in "Historico" (Grouped)')
#plt.ylabel('Frequency')
#plt.title('Frequency per Total Number of Words in "Historico"')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
