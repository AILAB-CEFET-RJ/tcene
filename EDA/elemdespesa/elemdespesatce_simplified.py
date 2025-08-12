import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the DataFrame from a Parquet file
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'
df = pd.read_parquet(file_path)

# Ensure 'ElemDespesaTCE' column is a string
df['ElemDespesaTCE'] = df['ElemDespesaTCE'].astype(str)

# Get all unique values in 'ElemDespesaTCE'
unique_elemdespesatce = df['ElemDespesaTCE'].unique()

# Count occurrences of each 'ElemDespesaTCE'
elemdespesatce_counts = Counter(df['ElemDespesaTCE'])

# Get the top 10 most common 'ElemDespesaTCE'
top_10_most_frequent = elemdespesatce_counts.most_common(10)

# Extract frequency values
frequencies = [count for count in elemdespesatce_counts.values()]

# Print statistics
print(f"Total number of 'ElemDespesaTCE': {len(df['ElemDespesaTCE'])}")
print(f"Number of unique 'ElemDespesaTCE': {len(unique_elemdespesatce)}")

# Print Top 10 most frequent 'ElemDespesaTCE'
print("Top 10 most frequent 'ElemDespesaTCE':")
for idx, (elem, count) in enumerate(top_10_most_frequent, start=1):
    print(f"{idx}. {elem} ({count} occurrences)")
    


# Additional stats for inconsistencies
total_count = len(df['ElemDespesaTCE'])
unique_count = len(unique_elemdespesatce)
single_occurrence_count = sum(1 for count in elemdespesatce_counts.values() if count == 1)
max_occurrence = max(elemdespesatce_counts.values())
min_occurrence = min(elemdespesatce_counts.values())
mean_occurrence = sum(elemdespesatce_counts.values()) / unique_count

# Print additional statistics
print("\nAdditional Statistics:")
print(f"Total number of occurrences: {total_count}")
print(f"Number of unique values: {unique_count}")
print(f"Number of values appearing only once: {single_occurrence_count}")
print(f"Percentage of values appearing only once: {100 * single_occurrence_count / unique_count:.2f}%")
print(f"Maximum occurrences of a single value: {max_occurrence}")
print(f"Minimum occurrences (excluding single occurrences): {min_occurrence if min_occurrence > 1 else 'N/A'}")
print(f"Mean occurrences per unique value: {mean_occurrence:.2f}")



# Plot histogram of frequencies
plt.figure(figsize=(10, 6))
plt.hist(frequencies, bins=100, edgecolor='black', alpha=0.75)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()