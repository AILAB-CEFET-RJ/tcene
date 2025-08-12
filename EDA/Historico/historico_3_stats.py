import pandas as pd
import numpy as np
from collections import Counter
import re

# Load the DataFrame from a Parquet file
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'
df = pd.read_parquet(file_path)

# Ensure 'Historico' column is treated as string
df['Historico'] = df['Historico'].astype(str)

# Check for missing values
missing_values = df['Historico'].isna().sum()
total_values = len(df)
missing_percentage = (missing_values / total_values) * 100

# Count unique values
unique_values = df['Historico'].nunique()

# Get all words from 'Historico' column
all_words = " ".join(df['Historico']).split()
word_counts = Counter(all_words)
most_common_words = word_counts.most_common(10)

# Character-level analysis: Identify non-alphabetic characters
special_characters = re.findall(r'[^a-zA-Z0-9\s]', " ".join(df['Historico']))
special_char_counts = Counter(special_characters)

# Sentence Length Analysis
text_lengths = df['Historico'].apply(lambda x: len(x.split()))

# Spell Checking
spell = SpellChecker()
misspelled_words = set(spell.unknown(all_words))
misspelled_counts = {word: word_counts[word] for word in misspelled_words if word in word_counts}
top_misspelled = sorted(misspelled_counts.items(), key=lambda x: x[1], reverse=True)[:10]

# Print Special Character Analysis
print("Special Character Frequencies:")
print(special_char_counts)


# Print Most Common Misspelled Words
print("Most Common Misspelled Words:")
print(top_misspelled)

# Save Key Findings to CSV
key_findings = {
    "Total Rows": [total_values],
    "Missing Values": [missing_values],
    "Missing Percentage": [f"{missing_percentage:.2f}%"],
    "Unique Descriptions": [unique_values],
    "Most Common Words": [most_common_words],
    "Top Special Characters": [special_char_counts.most_common(10)],
    "Sentence Length (Mean)": [np.mean(text_lengths)],
    "Sentence Length (Std Dev)": [np.std(text_lengths)],
    "Most Common Misspelled Words": [top_misspelled]
}

# Convert dictionary to DataFrame
key_findings_df = pd.DataFrame.from_dict(key_findings)

# Save the findings to a CSV file
csv_filename = "historico_findings.csv"
key_findings_df.to_csv(csv_filename, index=False)

print(f"\nKey findings saved to {csv_filename}")
