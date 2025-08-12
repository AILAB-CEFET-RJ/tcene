import pandas as pd
import re
from collections import Counter

# Load the DataFrame from a Parquet file
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'
df = pd.read_parquet(file_path)

# Ensure 'Historico' column is a string
df['Historico'] = df['Historico'].astype(str)

# Function to find special characters in each row
def find_special_chars(text):
    return re.findall(r'[^a-zA-Z0-9\s]', text)

# Apply function to detect special characters per row
df['special_chars'] = df['Historico'].apply(find_special_chars)

# Count rows that contain special characters
rows_with_special_chars = df[df['special_chars'].apply(lambda x: len(x) > 0)].shape[0]

# Count total occurrences of each special character
all_special_chars = [char for sublist in df['special_chars'] for char in sublist]
special_char_counts = Counter(all_special_chars)

# Convert results to a DataFrame
summary_df = pd.DataFrame(special_char_counts.items(), columns=['Special Character', 'Frequency'])
summary_df = summary_df.sort_values(by='Frequency', ascending=False)

# Print the summary
print(f"Total rows with special characters: {rows_with_special_chars} / {len(df)} ({(rows_with_special_chars / len(df)) * 100:.2f}%)")
print("\nSpecial Character Frequencies:")
print(summary_df)

# Optionally, save to a CSV file
summary_df.to_csv("special_character_analysis.csv", index=False)
