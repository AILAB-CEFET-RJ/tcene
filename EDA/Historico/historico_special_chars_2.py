import pandas as pd
import re
from collections import Counter
from spellchecker import SpellChecker

# Load the DataFrame from a Parquet file
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'
df = pd.read_parquet(file_path)

# Ensure 'Historico' column is a string
df['Historico'] = df['Historico'].astype(str)


# Get unique values
unique_historico = df['Historico'].unique()
num_unique_historico = len(unique_historico)

# Count occurrences of each unique value
historico_counts = df['Historico'].value_counts()


# Function to detect special characters and their positions
def find_special_chars(text):
    return [(match.start(), match.group()) for match in re.finditer(r'[^a-zA-Z0-9\s]', text)]

# Apply function to detect special characters in each row
df['special_char_positions'] = df['Historico'].apply(find_special_chars)

# Extract all words and count occurrences
all_words = " ".join(df['Historico']).split()
word_counts = Counter(all_words)

# Initialize spell checker
spell = SpellChecker()

# Identify misspelled words and their context
def find_misspelled_words(text):
    words = text.split()
    misspelled = spell.unknown(words)
    misused_contexts = []

    for i, word in enumerate(words):
        if word in misspelled:
            # Extract surrounding words for context
            before = words[i-1] if i > 0 else ""
            after = words[i+1] if i < len(words) - 1 else ""
            misused_contexts.append(f"{before} *{word}* {after}")
    
    return misused_contexts

df['misspelled_context'] = df['Historico'].apply(find_misspelled_words)

# Filter only rows where there is at least one misspelled word
df_misspelled = df[df['misspelled_context'].apply(lambda x: len(x) > 0)]

# Additional filter: Select rows where 'Historico' contains more than three consecutive underscores '___'
df_misspelled_with_underscores = df_misspelled[df_misspelled['Historico'].str.contains(r'_{3,}', regex=True)]

# Print filtered results
print("\nMisspelled 'Historico' values with more than three consecutive underscores:")
print(df_misspelled_with_underscores[['Historico', 'misspelled_context']])

# Save results to CSV file
csv_filename = "misspelled_with_underscores.csv"
df_misspelled_with_underscores[['Historico', 'special_char_positions', 'misspelled_context']].to_csv(csv_filename, index=False)

print(f"\nFiltered results saved to {csv_filename}")
