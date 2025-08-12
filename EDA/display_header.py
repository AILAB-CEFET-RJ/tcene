import pandas as pd

# Define the file path
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'

# Load the DataFrame from a Parquet file
df = pd.read_parquet(file_path)

idcontrato = df['IdContrato'].astype(str)

# Filter rows where 'IdContrato' is not 0
valid_idcontrato_df = df[idcontrato != '0']


valid_idcontrato_df = valid_idcontrato_df.head(8)

# Ensure full text display
pd.set_option('display.max_colwidth', None)

# Select only the intended columns and convert them to string
filtered_df = valid_idcontrato_df[['IdContrato', 'Unidade', 'ElemDespesaTCE', 'Credor']].astype(str)

filtered_historico = valid_idcontrato_df[['Historico']]

# Display the filtered dataframe
print(filtered_df)
print(filtered_historico)

