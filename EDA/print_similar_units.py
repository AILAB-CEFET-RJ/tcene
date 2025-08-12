
import pandas as pd
from processing_utils import group_similar_categories


    
# Load the DataFrame from a Parquet file
df = pd.read_parquet('tce.parquet')


unidades = df['Unidade']
elemdespesatce = df['ElemDespesaTCE']


grouped_series, grouped = group_similar_categories(unidades, 95)

# Filter only modified categories (original values different from grouped values)
changed_groups = {original: grouped_value for original, grouped_value in grouped.items() if original != grouped_value}

# Print only the categories that were changed
print("Grouped Categories (Only Modified Entries):")
for original, grouped_value in changed_groups.items():
    print(f"{original} -> {grouped_value}")