import yaml
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import re

file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'

# Load the DataFrame from a Parquet file
df = pd.read_parquet(file_path)


unidade = df['Unidade']
elemdespesatce = df['ElemDespesaTCE']

# Vérifier aussi les chaînes vides
empty_unidade = (df['Unidade'] == "").sum()
empty_elemdespesatce = (df['ElemDespesaTCE'] == "").sum()

print(f"Nombre de chaînes vides dans 'Unidade' : {empty_unidade}")
print(f"Nombre de chaînes vides dans 'ElemDespesaTCE' : {empty_elemdespesatce}")
