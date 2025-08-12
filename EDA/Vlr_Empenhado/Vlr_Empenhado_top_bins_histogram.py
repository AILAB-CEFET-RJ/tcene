import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the file path
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'

# Load the DataFrame from a Parquet file
df = pd.read_parquet(file_path)

# Gerando estatísticas gerais
stats = df['Vlr_Empenhado'].describe()

# Definir os bins com base no intervalo desejado
bin_width = df['Vlr_Empenhado'].std()  # Tamanho do intervalo
max_val = df['Vlr_Empenhado'].max()

# Criar os bins de acordo com o intervalo especificado
bins = np.arange(0, max_val + bin_width, bin_width)

# Filtrar bins até o limite desejado (3.5e7)
bins = bins[bins <= 3.5e7]

# Mostrar a faixa de cada bin
print("Bin edges:", bins)
print(f"Each bin represents an interval of approximately {bin_width:.2f}")


hist, bin_edges = np.histogram(df['Vlr_Empenhado'], bins=bins)

# Get the top 10 bins with the highest frequencies
top_indices = np.argsort(hist)[-10:][::-1]  # Sort in descending order

# Print the top 10 bins and their frequency counts
print("Top 10 Bins (Highest Frequencies):")
for i, idx in enumerate(top_indices, start=1):
    bin_range = f"[{bin_edges[idx]:,.2f} - {bin_edges[idx+1]:,.2f}]"
    print(f"Top {i}: {hist[idx]} occurrences in range {bin_range}")



# Plotar o histograma com os bins personalizados
plt.figure(figsize=(12, 6))
sns.histplot(df['Vlr_Empenhado'], bins=bins, kde=False, log=True)

# Configurar rótulos e título
plt.xlabel("Valor Empenhado (R$)")
plt.ylabel("Frequência")
plt.title("Histograma de Frequência dos Valores Empenhados")

# Definir limites do eixo x e y
plt.xlim(0, 3.5e7)  # Limite do eixo x até 35,000,000


# Exibir a grade
plt.grid(True)
plt.show()


