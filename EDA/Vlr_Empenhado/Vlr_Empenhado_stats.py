import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

# Define the file path
file_path = 'C:/Users/parai/Documents/Github - tcene/tcene/tce.parquet'

# Load the DataFrame from a Parquet file
df = pd.read_parquet(file_path)

# Aplicando uma mascara no df para que Vlr_Empenhado esteja limitado
#limite =  1264406.607
#df = df[(df['Vlr_Empenhado'] > 0) & (df['Vlr_Empenhado'] <= limite)]

# Hiperparams
bin_width = df['Vlr_Empenhado'].std()
min_val = 0  # Ignoring negative values
max_val = df['Vlr_Empenhado'].max()
top_n = 5 # top_n being analysed
qtd_bins = 10

# Create bins
bins = np.arange(min_val, max_val + bin_width, bin_width)
df['bin'] = pd.cut(df['Vlr_Empenhado'], bins=bins, right=False)

# Compute total amount in each bin
bin_totals = df.groupby('bin', observed=False)['Vlr_Empenhado'].sum().reset_index()
bin_frequencies = df['bin'].value_counts().sort_index()
intervals = df['bin'].cat.categories

entropy_unid_list = []
top_N_cover_unid_list = []
entropy_elem_list = []
top_N_cover_elem_list = []
entropy_credor_list = []
top_N_cover_credor_list = []



unidades = (
    df.groupby('bin', observed=False)['Unidade']
    .value_counts()
)

elem_despesa = (
    df.groupby('bin', observed=False)['ElemDespesaTCE']
    .value_counts()
)

credores = (
    df.groupby('bin', observed=False)['Credor']
    .value_counts()
)




for bin_num in range(0, qtd_bins):

    target_bin = intervals[bin_num]  # First one is [0.0, 1264406.607)

    # Get the first unidade's, elemdespesa's and credor's bin to be analysed
    selected_bin_unid = unidades.loc[target_bin]
    selected_bin_elem = elem_despesa.loc[target_bin]
    selected_bin_credor = credores.loc[target_bin]

    selected_unid_top_n = selected_bin_unid.head(top_n)
    selected_elemdespesa_top_n = selected_bin_elem.head(top_n)
    selected_bin_credor_top_n = selected_bin_credor.head(top_n)
    
    rest_sum_unid = selected_bin_unid.iloc[top_n:].sum()
    rest_sum_elem = selected_bin_elem.iloc[top_n:].sum()
    rest_sum_credor = selected_bin_credor.iloc[top_n:].sum()
    selected_unid_top_n['rest'] = rest_sum_unid
    selected_elemdespesa_top_n['rest'] = rest_sum_elem
    selected_bin_credor_top_n['rest'] = rest_sum_credor

        

    # transform in percentage
    selected_unid_top_n_pct = selected_unid_top_n / selected_unid_top_n.sum() * 100
    selected_elem_top_n_pct = selected_elemdespesa_top_n / selected_elemdespesa_top_n.sum() * 100
    selected_credor_top_n_pct = selected_bin_credor_top_n / selected_bin_credor_top_n.sum() * 100
    
    
    

    # Calculate entropy of the selected bin for unidade
    bin_probs_unid = selected_bin_unid / selected_bin_unid.sum()
    entropy_unid_list.append(entropy(bin_probs_unid))
    
    # Calculate entropy of the selected bin for elemdespesa
    bin_probs_elem = selected_bin_elem / selected_bin_elem.sum()
    entropy_elem_list.append(entropy(bin_probs_elem)) 
    
    # Calculate entropy of the selected bin for credor
    bin_probs_credor = selected_bin_credor / selected_bin_credor.sum()
    entropy_credor_list.append(entropy(bin_probs_credor))
    

    # STATS: significance of top_n
    top_N_cover_unid_list.append(selected_unid_top_n_pct[:top_n-1].sum())
    top_N_cover_elem_list.append(selected_elem_top_n_pct[:top_n-1].sum())
    top_N_cover_credor_list.append(selected_credor_top_n_pct[:top_n-1].sum())



    # Prints
    print(f"\n\nbin {bin_num}: {target_bin}")
    print(f"Total amount of all Empenhos together: {bin_totals.iloc[bin_num]['Vlr_Empenhado']}")
    print(f"Amount of Empenhos in bin {bin_num}: {bin_frequencies.iloc[bin_num]}")
    
    print(f"Quantity of Unidades in bin {bin_num}: {len(selected_bin_unid)}")
    print("\nUNIDADE ", selected_unid_top_n_pct)
    print(f"\nEntropy of Unidade distribution in bin {bin_num}: {entropy_unid_list[bin_num]:.2f}")
    print(f"Top {top_n} Unidades cover: {top_N_cover_unid_list[bin_num]:.2f}% of bin {bin_num}\n")
    
    print(f"Quantity of Elem Despesas in bin {bin_num}: {len(selected_bin_elem)}")
    print("\nELEMDESPESATCE ", selected_elem_top_n_pct)
    print(f"\nEntropy of ElemDespesaTCE distribution in bin {bin_num}: {entropy_elem_list[bin_num]:.2f}")
    print(f"Top {top_n} ElemDespesaTCE cover: {top_N_cover_elem_list[bin_num]:.2f}% of bin {bin_num}\n")
    
    print(f"Quantity of Credores in bin {bin_num}: {len(selected_bin_credor)}")
    print("\nCREDOR ", selected_credor_top_n_pct)
    print(f"\nEntropy of Credor distribution in bin {bin_num}: {entropy_credor_list[bin_num]:.2f}")
    print(f"Top {top_n} Credor cover: {top_N_cover_credor_list[bin_num]:.2f}% of bin {bin_num}\n")
    

## PLOT - ENTROPY per BIN
# Change the last interval to be aggregated
labels = intervals[:qtd_bins]
new_intervals = list(labels)
last = new_intervals[-1]
new_intervals[-1] = pd.Interval(left=last.left, right=intervals[-1].right, closed=last.closed)
labels = pd.IntervalIndex(new_intervals)# Create a new IntervalIndex
labels = [f"{i.left / 1e6:.1f}M - {i.right / 1e6:.1f}M" for i in new_intervals]


x = np.arange(len(labels))  # bar positions

# Width of the bars
width = 0.2

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))

# Graystone, teal, and emerald 
bars1 = ax.bar(x - width, entropy_unid_list, width, label=f'Unidades', color='#7f7f7f')  # Graystone
bars2 = ax.bar(x, entropy_elem_list, width, label=f'ElemDespesaTCE', color='#17becf')  # Teal
bars3 = ax.bar(x + width, entropy_credor_list, width, label=f'Credor', color='#2ca02c')  # Emerald

# Labels and titles
# ax.set_xlabel('Intervalo em milhões de Reais (R$)')
# ax.set_ylabel('Entropia')
# ax.set_title('Comparação de Entropia por Intervalo')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

# Optional: add value labels on top of bars
for bar in bars1 + bars2 + bars3:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()






## PLOT TOP_N MOST COMMON % per BIN
x = np.arange(len(labels))  # bar positions

# Width of the bars
width = 0.2

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))

# Graystone, teal, and emerald 
bars1 = ax.bar(x - width, top_N_cover_unid_list, width, label=f'Unidades', color='#7f7f7f')  # Graystone
bars2 = ax.bar(x, top_N_cover_elem_list, width, label=f'ElemDespesaTCE', color='#17becf')  # Teal
bars3 = ax.bar(x + width, top_N_cover_credor_list, width, label=f'Credor', color='#2ca02c')  # Emerald

# Labels and titles
# ax.set_xlabel('Intervalo em milhões de Reais (R$)')
# ax.set_ylabel(f'Porcentagem das Top {top_n} Unidades, ElemDespesa ou Credor')
# ax.set_title(f'Comparação de Porcentagem do Top {top_n} Unidades/ElemDespesa/Credor por Intervalo')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

# Optional: add value labels on top of bars
for bar in bars1 + bars2 + bars3:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


