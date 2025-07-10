import os
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
import yaml
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tools.chroma_utils import create_embeddings



with open('config.yaml') as f:
    config = yaml.safe_load(f)


df = pd.read_parquet(config['parquet_path'])
raw_documents_to_embed = df[['Historico', 'Unidade', 'ElemDespesaTCE', 'Credor']]
vlr_empenhado = df['Vlr_Empenhado']


## Excluir o que está abaixo
## --------------------------------------------------------------------------------- ##
elems = pd.unique(raw_documents_to_embed['ElemDespesaTCE'])
index = 7
mask = (raw_documents_to_embed['ElemDespesaTCE'] == elems[index]).values
raw_documents_grouped = raw_documents_to_embed.iloc[mask]
vlr_empenhado_grouped = vlr_empenhado.iloc[mask]
samples = raw_documents_grouped.astype(str).agg(', '.join, axis=1)
## --------------------------------------------------------------------------------- ##

# clusters = ...

#samples = raw_documents_to_embed.astype(str).agg(', '.join, axis=1)
print(type(samples))  # deve ser pd.Series


# Load a pre-trained transformer model for embeddings
model_name = config['embedding_model']
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print('Creating Embeddings...')
embeddings = create_embeddings(samples, model, tokenizer)


### Initializing Chromadb and collection
persistent_dir = 'data/chroma_db/'
os.makedirs(persistent_dir, exist_ok=True)

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("my_collection")

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="my_collection",
    embedding_function=model,
    persist_directory=persistent_dir
)

documents = [
    Document(
        page_content=row,
        metadata={
            'Vlr_Empenhado': vlr_empenhado_grouped.iloc[index],
            # 'Clusters': clusters.iloc[index]
        }
    )
    for index, row in enumerate(samples.tolist())
]

assert len(documents) == len(samples), "Mismatch between documents and embeddings!"

print('Inserting Documents to DB')
BATCH_SIZE = 5461
for i in range(0, len(documents), BATCH_SIZE):
    if i + BATCH_SIZE > len(documents):
        batch_docs = documents[i:]
    else:
        batch_docs = documents[i:i + BATCH_SIZE]
    collection.add(
        documents=[doc.page_content for doc in batch_docs],
        metadatas=[doc.metadata for doc in batch_docs],
        embeddings=embeddings[i:i + BATCH_SIZE].tolist(),
        ids=[f"doc_{j}" for j in range(i, i + len(batch_docs))]
    )
    
    
### Little query for testing

collection = persistent_client.get_collection("my_collection")
print(collection.count())


query_str = "TRATA SE DE DESPESA COM PAGAMENTO DE FGTS DOS SERVIDORES DA SAUDE NO MES DE JANEIRO DE 2018  RAT PREFEITURA ANGRA DOS REIS CONTRIBUICAO PARA O FGTS CAIXA ECONOMICA FEDERAL"

embed_query = create_embeddings(pd.Series(query_str), model, tokenizer)[0]


results = vector_store_from_client.similarity_search_by_vector(
    embedding=embed_query,
    k=2,
    # filter={"Cluster": "5"},
)
print(results)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")