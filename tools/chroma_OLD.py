import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
import yaml
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Getting started with Chroma DB
# https://medium.com/@pierrelouislet/getting-started-with-chroma-db-a-beginners-tutorial-6efa32300902 

def create_vector_store():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    os.environ['OPENAI_API_KEY'] = config['API_key']
    
    embedding_model = SentenceTransformer(config['embedding_model'])
        
    df = pd.read_parquet(config['parquet_path'])
    raw_documents = df[['Historico', 'Unidade', 'ElemDespesaTCE', 'Credor', 'Vlr_Empenhado']]

    print('Step 1: Extracting the Embeds')
    directory = config['output_embeddings']
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]

    all_embeddings = []
    for file_path in files:
        embeddings = np.load(file_path)
        all_embeddings.append(embeddings.astype(np.float32))  # Ensure float32 dtype

    # Combine all into one NumPy array
    all_embeddings_np = np.vstack(all_embeddings)  # Shape: (N, 384)

    print('Step 2: Convert text to Document objects')
    documents = [
        Document(
            page_content=row['Historico'],
            metadata={
                'Unidade': row['Unidade'],
                'ElemDespesaTCE': row['ElemDespesaTCE'],
                'Credor': row['Credor'],
                'Vlr_Empenhado': row['Vlr_Empenhado'],
            }
        )
        for _, row in raw_documents.iterrows()
    ]
    assert len(documents) == all_embeddings_np.shape[0], "Mismatch between documents and embeddings!"


    print('Step 3:  Create the Chroma vector store')
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")


    BATCH_SIZE = 5461
    for i in range(0, len(documents), BATCH_SIZE):
        if i + BATCH_SIZE > len(documents):
            batch_docs = documents[i:]
        else:
            batch_docs = documents[i:i + BATCH_SIZE]
        collection.add(
            documents=[doc.page_content for doc in batch_docs],
            metadatas=[doc.metadata for doc in batch_docs],
            embeddings=all_embeddings_np[i:i + BATCH_SIZE].tolist(),
            ids=[f"doc_{j}" for j in range(i, i + len(batch_docs))]
        )
    
    vector_store = Chroma(client=chroma_client, collection_name="my_collection", embedding_function=embedding_model, persist_directory=config['vector_store_dir']) 
    return vector_store