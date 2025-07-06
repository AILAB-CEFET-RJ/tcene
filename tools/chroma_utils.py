from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma

from transformers import AutoTokenizer, AutoModel
import torch
import yaml


def filter_results(collection, embed_query, threshold=1.5):
    results = collection.query(
        query_embeddings=[embed_query],
        n_results=1000,  # or total documents if small
        include=["distances", "documents", "metadatas"]
    )
    
    threshold = 1.5

    filtered = [
        {
            "document": doc,
            "metadata": meta,
            "distance": dist
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
        if dist <= threshold
    ]
    return filtered


    
def create_embeddings(samples, model, tokenizer, batch_size=64):
    all_embeddings = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
            all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0).numpy()

def load_model_tokenizer():
    # Open config file
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
    # Load a pre-trained transformer model for embeddings
    print('Loading model..')
    model_name = config['embedding_model']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    return model, tokenizer


def load_vector_store(model):
    
    # Define where your DB was saved
    persist_dir = 'chroma/'

    # Reconnect to the persisted DB
    persistent_client = PersistentClient(path='chroma/')

    # Reconnect to the vector store
    vector_store = Chroma(
        client=persistent_client,
        collection_name="my_collection",
        embedding_function=model,
        persist_directory=persist_dir,
    )
    
    collection = persistent_client.get_collection("my_collection")
    
    return vector_store, persistent_client, collection
