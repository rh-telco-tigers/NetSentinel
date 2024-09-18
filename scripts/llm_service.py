# app/llm_service.py

import faiss
import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer  # Added import
import logging
import yaml

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_faiss_index_and_metadata(index_file='faiss_index/index.faiss', metadata_file='faiss_index/metadata.json'):
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        index = faiss.read_index(index_file)
        with open(metadata_file, 'r') as f:
            metadata_store = json.load(f)
        return index, metadata_store
    else:
        logger.error("FAISS index or metadata file not found.")
        return None, None

def retrieve_relevant_data(query, embedding_model, index, metadata_store, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), top_k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata_store):
            results.append(metadata_store[idx])
    return results

def generate_response(query, context, tokenizer, model):
    # Combine query and context
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

    # Generate response
    outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def handle_query(query, embedding_model, index, metadata_store, tokenizer, model):
    retrieval_results = retrieve_relevant_data(query, embedding_model, index, metadata_store)
    context = ""
    for item in retrieval_results:
        context += (
            f"Event ID: {item.get('event_id', 'N/A')}, "
            f"Prediction: {'Attack' if item.get('prediction') == 1 else 'Normal'}, "
            f"Protocol: {item.get('protocol', 'N/A')}, "
            f"Source IP: {item.get('src_ip', 'N/A')}, "
            f"Destination IP: {item.get('dst_ip', 'N/A')}\n"
        )
    response = generate_response(query, context, tokenizer, model)
    return response

def main():
    config = load_config()
    embedding_model_name = config.get('embedding_model', {}).get('name', 'all-MiniLM-L6-v2')
    embedding_model = SentenceTransformer(embedding_model_name)
    logger.info(f"Embedding model '{embedding_model_name}' loaded")

    # Load FAISS index and metadata store
    index, metadata_store = load_faiss_index_and_metadata()
    if index is None or metadata_store is None:
        logger.error("Failed to load FAISS index and metadata. Exiting.")
        return

    # Load tokenizer and model
    model_name = "google/flan-t5-base"  # You can choose other models if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logger.info(f"Loaded model '{model_name}'")

    # Example usage
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = handle_query(query, embedding_model, index, metadata_store, tokenizer, model)
        print("Response:", response)

if __name__ == "__main__":
    main()
