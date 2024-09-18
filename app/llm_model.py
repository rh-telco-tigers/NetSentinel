# app/llm_model.py

import faiss
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import logging
import os

logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self, config):
        # Load embedding model
        embedding_model_name = config.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"Loaded embedding model '{embedding_model_name}'")

        # Load FAISS index and metadata
        faiss_index_path = config.get('FAISS_INDEX_PATH', 'faiss_index/index.faiss')
        metadata_path = config.get('METADATA_PATH', 'faiss_index/metadata.json')
        self.index, self.metadata_store = self.load_faiss_index_and_metadata(faiss_index_path, metadata_path)

        # Load the LLM model and tokenizer
        llm_model_name = config.get('LLM_MODEL_NAME', 'google/flan-t5-base')
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
        logger.info(f"Loaded LLM model '{llm_model_name}'")

    def load_faiss_index_and_metadata(self, index_file, metadata_file):
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            index = faiss.read_index(index_file)
            with open(metadata_file, 'r') as f:
                metadata_store = json.load(f)
            logger.info("FAISS index and metadata loaded")
            return index, metadata_store
        else:
            logger.error("FAISS index or metadata file not found.")
            return None, None

    def generate_response(self, query):
        # Retrieve relevant context
        context = self.retrieve_relevant_context(query)
        # Generate response using LLM
        response = self.generate_text(query, context)
        return response

    def retrieve_relevant_context(self, query, top_k=5):
        # Encode the query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        # Search in FAISS index
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        # Retrieve metadata
        context = ""
        for idx in indices[0]:
            if idx < len(self.metadata_store):
                item = self.metadata_store[idx]
                context += (
                    f"Event ID: {item.get('event_id', 'N/A')}, "
                    f"Prediction: {'Attack' if item.get('prediction') == 1 else 'Normal'}, "
                    f"Protocol: {item.get('protocol', 'N/A')}, "
                    f"Source IP: {item.get('src_ip', 'N/A')}, "
                    f"Destination IP: {item.get('dst_ip', 'N/A')}\n"
                )
        return context

    def generate_text(self, query, context):
        # Prepare the input for the model
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        inputs = self.tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

        # Generate the response
        outputs = self.model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
