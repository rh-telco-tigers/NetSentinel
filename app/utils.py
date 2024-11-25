# app/utils.py

import logging
import os
import json
import faiss
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import Dict


logger = logging.getLogger(__name__)

def setup_logging(log_level='DEBUG', log_file=None):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_faiss_index_and_metadata(faiss_index_path, metadata_store_path):
    try:
        with open(faiss_index_path, 'rb') as faiss_file:
            faiss_index = faiss.deserialize_index(faiss_file.read())
    except Exception as e:
        logging.error(f"Failed to load FAISS index from {faiss_index_path}: {e}")
        faiss_index = None  # Decide on a default behavior or raise if critical

    try:
        with open(metadata_store_path, 'r') as f:
            metadata_store = json.load(f)
    except JSONDecodeError as jde:
        logging.error(f"JSON decode error while loading metadata_store from {metadata_store_path}: {jde}")
        metadata_store = {}  # Assign a default empty dict or handle as needed
    except Exception as e:
        logging.error(f"Failed to load metadata_store from {metadata_store_path}: {e}")
        metadata_store = {}  # Assign a default empty dict or handle as needed

    return faiss_index, metadata_store

# Helper functions used in intent_handlers.py
def get_event_by_id(event_id, event_id_index):
    return event_id_index.get(event_id)

def get_events_by_src_ip(src_ip, metadata_store):
    return [item for item in metadata_store if item.get('src_ip') == src_ip]

def get_events_by_dst_ip(dst_ip, metadata_store):
    return [item for item in metadata_store if item.get('dst_ip') == dst_ip]

def get_all_attack_event_ids(metadata_store):
    return [item['event_id'] for item in metadata_store if item.get('prediction') == 1]

def get_recent_events(metadata_store: list, event_type: str, limit: int = 10) -> list:
    """
    Fetch recent events based on the event type and limit.
    
    Parameters:
        metadata_store (list): The list containing all event data.
        event_type (str): Type of events to fetch ('attack' or 'normal').
        limit (int): Number of recent events to retrieve.
    
    Returns:
        list: A list of recent events matching the specified type.
    """
    if event_type not in ['attack', 'normal']:
        logger.error(f"Invalid event_type: {event_type}. Must be 'attack' or 'normal'.")
        return []
    
    # Define the prediction value based on event type
    prediction_value = 1 if event_type == 'attack' else 0
    
    # Filter events based on prediction
    filtered_events = [event for event in metadata_store if event.get('prediction') == prediction_value]
    
    # Sort events by timestamp descending
    try:
        sorted_events = sorted(filtered_events, key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"), reverse=True)
    except KeyError:
        logger.error("One or more events are missing the 'timestamp' field.")
        sorted_events = []
    except ValueError as ve:
        logger.error(f"Timestamp format error: {ve}")
        sorted_events = []
    
    # Return the top 'limit' events
    return sorted_events[:limit]

def build_context_from_event_data(event_data):
    fields = [
        f"Event ID: {event_data.get('event_id', 'N/A')}",
        f"Prediction: {'Attack' if event_data.get('prediction') == 1 else 'Normal'}",
        f"Protocol: {event_data.get('protocol', 'N/A')}",
        f"Service: {event_data.get('service', 'N/A')}",
        f"State: {event_data.get('state', 'N/A')}",
        f"Source IP: {event_data.get('src_ip', 'N/A')}",
        f"Destination IP: {event_data.get('dst_ip', 'N/A')}",
        f"Prediction Probability: {event_data.get('prediction_proba', 'N/A')}"
    ]
    return "\n".join(fields)

def generate_response(input_text, tokenizer, llm_model, llm_model_type, max_context_length, max_answer_length):
    """
    Generate a response using the appropriate LLM model type.
    """
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=max_context_length,
        truncation=True
    ).to(llm_model.device)

    if llm_model_type == 'seq2seq':
        outputs = llm_model.generate(
            inputs,
            max_length=max_answer_length,
            num_beams=5,
            early_stopping=True
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    elif llm_model_type == 'causal':
        total_max_length = inputs.shape[1] + max_answer_length
        outputs = llm_model.generate(
            inputs,
            max_length=total_max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the generated answer
        response_text = full_response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
    else:
        logger.error(f"Unsupported llm_model_type: {llm_model_type}")
        raise ValueError(f"Unsupported llm_model_type: {llm_model_type}")

    return response_text


def extract_namespace(entities: Dict) -> str:
    """
    Extract the namespace from entities, handling both 'namespace:' and regular namespace references.
    """
    namespace = entities.get('namespace')
    
    if namespace:
        if 'namespace:' in namespace:
            namespace = namespace.replace('namespace:', '').strip()

        return namespace.strip()

    return None

