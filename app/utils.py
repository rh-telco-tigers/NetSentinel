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


def generate_response(input_text, remote_llm_client, max_answer_length):
    """
    Generate a response using the remote LLM client.

    Parameters:
        input_text (str): The user's input text.
        remote_llm_client (RemoteLLMClient): The client to interact with the remote LLM service.
        max_answer_length (int): The maximum length of the generated response.

    Returns:
        str: The generated response text.
    """
    try:
        response_text = remote_llm_client.generate_response(
            input_text=input_text,
            max_length=max_answer_length
        )
        if not response_text.strip():
            logger.error("Remote LLM returned an empty response.")
            return "Sorry, I couldn't generate a response at the moment."
        return response_text
    except Exception as e:
        logger.error(f"Error generating response from remote LLM: {e}")
        return "Sorry, I couldn't generate a response at the moment."


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

