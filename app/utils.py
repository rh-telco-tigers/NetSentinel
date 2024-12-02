# app/utils.py

import logging
from pymilvus import Collection
from datetime import datetime
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

def get_event_by_id(event_id, collection: Collection):
    expr = f'metadata["event_id"] == "{event_id}"'
    results = collection.query(expr=expr, output_fields=["metadata"])
    return results[0]['metadata'] if results else None

def get_events_by_src_ip(src_ip, collection: Collection):
    expr = f'metadata["src_ip"] == "{src_ip}"'
    results = collection.query(expr=expr, output_fields=["metadata"])
    return [result['metadata'] for result in results]

def get_events_by_dst_ip(dst_ip, collection: Collection):
    expr = f'metadata["dst_ip"] == "{dst_ip}"'
    results = collection.query(expr=expr, output_fields=["metadata"])
    return [result['metadata'] for result in results]

def get_all_attack_event_ids(collection: Collection):
    # Adjust the prediction value based on your actual attack class labels
    attack_prediction_values = [1]  # Replace with actual values if different
    expr = f'metadata["prediction"] in {attack_prediction_values}'
    results = collection.query(expr=expr, output_fields=["metadata"])
    return [result['metadata']['event_id'] for result in results]

def get_recent_events(collection: Collection, event_type: str, limit: int = 10) -> list:
    if event_type not in ['attack', 'normal']:
        logger.error(f"Invalid event_type: {event_type}. Must be 'attack' or 'normal'.")
        return []

    # Adjust the prediction values based on your actual class labels
    prediction_values = {
        'attack': [1],    # Replace with actual attack prediction values
        'normal': [0]     # Replace with actual normal prediction values
    }
    expr = f'metadata["prediction"] in {prediction_values[event_type]}'
    results = collection.query(expr=expr, output_fields=["metadata"])

    # Extract metadata from results
    events = [result['metadata'] for result in results]

    try:
        sorted_events = sorted(
            events,
            key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"),
            reverse=True
        )
    except KeyError:
        logger.error("One or more events are missing the 'timestamp' field.")
        sorted_events = []
    except ValueError as ve:
        logger.error(f"Timestamp format error: {ve}")
        sorted_events = []

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
        f"Prediction Probability: {event_data.get('probabilities', 'N/A')}"
    ]
    return "\n".join(fields)

def generate_response(input_text, remote_llm_client, max_answer_length):
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
    namespace = entities.get('namespace')

    if namespace:
        if 'namespace:' in namespace:
            namespace = namespace.replace('namespace:', '').strip()

        return namespace.strip()

    return None
