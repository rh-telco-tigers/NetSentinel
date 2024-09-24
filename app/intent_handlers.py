# app/intent_handlers.py

import logging
from .utils import (
    get_event_by_id,
    get_all_attack_event_ids,
    get_events_by_src_ip,
    get_events_by_dst_ip,
    build_context_from_event_data,
    generate_response
)

logger = logging.getLogger(__name__)

def handle_greet(entities, **kwargs):
    return "Hello! How can I assist you today?"

def handle_goodbye(entities, **kwargs):
    return "Goodbye! If you have more questions, feel free to ask."

def handle_get_event_info(entities, event_id_index, **kwargs):
    event_id = entities.get('event_id')
    if not event_id:
        return "Please provide a valid event ID."

    event_data = get_event_by_id(event_id, event_id_index)
    if not event_data:
        return "No data found for the provided event ID."

    text = entities.get('text', '').lower()
    if "source ip" in text:
        return f"Source IP: {event_data.get('src_ip', 'N/A')}"
    elif "destination ip" in text:
        return f"Destination IP: {event_data.get('dst_ip', 'N/A')}"
    elif "attack" in text:
        prediction = event_data.get('prediction')
        return "Yes, it is an attack." if prediction == 1 else "No, it is not an attack."
    elif "prediction probability" in text:
        proba = event_data.get('prediction_proba', 'N/A')
        return f"Prediction Probability: {proba}"
    elif "what kind of traffic" in text or "traffic type" in text:
        protocol = event_data.get('protocol', 'N/A')
        service = event_data.get('service', 'N/A')
        return f"Protocol: {protocol}, Service: {service}"
    else:
        return build_context_from_event_data(event_data)

def handle_list_attack_events(entities, metadata_store, **kwargs):
    event_ids = get_all_attack_event_ids(metadata_store)
    if event_ids:
        return "Attack Event IDs:\n" + "\n".join(event_ids)
    else:
        return "No attack events found."

def handle_get_events_by_ip(entities, metadata_store, **kwargs):
    ip_address = entities.get('ip_address')
    if not ip_address:
        return "Please provide a valid IP address."

    text = entities.get('text', '').lower()
    if "source ip" in text:
        events = get_events_by_src_ip(ip_address, metadata_store)
    elif "destination ip" in text:
        events = get_events_by_dst_ip(ip_address, metadata_store)
    else:
        return "Please specify whether you are interested in source IP or destination IP."

    if events:
        return "\n".join([f"Event ID: {e['event_id']}" for e in events])
    else:
        return "No events found for the specified IP."

def handle_ask_who_are_you(entities, **kwargs):
    return "I am a network monitoring assistant designed to help you analyze events, attacks, and IP traffic."

def handle_ask_how_are_you(entities, **kwargs):
    return "I'm just a program, but thank you for asking! How can I help you?"

def handle_ask_help(entities, **kwargs):
    return ("I can help you with queries regarding network events, attacks, and IP information.\n"
            "For example:\n"
            "- Get details about an event by its ID\n"
            "- List attack events\n"
            "- Find events related to a specific IP address\n"
            "How can I assist you today?")

def handle_thank_you(entities, **kwargs):
    return "You're welcome! I'm here to help whenever you need me."

def handle_ask_farewell(entities, **kwargs):
    return "Goodbye! Looking forward to helping you again."

def handle_ask_joke(entities, **kwargs):
    return "Why don't computers get tired? Because they have chips to keep them going!"

def handle_ask_capabilities(entities, **kwargs):
    return ("I can help you with the following:\n"
            "- Look up network events by event ID\n"
            "- List events by IP address (source/destination)\n"
            "- Identify attack events\n"
            "- Provide IP-related event details\n"
            "How can I assist you today?")

def handle_general_question(entities, **kwargs):
    """
    Handle general technical questions by generating a response using the LLM.
    """
    user_text = entities.get('text', '')
    tokenizer = kwargs.get('tokenizer')
    llm_model = kwargs.get('llm_model')
    llm_model_type = kwargs.get('llm_model_type', 'seq2seq')

    if not all([user_text, tokenizer, llm_model]):
        logger.error("Missing components for generating LLM response.")
        return "Sorry, I couldn't process your request at the moment."

    # Generate a response using the LLM
    input_text = f"User asked: {user_text}\nPlease provide a helpful and accurate response."
    try:
        response_text = generate_response(
            input_text,
            tokenizer,
            llm_model,
            llm_model_type,
            max_context_length=512,
            max_answer_length=150
        )
        return response_text
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return "Sorry, I couldn't generate a response at the moment."

def handle_fallback(entities, **kwargs):
    return "Sorry, I didn't understand that. Can you please rephrase?"


# Mapping of intents to handler functions
INTENT_HANDLERS = {
    'greet': handle_greet,
    'goodbye': handle_goodbye,
    'get_event_info': handle_get_event_info,
    'list_attack_events': handle_list_attack_events,
    'get_events_by_ip': handle_get_events_by_ip,
    'ask_who_are_you': handle_ask_who_are_you,
    'ask_how_are_you': handle_ask_how_are_you,
    'ask_help': handle_ask_help,
    'thank_you': handle_thank_you,
    'ask_farewell': handle_ask_farewell,
    'ask_joke': handle_ask_joke,
    'ask_capabilities': handle_ask_capabilities,
    'general_question': handle_general_question,
    'fallback': handle_fallback,
}
