# app/routes.py

from flask import Blueprint, request, jsonify, current_app
import logging
import numpy as np
import asyncio
import hmac
import hashlib
import time

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# Import the intent handlers
from .intent_handlers import INTENT_HANDLERS
from .utils import generate_response

# processed_event_ids = set()

def verify_slack_request(signing_secret, request):

    timestamp = request.headers.get('X-Slack-Request-Timestamp', '')
    if not timestamp:
        logger.warning("No X-Slack-Request-Timestamp header found.")
        return False

    try:
        timestamp = int(timestamp)
    except ValueError:
        logger.warning("Invalid X-Slack-Request-Timestamp header.")
        return False

    # Prevent replay attacks by checking the timestamp
    if abs(time.time() - timestamp) > 60 * 5:
        logger.warning("Request timestamp is too old.")
        return False

    sig_basestring = f"v0:{timestamp}:{request.get_data(as_text=True)}"
    my_signature = 'v0=' + hmac.new(
        signing_secret.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()

    slack_signature = request.headers.get('X-Slack-Signature', '')
    if not slack_signature:
        logger.warning("No X-Slack-Signature header found.")
        return False

    if not hmac.compare_digest(my_signature, slack_signature):
        logger.warning("Invalid Slack signature.")
        return False

    return True

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

# Slack Integration Class
class SlackHandler:
    def __init__(self, app):
        self.signing_secret = app.config['SLACK_CONFIG'].get('slack_signing_secret')
        self.bot_user_id = app.config['SLACK_CONFIG'].get('bot_user_id')
        self.slack_client = app.persistent_state.get('slack_client')

        if not self.signing_secret:
            logger.error("Slack signing secret is not configured.")
            raise ValueError("Slack signing secret is missing.")
        if not self.bot_user_id:
            logger.error("Slack bot user ID is not configured.")
            raise ValueError("Slack bot user ID is missing.")
        if not self.slack_client:
            logger.error("Slack client is not initialized.")
            raise ValueError("Slack client is missing.")

    def verify_request(self, request):
        return verify_slack_request(self.signing_secret, request)

    def is_bot_message(self, user_id):
        return user_id == self.bot_user_id

    def send_message(self, channel, text):
        self.slack_client.send_message(channel, text)

# Intent Handlers
def handle_get_event_info(entities, event_id_index):
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

def handle_list_attack_events(metadata_store):
    event_ids = get_all_attack_event_ids(metadata_store)
    if event_ids:
        return "Attack Event IDs:\n" + "\n".join(event_ids)
    else:
        return "No attack events found."

def handle_get_events_by_ip(entities, metadata_store):
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

# Route Handlers
@api_bp.route('/')
def health_check():
    return 'OK', 200

@api_bp.route('/slack/events', methods=['POST'])
def slack_events():
    data = request.get_json()
    if not data:
        logger.warning("No data received from Slack.")
        return jsonify({"error": "No data received"}), 400

    # Initialize Slack Handler
    try:
        slack_handler = SlackHandler(current_app)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    # Verify request
    if not slack_handler.verify_request(request):
        return jsonify({"error": "Invalid request signature"}), 403

    # Handle URL Verification challenge
    if 'challenge' in data:
        return jsonify({"challenge": data['challenge']}), 200

    event = data.get('event', {})
    logger.info(f"Event Data from slack: {event}")
    # event_id = event.get('event_id')
    user_text = event.get('text')
    channel = event.get('channel')
    user = event.get('user')

    # Check if the event has already been processed
    # if event_id in processed_event_ids:
    #     logger.info(f"Event ID {event_id} has already been processed. Skipping.")
    #     return jsonify({"status": "Event already processed"}), 200

    # Add the event_id to the processed set
    # processed_event_ids.add(event_id)

    if not user_text:
        logger.warning("No text found in Slack event.")
        return jsonify({"error": "No text found in event"}), 400

    if not channel:
        logger.warning("No channel specified in Slack event.")
        return jsonify({"error": "No channel specified"}), 400

    logger.info(f"Bot User ID: {slack_handler.bot_user_id}")
    logger.info(f"User ID in the event: {user}")

    if slack_handler.is_bot_message(user):
        logger.info("Message is from the bot itself. Ignoring.")
        return jsonify({"status": "Message from bot ignored"}), 200

    try:
        # Access models and data from app's persistent_state
        nlu_interpreter = current_app.persistent_state.get('nlu_interpreter')
        metadata_store = current_app.persistent_state.get('metadata_store')
        event_id_index = current_app.persistent_state.get('event_id_index')
        embedding_model = current_app.persistent_state.get('embedding_model')
        tokenizer = current_app.persistent_state.get('tokenizer')
        llm_model = current_app.persistent_state.get('llm_model')
        ocp_client = current_app.persistent_state.get('ocp_client')
        llm_model_type = current_app.config.get('RAG_CONFIG', {}).get('llm_model_type', 'seq2seq')

        missing_components = []

        # Check each component and log specific error if not loaded
        if not nlu_interpreter:
            missing_components.append('NLU Interpreter')
        if not metadata_store:
            missing_components.append('Metadata Store')
        if not event_id_index:
            missing_components.append('Event ID Index')
        if not embedding_model:
            missing_components.append('Embedding Model')
        if not tokenizer:
            missing_components.append('Tokenizer')
        if not llm_model:
            missing_components.append('LLM Model')
        if not ocp_client:
            missing_components.append('OCP Client')

        # If any component is missing, log the details and notify via Slack
        if missing_components:
            missing_components_str = ', '.join(missing_components)
            logger.error(f"The following components are not loaded: {missing_components_str}")
            slack_handler.send_message(channel, f"Internal error: Missing components - {missing_components_str}")
            return jsonify({"status": "Message sent to Slack"}), 500

        # Parse the user's question with the Rasa NLU model using parse_message
        question = user_text.strip()
        nlu_result = asyncio.run(nlu_interpreter.parse_message(question))

        # Extract the intent and entities from the response
        if nlu_result:
            intent = nlu_result['intent']['name']
            confidence = nlu_result['intent']['confidence']
            entities = {entity['entity']: entity['value'] for entity in nlu_result['entities']}
            entities['text'] = question
        else:
            logger.error("No response received from the NLU model")
            slack_handler.send_message(channel, "Internal error: NLU model did not return a valid response.")
            return jsonify({"status": "Message sent to Slack"}), 500

        response_text = ""

        # Define a confidence threshold
        CONFIDENCE_THRESHOLD = 0.4

        if confidence < CONFIDENCE_THRESHOLD:
            # Use LLM for low confidence intents
            input_text = f"User asked: {question}\nPlease provide a helpful and accurate response."
            response_text = generate_response(
                input_text,
                tokenizer,
                llm_model,
                llm_model_type,
                max_context_length=512,
                max_answer_length=150
            )
        else:
            # Handle the intent using the intent handlers
            handler = INTENT_HANDLERS.get(intent, None)
            if handler:
                if intent in ['fallback', 'general_question']:
                    # Pass necessary components for LLM
                    response_text = handler(
                        entities,
                        tokenizer=tokenizer,
                        llm_model=llm_model,
                        llm_model_type=llm_model_type
                    )
                else:
                    # Pass necessary data to the handler
                    result = handler(
                        entities,
                        event_id_index=event_id_index,
                        metadata_store=metadata_store,
                        ocp_client=ocp_client
                    )

                    # Extract query, output, and final message if returned in the result
                    if isinstance(result, dict):
                        query = result.get("query", "No query executed.")
                        output = result.get("output", [])
                        final_message = result.get("final_message", "No final message.")
                        
                        output_formatted = ""
                        if output:
                            if isinstance(output, list):
                                output_formatted = "\n".join(f"- {item}" for item in output)
                            elif isinstance(output, dict):
                                # Handle dictionary outputs if any
                                output_formatted = "\n".join(f"- {key}: {value}" for key, value in output.items())
                        
                        response_text = ""
                        if query:
                            response_text += f"*Query Executed:*\n{query}\n\n"
                        if output_formatted:
                            response_text += f"*Output:*\n{output_formatted}\n\n"
                        if final_message:
                            response_text += f"*Message:*\n{final_message}"
                    else:
                        response_text = result
            else:
                # Use LLM for unknown intents
                input_text = f"User asked: {question}\nPlease provide a helpful and accurate response."
                response_text = generate_response(
                    input_text,
                    tokenizer,
                    llm_model,
                    llm_model_type,
                    max_context_length=512,
                    max_answer_length=150
                )

        # Send response back to Slack channel
        slack_handler.send_message(channel, response_text)
        return jsonify({"status": "Message sent to Slack"}), 200

    except Exception as e:
        logger.error(f"Error handling Slack event: {e}")
        slack_handler.send_message(channel, "An error occurred while processing your request.")
        return jsonify({"error": "Failed to handle Slack event"}), 500