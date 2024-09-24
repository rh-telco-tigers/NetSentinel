# app/routes.py

from flask import Blueprint, request, jsonify, current_app
import logging
import numpy as np
import re

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# Regular expressions
UUID_REGEX = re.compile(
    r'\b[0-9a-fA-F]{8}-'
    r'[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{12}\b'
)

IP_REGEX = re.compile(
    r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
)

# Helper Functions
def extract_event_id(text):
    match = UUID_REGEX.search(text)
    return match.group(0) if match else None

def extract_ip_addresses(text):
    return IP_REGEX.findall(text)

def get_event_by_id(event_id, event_id_index):
    return event_id_index.get(event_id)

def get_events_by_src_ip(src_ip, metadata_store):
    return [item for item in metadata_store if item.get('src_ip') == src_ip]

def get_events_by_dst_ip(dst_ip, metadata_store):
    return [item for item in metadata_store if item.get('dst_ip') == dst_ip]

def get_recent_attack_events(metadata_store, num_events=5):
    attack_events = [item for item in metadata_store if item.get('prediction') == 1]
    # Assuming metadata_store is ordered by time, newest first
    return attack_events[:num_events]

def get_all_attack_event_ids(metadata_store):
    return [item['event_id'] for item in metadata_store if item.get('prediction') == 1]

def verify_slack_request(signing_secret, request):
    import hmac
    import hashlib
    import time

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

def build_context_from_metadata(indices, metadata_store):
    """
    Build context string from metadata based on indices retrieved from FAISS index.
    """
    context_lines = []
    for idx in indices[0]:
        if idx < len(metadata_store):
            item = metadata_store[idx]
            line = (
                f"Event ID: {item.get('event_id', 'N/A')}, "
                f"Prediction: {'Attack' if item.get('prediction') == 1 else 'Normal'}, "
                f"Protocol: {item.get('protocol', 'N/A')}, "
                f"Source IP: {item.get('src_ip', 'N/A')}, "
                f"Destination IP: {item.get('dst_ip', 'N/A')}"
            )
            context_lines.append(line)
    context = "\n".join(context_lines)
    return context

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

# Route Handlers
@api_bp.route('/')
def health_check():
    return 'OK', 200

@api_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        logger.warning("No input data provided for prediction.")
        return jsonify({"error": "No input data provided"}), 400

    try:
        predictive_model = current_app.persistent_state.get('predictive_model')
        if not predictive_model:
            logger.error("Predictive model is not loaded.")
            return jsonify({"error": "Model is not loaded."}), 500

        required_fields = [
            'proto', 'service', 'state',
            'sbytes', 'dbytes', 'sttl', 'dttl',
            'sloss', 'dloss', 'sload', 'dload',
            'spkts', 'dpkts'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.warning(f"Missing required fields: {', '.join(missing_fields)}")
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        features = [
            data['proto'],
            data['service'],
            data['state'],
            float(data['sbytes']),
            float(data['dbytes']),
            float(data['sttl']),
            float(data['dttl']),
            float(data['sloss']),
            float(data['dloss']),
            float(data['sload']),
            float(data['dload']),
            float(data['spkts']),
            float(data['dpkts'])
        ]

        prediction = predictive_model.predict([features])[0]

        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed."}), 500

@api_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data:
        logger.warning("No question provided for chat.")
        return jsonify({"error": "No question provided."}), 400

    question = data['question']
    try:
        # Access models and data from app's persistent_state
        embedding_model = current_app.persistent_state.get('embedding_model')
        tokenizer = current_app.persistent_state.get('tokenizer')
        llm_model = current_app.persistent_state.get('llm_model')
        faiss_index = current_app.persistent_state.get('faiss_index')
        metadata_store = current_app.persistent_state.get('metadata_store')
        event_id_index = current_app.persistent_state.get('event_id_index')

        if not all([embedding_model, tokenizer, llm_model, faiss_index, metadata_store, event_id_index]):
            logger.error("One or more RAG components are not loaded.")
            return jsonify({"error": "RAG components are not loaded."}), 500

        # Retrieve RAG configuration
        rag_config = current_app.config.get('RAG_CONFIG', {})
        num_contexts = rag_config.get('num_contexts', 5)
        max_context_length = rag_config.get('max_context_length', 512)
        max_answer_length = rag_config.get('max_answer_length', 150)
        llm_model_type = rag_config.get('llm_model_type', 'seq2seq')

        context = ""
        # Check if the user's message contains an event_id
        event_id = extract_event_id(question)
        if event_id:
            # Retrieve the event data directly from event_id_index
            event_data = get_event_by_id(event_id, event_id_index)
            if event_data:
                # Handle specific questions about the event
                if "is this event an attack" in question.lower():
                    prediction = event_data.get('prediction')
                    response_text = "Yes" if prediction == 1 else "No"
                    return jsonify({"response": response_text}), 200
                elif "give me source ip" in question.lower():
                    src_ip = event_data.get('src_ip', 'N/A')
                    return jsonify({"response": f"Source IP: {src_ip}"}), 200
                elif "give me destination ip" in question.lower():
                    dst_ip = event_data.get('dst_ip', 'N/A')
                    return jsonify({"response": f"Destination IP: {dst_ip}"}), 200
                elif "what kind of traffic" in question.lower():
                    protocol = event_data.get('protocol', 'N/A')
                    service = event_data.get('service', 'N/A')
                    return jsonify({"response": f"Protocol: {protocol}, Service: {service}"}), 200
                else:
                    context = build_context_from_event_data(event_data)
            else:
                response_text = "No data found for the provided event ID."
                return jsonify({"response": response_text}), 200
        else:
            # Handle special queries
            if "list down all event id" in question.lower() and "attack" in question.lower():
                event_ids = get_all_attack_event_ids(metadata_store)
                if event_ids:
                    response_text = "Attack Event IDs:\n" + "\n".join(event_ids)
                else:
                    response_text = "No attack events found."
                return jsonify({"response": response_text}), 200
            elif "find all events which is attack type" in question.lower():
                event_ids = get_all_attack_event_ids(metadata_store)
                if event_ids:
                    response_text = "Attack Event IDs:\n" + "\n".join(event_ids)
                else:
                    response_text = "No attack events found."
                return jsonify({"response": response_text}), 200
            else:
                # Use LLM to generate response
                # Build context using FAISS index
                query_embedding = embedding_model.encode(question, convert_to_numpy=True)
                distances, indices = faiss_index.search(
                    np.array([query_embedding]).astype('float32'),
                    k=num_contexts
                )
                # Build context from metadata
                context = build_context_from_metadata(indices, metadata_store)

        # Generate response using the LLM model
        input_text = (
            f"Here is the event data:\n{context}\n\n"
            f"Based on the event data above, please answer the following question:\n{question}\n\n"
            f"Answer:"
        )

        response_text = generate_response(
            input_text,
            tokenizer,
            llm_model,
            llm_model_type,
            max_context_length,
            max_answer_length
        )

        return jsonify({"response": response_text}), 200

    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return jsonify({"error": "Failed to generate response."}), 500

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
    user_text = event.get('text')
    channel = event.get('channel')
    user = event.get('user')

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
        embedding_model = current_app.persistent_state.get('embedding_model')
        tokenizer = current_app.persistent_state.get('tokenizer')
        llm_model = current_app.persistent_state.get('llm_model')
        faiss_index = current_app.persistent_state.get('faiss_index')
        metadata_store = current_app.persistent_state.get('metadata_store')
        event_id_index = current_app.persistent_state.get('event_id_index')

        if not all([embedding_model, tokenizer, llm_model, faiss_index, metadata_store, event_id_index]):
            logger.error("One or more RAG components are not loaded.")
            return jsonify({"error": "Server components are not loaded."}), 500

        # Retrieve RAG configuration
        rag_config = current_app.config.get('RAG_CONFIG', {})
        num_contexts = rag_config.get('num_contexts', 5)
        max_context_length = rag_config.get('max_context_length', 512)
        max_answer_length = rag_config.get('max_answer_length', 150)
        llm_model_type = rag_config.get('llm_model_type', 'seq2seq')

        question = user_text.strip()

        context = ""
        # Check if the user's message contains an event_id
        event_id = extract_event_id(question)
        if event_id:
            # Retrieve the event data directly from event_id_index
            event_data = get_event_by_id(event_id, event_id_index)
            if event_data:
                # Handle specific questions about the event
                if "is this event an attack" in question.lower():
                    prediction = event_data.get('prediction')
                    response_text = "Yes" if prediction == 1 else "No"
                    slack_handler.send_message(channel, response_text)
                    return jsonify({"status": "Message sent to Slack"}), 200
                elif "give me source ip" in question.lower():
                    src_ip = event_data.get('src_ip', 'N/A')
                    slack_handler.send_message(channel, f"Source IP: {src_ip}")
                    return jsonify({"status": "Message sent to Slack"}), 200
                elif "give me destination ip" in question.lower():
                    dst_ip = event_data.get('dst_ip', 'N/A')
                    slack_handler.send_message(channel, f"Destination IP: {dst_ip}")
                    return jsonify({"status": "Message sent to Slack"}), 200
                elif "what kind of traffic" in question.lower():
                    protocol = event_data.get('protocol', 'N/A')
                    service = event_data.get('service', 'N/A')
                    slack_handler.send_message(channel, f"Protocol: {protocol}, Service: {service}")
                    return jsonify({"status": "Message sent to Slack"}), 200
                else:
                    context = build_context_from_event_data(event_data)
            else:
                response_text = "No data found for the provided event ID."
                slack_handler.send_message(channel, response_text)
                return jsonify({"status": "Message sent to Slack"}), 200
        else:
            # Handle special queries
            if "list down all event id" in question.lower() and "attack" in question.lower():
                event_ids = get_all_attack_event_ids(metadata_store)
                if event_ids:
                    response_text = "Attack Event IDs:\n" + "\n".join(event_ids)
                else:
                    response_text = "No attack events found."
                slack_handler.send_message(channel, response_text)
                return jsonify({"status": "Message sent to Slack"}), 200
            elif "find all events which is attack type" in question.lower():
                event_ids = get_all_attack_event_ids(metadata_store)
                if event_ids:
                    response_text = "Attack Event IDs:\n" + "\n".join(event_ids)
                else:
                    response_text = "No attack events found."
                slack_handler.send_message(channel, response_text)
                return jsonify({"status": "Message sent to Slack"}), 200
            else:
                # Use LLM to generate response
                # Build context using FAISS index
                query_embedding = embedding_model.encode(question, convert_to_numpy=True)
                distances, indices = faiss_index.search(
                    np.array([query_embedding]).astype('float32'),
                    k=num_contexts
                )
                # Build context from metadata
                context = build_context_from_metadata(indices, metadata_store)

        # Generate response using the LLM model
        input_text = (
            f"Here is the event data:\n{context}\n\n"
            f"Based on the event data above, please answer the following question:\n{question}\n\n"
            f"Answer:"
        )

        response_text = generate_response(
            input_text,
            tokenizer,
            llm_model,
            llm_model_type,
            max_context_length,
            max_answer_length
        )

        # Send response back to Slack channel
        slack_handler.send_message(channel, response_text)

        return jsonify({"status": "Message sent to Slack"}), 200

    except Exception as e:
        logger.error(f"Error handling Slack event: {e}")
        return jsonify({"error": "Failed to handle Slack event"}), 500
