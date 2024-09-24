# app/routes.py

from flask import Blueprint, request, jsonify, current_app
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)


# Helper Functions
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
        # Access models and FAISS index from app's persistent_state
        embedding_model = current_app.persistent_state.get('embedding_model')
        tokenizer = current_app.persistent_state.get('tokenizer')
        llm_model = current_app.persistent_state.get('llm_model')
        faiss_index = current_app.persistent_state.get('faiss_index')
        metadata_store = current_app.persistent_state.get('metadata_store')

        if not all([embedding_model, tokenizer, llm_model, faiss_index, metadata_store]):
            logger.error("One or more RAG components are not loaded.")
            return jsonify({"error": "RAG components are not loaded."}), 500

        # Retrieve RAG configuration
        rag_config = current_app.config.get('RAG_CONFIG', {})
        num_contexts = rag_config.get('num_contexts', 5)
        max_context_length = rag_config.get('max_context_length', 512)
        max_answer_length = rag_config.get('max_answer_length', 150)
        llm_model_type = rag_config.get('llm_model_type', 'seq2seq')

        # Retrieve relevant data using the embedding model and FAISS index
        query_embedding = embedding_model.encode(question, convert_to_numpy=True)
        distances, indices = faiss_index.search(
            np.array([query_embedding]).astype('float32'),
            k=num_contexts
        )

        # Build context from metadata
        context = build_context_from_metadata(indices, metadata_store)

        # Generate response using the LLM model
        input_text = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
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
        # Access models and FAISS index from app's persistent_state
        embedding_model = current_app.persistent_state.get('embedding_model')
        tokenizer = current_app.persistent_state.get('tokenizer')
        llm_model = current_app.persistent_state.get('llm_model')
        faiss_index = current_app.persistent_state.get('faiss_index')
        metadata_store = current_app.persistent_state.get('metadata_store')

        if not all([embedding_model, tokenizer, llm_model, faiss_index, metadata_store]):
            logger.error("One or more RAG components are not loaded.")
            return jsonify({"error": "Server components are not loaded."}), 500

        # Retrieve RAG configuration
        rag_config = current_app.config.get('RAG_CONFIG', {})
        num_contexts = rag_config.get('num_contexts', 5)
        max_context_length = rag_config.get('max_context_length', 512)
        max_answer_length = rag_config.get('max_answer_length', 150)
        llm_model_type = rag_config.get('llm_model_type', 'seq2seq')

        # Retrieve relevant data using the embedding model and FAISS index
        query_embedding = embedding_model.encode(user_text, convert_to_numpy=True)
        distances, indices = faiss_index.search(
            np.array([query_embedding]).astype('float32'),
            k=num_contexts
        )

        # Build context from metadata
        context = build_context_from_metadata(indices, metadata_store)

        # Generate response using the LLM model
        input_text = f"Context:\n{context}\n\nQuestion:\n{user_text}\n\nAnswer:"
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
