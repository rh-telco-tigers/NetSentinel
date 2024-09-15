# app/routes.py

from flask import Blueprint, request, jsonify
from .models import PredictiveModel, LLMModel
from .slack_integration import SlackClient
import logging
import os
import hmac
import hashlib
import time

api_bp = Blueprint('api', __name__)

logger = logging.getLogger(__name__)

# Initialize models and Slack client once
predictive_model = None
llm_model = None
slack_client = None

def initialize_models(app):
    global predictive_model, llm_model, slack_client
    model_config = app.config['MODEL_CONFIG']

    # Initialize Predictive Model
    predictive_model_path = model_config.get('predictive_model_path')
    if not predictive_model_path or not os.path.exists(predictive_model_path):
        logger.error(f"Predictive model path is invalid: {predictive_model_path}")
        raise FileNotFoundError(f"Predictive model not found at {predictive_model_path}")
    predictive_model = PredictiveModel(predictive_model_path)

    # Initialize LLM Model
    llm_model_path = model_config.get('llm_model_path')
    if not llm_model_path or not os.path.exists(llm_model_path):
        logger.error(f"LLM model path is invalid: {llm_model_path}")
        raise FileNotFoundError(f"LLM model not found at {llm_model_path}")
    llm_model = LLMModel(llm_model_path)

    # Initialize Slack Client
    slack_config = app.config['SLACK_CONFIG']
    slack_bot_token = slack_config.get('slack_bot_token')
    if not slack_bot_token:
        logger.error("Slack bot token is not configured.")
        raise ValueError("Slack bot token is missing.")
    slack_client = SlackClient(slack_bot_token)

# @api_bp.before_app_first_request
# def before_first_request():
#     app = api_bp.blueprints['api'].app
#     initialize_models(app)
#     logger.info("Models and Slack client initialized.")

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

@api_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        logger.warning("No input data provided for prediction.")
        return jsonify({"error": "No input data provided"}), 400

    try:
        result = predictive_model.predict(data)
        return jsonify(result)
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
        response = llm_model.generate_response(question)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return jsonify({"error": "Failed to generate response."}), 500

@api_bp.route('/slack/events', methods=['POST'])
def slack_events():
    data = request.get_json()
    if not data:
        logger.warning("No data received from Slack.")
        return jsonify({"error": "No data received"}), 400

    # Verify request
    signing_secret = api_bp.blueprints['api'].app.config['SLACK_CONFIG'].get('slack_signing_secret')
    if not signing_secret:
        logger.error("Slack signing secret is not configured.")
        return jsonify({"error": "Server configuration error."}), 500

    if not verify_slack_request(signing_secret, request):
        return jsonify({"error": "Invalid request signature"}), 403

    # Handle URL Verification challenge
    if 'challenge' in data:
        return jsonify({"challenge": data['challenge']}), 200

    event = data.get('event', {})
    if 'text' not in event:
        logger.warning("No text found in Slack event.")
        return jsonify({"error": "No text found in event"}), 400

    user_text = event['text']
    channel = event.get('channel')

    if not channel:
        logger.warning("No channel specified in Slack event.")
        return jsonify({"error": "No channel specified"}), 400

    try:
        # Generate response using LLM
        response_text = llm_model.generate_response(user_text)
        # Send response back to Slack channel
        slack_client.send_message(channel, response_text)
        return jsonify({"status": "Message sent to Slack"}), 200
    except Exception as e:
        logger.error(f"Error handling Slack event: {e}")
        return jsonify({"error": "Failed to handle Slack event"}), 500
