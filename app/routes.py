# app/routes.py

from flask import Blueprint, request, jsonify, current_app
import logging
import os
import hmac
import hashlib
import time

api_bp = Blueprint('api', __name__)

logger = logging.getLogger(__name__)

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
        # Access the model from current_app.persistent_state
        predictive_model = current_app.persistent_state.get('predictive_model')
        if not predictive_model:
            logger.error("Predictive model is not loaded.")
            return jsonify({"error": "Model is not loaded."}), 500

        # Extract features in the correct order
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

        # The model expects a 2D array
        features = [features]

        # Make prediction
        prediction = predictive_model.predict(features)[0]

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
        # Access the LLM model from current_app.persistent_state
        llm_model = current_app.persistent_state.get('llm_model')
        if not llm_model:
            logger.error("LLM model is not loaded.")
            return jsonify({"error": "LLM model is not loaded."}), 500

        response = llm_model.generate_response(question)
        return jsonify({"response": response}), 200
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
    signing_secret = current_app.config['SLACK_CONFIG'].get('slack_signing_secret')
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
    user = event.get('user')

    if not channel:
        logger.warning("No channel specified in Slack event.")
        return jsonify({"error": "No channel specified"}), 400

    # **Prevent the bot from responding to its own messages**
    bot_user_id = current_app.config['SLACK_CONFIG'].get('bot_user_id')
    
    # Print bot user ID for debugging
    logger.info(f"Bot User ID: {bot_user_id}")
    logger.info(f"User ID in the event: {user}")
    
    if user == bot_user_id:
        logger.info("Message is from the bot itself. Ignoring.")
        return jsonify({"status": "Message from bot ignored"}), 200

    try:
        # Access LLM model and Slack client from current_app.persistent_state
        llm_model = current_app.persistent_state.get('llm_model')
        slack_client = current_app.persistent_state.get('slack_client')
        if not llm_model:
            logger.error("LLM model is not loaded.")
            return jsonify({"error": "LLM model is not loaded."}), 500
        if not slack_client:
            logger.error("Slack client is not initialized.")
            return jsonify({"error": "Slack client is not initialized."}), 500

        # Generate response using LLM
        response_text = llm_model.generate_response(user_text)
        # Send response back to Slack channel
        slack_client.send_message(channel, response_text)
        return jsonify({"status": "Message sent to Slack"}), 200
    except Exception as e:
        logger.error(f"Error handling Slack event: {e}")
        return jsonify({"error": "Failed to handle Slack event"}), 500
