# app/routes.py

from flask import Blueprint, request, jsonify, current_app
import logging
import asyncio
import hmac
import hashlib
import time
from threading import Lock

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# Import the intent handlers
from .intent_handlers import INTENT_HANDLERS
from .utils import generate_response

# Initialize a set to store processed 'ts' values and a dict to track their timestamps
processed_ts = set()
ts_timestamps = {}
lock = Lock()  # To ensure thread-safe operations on the sets

# Define the time window for keeping 'ts' values (e.g., 1 hour)
TS_EXPIRATION_SECONDS = 3600

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

# Function to clean up old 'ts' entries
def cleanup_processed_ts():
    with lock:
        current_time = time.time()
        expiration_time = current_time - TS_EXPIRATION_SECONDS
        ts_to_remove = [ts for ts, timestamp in ts_timestamps.items() if timestamp < expiration_time]
        for ts in ts_to_remove:
            processed_ts.discard(ts)
            del ts_timestamps[ts]
        if ts_to_remove:
            logger.debug(f"Cleaned up {len(ts_to_remove)} old ts entries.")

# Background thread for periodic cleanup
import threading

def start_cleanup_thread():
    def run_cleanup():
        while True:
            time.sleep(600)  # Cleanup every 10 minutes
            cleanup_processed_ts()

    thread = threading.Thread(target=run_cleanup, daemon=True)
    thread.start()

start_cleanup_thread()


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

    event_type = event.get('type')

    # Only process 'message' events
    if event_type != 'message':
        logger.info(f"Ignoring event type: {event_type}")
        return jsonify({"status": f"Ignored event type: {event_type}"}), 200

    ts = event.get('ts') or event.get('event_ts')
    if not ts:
        logger.warning("No timestamp found in Slack event.")
        return jsonify({"error": "No timestamp found in event"}), 400

    # event_id = event.get('event_id')
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
    
    # Check if the event has already been processed
    with lock:
        if ts in processed_ts:
            logger.info(f"Event with ts {ts} has already been processed. Skipping.")
            return jsonify({"status": "Event already processed"}), 200
        else:
            # Add the ts to the processed set with current timestamp
            processed_ts.add(ts)
            ts_timestamps[ts] = time.time()

    try:
        # Access models and data from app's persistent_state
        nlu_interpreter = current_app.persistent_state.get('nlu_interpreter')
        ocp_client = current_app.persistent_state.get('ocp_client')
        remote_llm_client = current_app.persistent_state.get('remote_llm_client')
        milvus_client = current_app.persistent_state.get('milvus_client')
        collection = milvus_client.collection if milvus_client else None

        # Check for missing components
        missing_components = []

        if not nlu_interpreter:
            missing_components.append('NLU Interpreter')
        if not collection:
            missing_components.append('Milvus Collection')
        if not ocp_client:
            missing_components.append('OCP Client')
        if not remote_llm_client:
            missing_components.append('Remote LLM Client')

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
                remote_llm_client=remote_llm_client,
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
                        remote_llm_client=remote_llm_client
                    )
                else:
                    # Pass necessary data to the handler
                    result = handler(
                        entities,
                        collection=collection,
                        ocp_client=ocp_client,
                        remote_llm_client=remote_llm_client
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
                    remote_llm_client=remote_llm_client,
                    max_answer_length=150
                )

        # Send response back to Slack channel
        slack_handler.send_message(channel, response_text)
        return jsonify({"status": "Message sent to Slack"}), 200

    except Exception as e:
        logger.error(f"Error handling Slack event: {e}")
        slack_handler.send_message(channel, "An error occurred while processing your request.")
        return jsonify({"error": "Failed to handle Slack event"}), 500