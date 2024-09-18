# app/__init__.py

from flask import Flask
import yaml
import os
import logging
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_client import REGISTRY
from prometheus_flask_exporter import PrometheusMetrics
import requests

# Import your blueprints and utilities
from .routes import api_bp
from .utils import setup_logging
from .models import PredictiveModel, LLMModel
from .slack_integration import SlackClient

logger = logging.getLogger(__name__)

def fetch_and_set_bot_user_id(app):
    slack_bot_token = app.config['SLACK_CONFIG'].get('slack_bot_token')
    if not slack_bot_token:
        raise ValueError("Slack bot token is missing from the configuration.")

    response = requests.post(
        "https://slack.com/api/auth.test",
        headers={"Authorization": f"Bearer {slack_bot_token}"}
    )

    if response.status_code == 200:
        bot_info = response.json()
        if bot_info.get('ok'):
            app.config['SLACK_CONFIG']['bot_user_id'] = bot_info.get('user_id')
            logger.info(f"Bot User ID set to: {bot_info.get('user_id')}")
        else:
            error_msg = bot_info.get('error', 'Unknown error')
            raise ValueError(f"Error fetching bot info: {error_msg}")
    else:
        raise ValueError(f"Failed to fetch bot user ID from Slack API. Status code: {response.status_code}")

def create_app(config_path='config.yaml', registry=None):
    # Configure logging
    setup_logging()

    logger.info("Starting app creation.")

    # Set up Prometheus CollectorRegistry
    if registry is None:
        registry = REGISTRY

    # Unregister existing collectors to prevent duplicates
    collectors_to_unregister = ['app_info']  # List of collectors to unregister
    for collector in collectors_to_unregister:
        if collector in registry._names_to_collectors:
            registry.unregister(registry._names_to_collectors[collector])
            logger.info(f"Unregistered collector: {collector}")

    # Load environment variables from .env
    load_dotenv()

    app = Flask(__name__)

    # Load configuration
    config_file_path = os.path.join(os.path.dirname(__file__), config_path)
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found at {config_file_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_file_path}")

    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Application configurations
    app.config['API_CONFIG'] = config.get('api_config', {})
    app.config['MODEL_CONFIG'] = config.get('model_config', {})
    app.config['SLACK_CONFIG'] = {
        'slack_bot_token': os.getenv('SLACK_BOT_TOKEN', config.get('slack_config', {}).get('slack_bot_token', '')),
        'slack_signing_secret': os.getenv('SLACK_SIGNING_SECRET', config.get('slack_config', {}).get('slack_signing_secret', '')),
        'slack_channel': config.get('slack_config', {}).get('slack_channel', '#netsentenial')
    }
    app.config['TRAINING_CONFIG'] = config.get('training_config', {})

    # Setup logging with configured log level and file
    log_level = app.config['TRAINING_CONFIG'].get('log_level', 'INFO').upper()
    log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'app.log')
    setup_logging(log_level, log_file)
    logger.info("Logging is set up.")

    # Initialize Prometheus Metrics
    metrics = PrometheusMetrics(app, registry=registry)
    metrics.info('app_info', 'NetSentenial Backend API', version='1.0.0')

    # Initialize Rate Limiter
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    limiter.init_app(app)

    # Initialize Models and Slack Client
    try:
        model_config = app.config['MODEL_CONFIG']
        predictive_model_path = model_config.get('predictive_model_path')
        llm_model_name = model_config.get('llm_model_name', 'google/flan-t5-base')
        embedding_model_name = model_config.get('embedding_model_name', 'all-MiniLM-L6-v2')

        # Load the predictive model
        if not predictive_model_path or not os.path.exists(predictive_model_path):
            logger.error(f"Predictive model path is invalid: {predictive_model_path}")
            raise FileNotFoundError(f"Predictive model not found at {predictive_model_path}")

        predictive_model = PredictiveModel(predictive_model_path)

        # Load the embedding model for RAG
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"Embedding model '{embedding_model_name}' loaded.")

        # Load the LLM model for RAG
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
        logger.info(f"LLM model '{llm_model_name}' loaded.")

        # Load FAISS index and metadata
        from .utils import load_faiss_index_and_metadata  # You need to implement this function
        faiss_index, metadata_store = load_faiss_index_and_metadata()

        # Initialize Slack Client
        slack_config = app.config['SLACK_CONFIG']
        slack_bot_token = slack_config.get('slack_bot_token')
        if not slack_bot_token:
            logger.error("Slack bot token is not configured.")
            raise ValueError("Slack bot token is missing.")
        slack_client = SlackClient(slack_bot_token)

        # Fetch and set the bot user ID
        fetch_and_set_bot_user_id(app)

        # Attach models and clients to app for access in routes
        app.persistent_state = {
            'predictive_model': predictive_model,
            'embedding_model': embedding_model,
            'tokenizer': tokenizer,
            'llm_model': llm_model,
            'faiss_index': faiss_index,
            'metadata_store': metadata_store,
            'slack_client': slack_client
        }

        logger.info("Models, FAISS index, and Slack client initialized.")
        logger.info("Persistent state set with components: %s", list(app.persistent_state.keys()))


    except Exception as e:
        logger.error(f"Failed to initialize models or Slack client: {e}")
        raise e

    # Register Blueprints
    app.register_blueprint(api_bp)

    return app
