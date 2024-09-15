# app/__init__.py

from flask import Flask
import yaml
import os
from .routes import api_bp
from .utils import setup_logging
import logging
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_client import REGISTRY, CollectorRegistry
from prometheus_flask_exporter import PrometheusMetrics
from .models import PredictiveModel, LLMModel
from .slack_integration import SlackClient

def create_app(config_path='config.yaml', registry=None):
    # Configure logging
    setup_logging()

    logger = logging.getLogger(__name__)
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

    app.config['API_CONFIG'] = config.get('api_config', {})
    app.config['MODEL_CONFIG'] = config.get('model_config', {})
    app.config['SLACK_CONFIG'] = {
        'slack_bot_token': os.getenv('SLACK_BOT_TOKEN', config.get('slack_config', {}).get('slack_bot_token', '')),
        'slack_signing_secret': os.getenv('SLACK_SIGNING_SECRET', config.get('slack_config', {}).get('slack_signing_secret', '')),
        'slack_channel': config.get('slack_config', {}).get('slack_channel', '#general')
    }
    app.config['TRAINING_CONFIG'] = config.get('training_config', {})

    # Setup logging
    log_level = app.config['TRAINING_CONFIG'].get('log_level', 'INFO').upper()
    log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'app.log')
    setup_logging(log_level, log_file)
    logger = logging.getLogger(__name__)
    logger.info("Logging is set up.")

    # Initialize Prometheus Metrics
    metrics = PrometheusMetrics(app, registry=registry)
    metrics.info('app_info', 'NetSentenial Backend API', version='1.0.0')

    # Initialize Limiter
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    limiter.init_app(app)

    # Initialize Models and Slack Client
    try:
        model_config = app.config['MODEL_CONFIG']
        predictive_model_path = model_config.get('predictive_model_path')
        llm_model_path = model_config.get('llm_model_path')

        if not predictive_model_path or not os.path.exists(predictive_model_path):
            logger.error(f"Predictive model path is invalid: {predictive_model_path}")
            raise FileNotFoundError(f"Predictive model not found at {predictive_model_path}")

        if not llm_model_path or not os.path.exists(llm_model_path):
            logger.error(f"LLM model path is invalid: {llm_model_path}")
            raise FileNotFoundError(f"LLM model not found at {llm_model_path}")

        predictive_model = PredictiveModel(predictive_model_path)
        llm_model = LLMModel(llm_model_path)

        slack_config = app.config['SLACK_CONFIG']
        slack_bot_token = slack_config.get('slack_bot_token')
        if not slack_bot_token:
            logger.error("Slack bot token is not configured.")
            raise ValueError("Slack bot token is missing.")
        slack_client = SlackClient(slack_bot_token)

        # Attach models and slack_client to app for access in routes
        app.persistent_state = {
            'predictive_model': predictive_model,
            'llm_model': llm_model,
            'slack_client': slack_client
        }

        logger.info("Models and Slack client initialized.")

    except Exception as e:
        logger.error(f"Failed to initialize models or Slack client: {e}")
        raise e

    # Register Blueprints
    app.register_blueprint(api_bp)

    return app
