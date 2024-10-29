import json
import yaml
from kafka import KafkaConsumer, KafkaProducer
import logging
import sys
import signal
import time
import joblib
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

def setup_logging(config):
    """
    Configure logging based on the configuration.
    """
    log_level = config.get('logging', {}).get('level', 'INFO').upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger.setLevel(numeric_level)

def load_config(config_path='config.yaml'):
    """
    Load configuration from the YAML file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def create_kafka_producer(kafka_bootstrap_servers):
    """
    Create and return a Kafka producer using Zeek-like configuration style.
    """
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
            acks="all",
            security_protocol="SASL_SSL",
            sasl_mechanism="SCRAM-SHA-512",
            sasl_plain_username=os.getenv("KAFKA_USERNAME"),
            sasl_plain_password=os.getenv("KAFKA_PASSWORD"),
            ssl_cafile="/usr/local/share/ca-certificates/ca.crt",
            ssl_check_hostname=False 
        )
        logger.info(f"Connected to Kafka at {kafka_bootstrap_servers}")
        return producer
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        sys.exit(1)

def create_kafka_consumer(kafka_bootstrap_servers, topic):
    """
    Create and return a Kafka consumer using Zeek-like configuration style.
    """
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='data_processor_group',
            security_protocol="SASL_SSL",
            sasl_mechanism="SCRAM-SHA-512",
            sasl_plain_username=os.getenv("KAFKA_USERNAME"),
            sasl_plain_password=os.getenv("KAFKA_PASSWORD"),
            ssl_cafile="/usr/local/share/ca-certificates/ca.crt",
            ssl_check_hostname=False 
        )
        logger.info(f"Connected to Kafka topic '{topic}' as consumer")
        return consumer
    except Exception as e:
        logger.error(f"Failed to connect to Kafka as consumer: {e}")
        sys.exit(1)

def load_preprocessor(preprocessor_path):
    """
    Load the preprocessor used during model training.
    """
    try:
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        return preprocessor
    except Exception as e:
        logger.error(f"Failed to load preprocessor: {e}")
        sys.exit(1)

def get_feature_names(preprocessor):
    """
    Extract feature names from the preprocessor.
    """
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            # For categorical features, get feature names from OneHotEncoder
            ohe = transformer
            ohe_feature_names = ohe.get_feature_names_out(columns)
            feature_names.extend(ohe_feature_names)
        elif name == 'num':
            # Numerical features
            feature_names.extend(columns)
    return feature_names

def process_data(raw_data, preprocessor):
    """
    Transform raw data into the format expected by the predictive model.
    """
    try:
        # Convert raw_data (dict) to DataFrame
        df = pd.DataFrame([raw_data])

        # List of relevant columns for processing
        selected_columns = [
            'proto', 'service', 'state', 'sbytes', 'dbytes', 'sttl', 'dttl',
            'sloss', 'dloss', 'sload', 'dload', 'spkts', 'dpkts'
        ]

        # Ensure all selected columns are present in the data
        df = df.reindex(columns=selected_columns, fill_value=np.nan)

        # Handle missing values
        df = handle_missing_values(df, selected_columns)

        # Apply the preprocessor
        X_processed = preprocessor.transform(df[selected_columns])

        # Extract feature names
        feature_names = get_feature_names(preprocessor)

        # Convert processed data to dictionary
        processed_features = dict(zip(feature_names, X_processed[0]))

        # Prepare the final processed data
        processed_data = {
            'features': processed_features,   # Processed features for prediction
            'original_data': raw_data         # Original raw data for reference
        }

        return processed_data
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None

def handle_missing_values(df, selected_columns):
    """
    Handles missing values in the DataFrame.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Fill missing values for categorical columns with 'Unknown'
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # Fill missing values for numerical columns with 0 (or other strategies)
    df[numerical_cols] = df[numerical_cols].fillna(0)

    return df

def signal_handler(sig, frame, consumer, producer):
    """
    Handle termination signals for graceful shutdown.
    """
    logger.info("Shutting down gracefully...")
    consumer.close()
    producer.close()
    sys.exit(0)

def main():
    """
    Main function to run the data processor.
    """
    config = load_config()
    setup_logging(config)

    # Load Kafka and preprocessor configuration
    kafka_bootstrap = config.get('kafka_config', {}).get('bootstrap', 'localhost:9092')
    raw_topic = config.get('kafka_config', {}).get('raw_topic', 'raw-traffic-data')
    processed_topic = config.get('kafka_config', {}).get('processed_topic', 'processed-traffic-data')
    preprocessor_path = config.get('preprocessor', {}).get('path', 'data/processed/preprocessor.pkl')

    # Create Kafka consumer and producer
    consumer = create_kafka_consumer(kafka_bootstrap, raw_topic)
    producer = create_kafka_producer(kafka_bootstrap)

    # Load the preprocessor
    preprocessor = load_preprocessor(preprocessor_path)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, consumer, producer))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, consumer, producer))

    logger.info("Starting data processing...")

    try:
        for message in consumer:
            raw_data = message.value
            logger.debug(f"Consumed raw data: {raw_data}")

            # Process the raw data
            processed_data = process_data(raw_data, preprocessor)
            if processed_data:
                # Publish processed data to the Kafka topic
                producer.send(processed_topic, processed_data)
                producer.flush()
                logger.info(f"Published processed data to '{processed_topic}': {processed_data}")
            else:
                logger.warning("Processed data is None. Skipping.")
    except Exception as e:
        logger.error(f"Error in main processing loop: {e}")
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    main()
