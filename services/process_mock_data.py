# process_mock_data.py

import json
import yaml
from kafka import KafkaConsumer, KafkaProducer
import logging
import sys
import signal
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

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
    Create and return a Kafka producer.
    """
    try:
        use_ssl = os.path.exists("/usr/local/share/ca-certificates/ca.crt")

        producer_args = {
            "bootstrap_servers": kafka_bootstrap_servers,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "retries": 5,
            "acks": "all",
            "security_protocol": "SASL_SSL" if use_ssl else "SASL_PLAINTEXT",
            "sasl_mechanism": "SCRAM-SHA-512" if use_ssl else "PLAIN",
            "sasl_plain_username": os.getenv("KAFKA_USERNAME", "admin"),
            "sasl_plain_password": os.getenv("KAFKA_PASSWORD", "secret-password"),
        }

        if use_ssl:
            producer_args["ssl_cafile"] = "/usr/local/share/ca-certificates/ca.crt"
            producer_args["ssl_check_hostname"] = False

        producer = KafkaProducer(**producer_args)
        logger.info(f"Connected to Kafka at {kafka_bootstrap_servers}")
        return producer
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        sys.exit(1)

def create_kafka_consumer(kafka_bootstrap_servers, topic):
    """
    Create and return a Kafka consumer.
    """
    try:
        use_ssl = os.path.exists("/usr/local/share/ca-certificates/ca.crt")

        consumer_args = {
            "bootstrap_servers": kafka_bootstrap_servers,
            "value_deserializer": lambda v: json.loads(v.decode('utf-8')),
            "auto_offset_reset": 'latest',
            "enable_auto_commit": True,
            "group_id": 'data_processor_group',
            "security_protocol": "SASL_SSL" if use_ssl else "SASL_PLAINTEXT",
            "sasl_mechanism": "SCRAM-SHA-512" if use_ssl else "PLAIN",
            "sasl_plain_username": os.getenv("KAFKA_USERNAME", "admin"),
            "sasl_plain_password": os.getenv("KAFKA_PASSWORD", "secret-password"),
        }

        if use_ssl:
            consumer_args["ssl_cafile"] = "/usr/local/share/ca-certificates/ca.crt"
            consumer_args["ssl_check_hostname"] = False

        consumer = KafkaConsumer(topic, **consumer_args)
        logger.info(f"Connected to Kafka topic '{topic}' as consumer")
        return consumer
    except Exception as e:
        logger.error(f"Failed to connect to Kafka as consumer: {e}")
        sys.exit(1)

def load_encoders_and_scaler(encoders_path):
    """
    Load ordinal encoder and scaler used during model training.
    """
    try:
        ordinal_encoder = joblib.load(os.path.join(encoders_path, 'ordinal_encoder.joblib'))
        scaler = joblib.load(os.path.join(encoders_path, 'scaler.joblib'))
        logger.info(f"Encoder and scaler loaded from {encoders_path}")
        return ordinal_encoder, scaler
    except Exception as e:
        logger.error(f"Failed to load encoder and scaler: {e}")
        sys.exit(1)

def process_data(raw_data, ordinal_encoder, scaler):
    try:
        # Convert raw_data (dict) to DataFrame
        df = pd.DataFrame([raw_data])

        # Define all feature columns (excluding columns to be excluded)
        columns_to_exclude = ['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', 'Label', 'attack_cat', 'event_id']

        # Define all columns that will be used for prediction
        feature_columns = [col for col in df.columns if col not in columns_to_exclude]

        # Ensure all selected columns are present in the data
        df = df.reindex(columns=feature_columns, fill_value=np.nan)

        # Log the columns and first row
        logger.debug(f"Columns in DataFrame before encoding: {df.columns.tolist()}")
        logger.debug(f"First row of data: {df.iloc[0].to_dict()}")

        # Handle missing values
        df = handle_missing_values(df, feature_columns)

        # Identify categorical columns (excluding columns to be excluded)
        categorical_cols = [
            'proto', 'state', 'service',
            'ct_ftp_cmd',  # If treated as categorical during training
            # Add any other categorical columns used during training
        ]

        # Ensure categorical columns are of type string
        df[categorical_cols] = df[categorical_cols].astype(str)

        # Encode categorical variables
        df[categorical_cols] = ordinal_encoder.transform(df[categorical_cols])

        # Scale numerical features
        numerical_cols = [col for col in feature_columns if col not in categorical_cols]
        df[numerical_cols] = df[numerical_cols].astype(float)
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        # Convert processed data to dictionary
        processed_features = df.iloc[0].to_dict()

        # Prepare the final processed data
        processed_data = {
            'features': processed_features,
            'original_data': raw_data  # Keep the original data including excluded columns
        }

        return processed_data
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None

def handle_missing_values(df, selected_columns):
    # The same logic applies, just use selected_columns
    categorical_cols = [
        'proto', 'state', 'service',
        'ct_ftp_cmd',  # If treated as categorical during training
        # Add any other categorical columns used during training
    ]
    numerical_cols = [col for col in selected_columns if col not in categorical_cols]

    # Fill missing values for categorical columns with 'Unknown'
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # Fill missing values for numerical columns with 0
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

    kafka_bootstrap = config.get('kafka_config', {}).get('bootstrap', 'localhost:9092')
    raw_topic = config.get('kafka_config', {}).get('raw_topic', 'raw-traffic-data')
    processed_topic = config.get('kafka_config', {}).get('processed_topic', 'processed-traffic-data')
    encoders_path = config.get('preprocessor', {}).get('encoders_path', 'models/encoders')

    # Create Kafka consumer and producer
    consumer = create_kafka_consumer(kafka_bootstrap, raw_topic)
    producer = create_kafka_producer(kafka_bootstrap)

    # Load ordinal encoder and scaler
    ordinal_encoder, scaler = load_encoders_and_scaler(encoders_path)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, consumer, producer))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, consumer, producer))

    logger.info("Starting data processing...")

    try:
        for message in consumer:
            raw_data = message.value
            logger.debug(f"Consumed raw data: {raw_data}")

            # Process the raw data
            processed_data = process_data(raw_data, ordinal_encoder, scaler)
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
