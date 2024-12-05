# predict_and_store.py

import json
import yaml
from kafka import KafkaConsumer
import logging
import sys
import signal
import os
from datetime import datetime
from remote_predictive_model_client import RemotePredictiveModelClient
from milvus_client import MilvusClient

logger = logging.getLogger(__name__)

def setup_logging(config):
    log_level = config.get('logging', {}).get('level', 'INFO').upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger.setLevel(numeric_level)

def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def create_kafka_consumer(kafka_bootstrap_servers, topic):
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

def signal_handler(sig, frame, consumer, milvus_client):
    logger.info("Shutting down gracefully...")
    consumer.close()
    milvus_client.close()
    sys.exit(0)

def main():
    config = load_config()
    setup_logging(config)

    kafka_bootstrap = config.get('kafka', {}).get('bootstrap_servers', 'localhost:9092')
    processed_topic = config.get('kafka', {}).get('topics', {}).get('processed', 'processed-traffic-data')

    # Load the remote predictive model client
    models_config = config.get('models', {})
    predictive_model_config = models_config.get('predictive', {})
    model_url = predictive_model_config.get('url')
    model_token = predictive_model_config.get('token')
    verify_ssl = predictive_model_config.get('verify_ssl', True)

    remote_model_client = RemotePredictiveModelClient(
        url=model_url,
        token=model_token,
        verify_ssl=verify_ssl
    )

    # Initialize Milvus client
    milvus_config = config.get('milvus', {})
    milvus_host = milvus_config.get('host', 'localhost')
    milvus_port = milvus_config.get('port', '19530')
    collection_name = milvus_config.get('collection_name', 'my_collection')
    milvus_secure = milvus_config.get('secure', False)

    milvus_client = MilvusClient(
        host=milvus_host,
        port=milvus_port,
        collection_name=collection_name,
        embedding_dim=6,
        secure=milvus_secure
    )

    # Create Kafka consumer
    consumer = create_kafka_consumer(kafka_bootstrap, processed_topic)

    # Register signal handlers
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, consumer, milvus_client))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, consumer, milvus_client))

    logger.info("Starting prediction and storage service...")

    try:
        for message in consumer:
            data = message.value
            logger.debug(f"Consumed processed data: {data}")

            features_dict = data.get('features', {})
            original_data = data.get('original_data', {})

            # Prepare features as a list of float values in the correct order
            features = prepare_features(features_dict)

            if features is None:
                logger.warning("Failed to prepare features. Skipping.")
                continue

            prediction, probabilities = remote_model_client.predict(features)
            if prediction is not None:
                # Handle timestamp
                timestamp_str = get_timestamp(original_data)

                enriched_data = {
                    'event_id': original_data.get('event_id', None),
                    'src_ip': original_data.get('srcip', None),
                    'dst_ip': original_data.get('dstip', None),
                    'protocol': original_data.get('proto', None),
                    'service': original_data.get('service', None),
                    'state': original_data.get('state', None),
                    'prediction': int(prediction),
                    'probabilities': probabilities,  # Store full probabilities
                    'timestamp': timestamp_str
                }

                # Use the probability of the predicted class
                try:
                    predicted_prob = float(probabilities[int(prediction)])
                except (IndexError, ValueError, TypeError):
                    predicted_prob = 0.0
                    logger.warning("Invalid prediction probability.")

                vector = [
                    predicted_prob,
                    float(prediction),
                    float(original_data.get('sbytes', 0)),
                    float(original_data.get('dbytes', 0)),
                    float(original_data.get('Spkts', 0)),
                    float(original_data.get('Dpkts', 0))
                ]

                store_in_milvus(vector, enriched_data, milvus_client)
            else:
                logger.warning("Prediction failed. Skipping.")
    except Exception as e:
        logger.error(f"Error in prediction loop: {e}")
    finally:
        consumer.close()
        milvus_client.close()

def prepare_features(features_dict):
    # Define the expected feature names in order
    feature_names = [
        'proto', 'state', 'dur', 'sbytes', 'dbytes',
        'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
        'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
        'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm'
    ]

    features = []
    try:
        for name in feature_names:
            value = features_dict.get(name, 0.0)
            features.append(float(value))
        return features
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return None

def get_timestamp(original_data):
    incoming_timestamp = original_data.get('Stime')
    if incoming_timestamp:
        try:
            # Validate and parse incoming timestamp
            parsed_timestamp = datetime.fromisoformat(incoming_timestamp)
            return parsed_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            logger.warning(f"Invalid timestamp format: {incoming_timestamp}. Using current time.")
    # Generate current timestamp
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def store_in_milvus(vector, metadata, milvus_client):
    try:
        vectors = [vector]  # List of vectors (each vector is a list of floats)
        metadatas = [metadata]
        milvus_client.insert(vectors, metadatas)
        logger.info(f"Stored event ID {metadata['event_id']} in Milvus collection")
    except Exception as e:
        logger.error(f"Error storing data in Milvus: {e}")

if __name__ == "__main__":
    main()
