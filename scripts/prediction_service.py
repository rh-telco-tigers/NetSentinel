import json
import yaml
from kafka import KafkaConsumer
import logging
import sys
import signal
import joblib
import numpy as np
import os
import threading
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime

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
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='prediction_service_group_v2'
        )
        logger.info(f"Connected to Kafka topic '{topic}' as consumer")
        return consumer
    except Exception as e:
        logger.error(f"Failed to connect to Kafka as consumer: {e}")
        sys.exit(1)

def load_model(model_dir, model_filename='model.joblib'):
    model_path = os.path.join(model_dir, model_filename)
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

def predict(model, data):
    try:
        feature_values = np.array(list(data['features'].values())).reshape(1, -1)
        prediction = model.predict(feature_values)
        prediction_proba = model.predict_proba(feature_values)
        return prediction[0], prediction_proba[0][1]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, None

def generate_embedding(text, embedding_model):
    try:
        return embedding_model.encode(text, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return None

def store_in_faiss_index(embedding, metadata, index, metadata_store, lock, faiss_index_path, metadata_store_path):
    try:
        with lock:
            index.add(np.array([embedding]).astype('float32'))
            metadata_store.append(metadata)
            # Ensure directories exist before saving
            os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
            os.makedirs(os.path.dirname(metadata_store_path), exist_ok=True)
            # Save the FAISS index
            faiss.write_index(index, faiss_index_path)
            # Save the metadata store
            with open(metadata_store_path, 'w') as f:
                json.dump(metadata_store, f)
            logger.info(f"Stored event ID {metadata['event_id']} in FAISS index")
    except Exception as e:
        logger.error(f"Error storing data in FAISS index: {e}")

def load_faiss_index(embedding_dimension, faiss_index_path, metadata_store_path):
    try:
        if os.path.exists(faiss_index_path) and os.path.exists(metadata_store_path):
            index = faiss.read_index(faiss_index_path)
            with open(metadata_store_path, 'r') as f:
                metadata_store = json.load(f)
            logger.info("FAISS index and metadata loaded from disk")
        else:
            # Ensure directories exist
            os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
            index = faiss.IndexFlatL2(embedding_dimension)
            metadata_store = []
            logger.info("Initialized new FAISS index and metadata store")
        return index, metadata_store
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        sys.exit(1)

def signal_handler(sig, frame, consumer):
    logger.info("Shutting down gracefully...")
    consumer.close()
    sys.exit(0)

def main():
    try:
        config = load_config()
        setup_logging(config)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return

    kafka_bootstrap = config.get('kafka_config', {}).get('bootstrap', 'localhost:9092')
    processed_topic = config.get('kafka_config', {}).get('processed_topic', 'processed-traffic-data')

    # RAG configuration
    rag_config = config.get('rag_config', {})
    embedding_model_name = rag_config.get('embedding_model_name', 'all-MiniLM-L6-v2')
    faiss_index_path = rag_config.get('faiss_index_path', os.path.join('data', 'faiss_index', 'index.faiss'))
    metadata_store_path = rag_config.get('metadata_store_path', os.path.join('data', 'faiss_index', 'metadata.json'))

    # Ensure directories exist
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_store_path), exist_ok=True)

    # Load Kafka consumer and models
    consumer = create_kafka_consumer(kafka_bootstrap, processed_topic)

    model_dir = config.get('predictive_model_config', {}).get('model_dir', os.path.join('models', 'predictive_model'))
    model_filename = config.get('predictive_model_config', {}).get('model_filename', 'model.joblib')
    model = load_model(model_dir, model_filename)

    embedding_model = SentenceTransformer(embedding_model_name)
    logger.info(f"Embedding model '{embedding_model_name}' loaded")

    embedding_dimension = embedding_model.get_sentence_embedding_dimension()
    index, metadata_store = load_faiss_index(embedding_dimension, faiss_index_path, metadata_store_path)

    index_lock = threading.Lock()

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, consumer))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, consumer))

    logger.info("Starting prediction service...")

    try:
        for message in consumer:
            data = message.value
            logger.debug(f"Consumed processed data: {data}")

            features = data['features']
            original_data = data['original_data']

            prediction, prediction_proba = predict(model, data)
            if prediction is not None:
                # Extract or generate timestamp
                incoming_timestamp = original_data.get('timestamp')  # Adjust based on your data structure
                if incoming_timestamp:
                    try:
                        # Validate and parse incoming timestamp
                        parsed_timestamp = datetime.strptime(incoming_timestamp, "%Y-%m-%d %H:%M:%S")
                        timestamp_str = parsed_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        logger.warning(f"Invalid timestamp format: {incoming_timestamp}. Using current time.")
                        timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # Generate current timestamp
                    timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

                enriched_data = {
                    'event_id': original_data.get('event_id', None),
                    'src_ip': original_data.get('src_ip', None),
                    'dst_ip': original_data.get('dst_ip', None),
                    'protocol': original_data.get('proto', None),
                    'service': original_data.get('service', None),
                    'state': original_data.get('state', None),
                    'prediction': int(prediction),
                    'prediction_proba': float(prediction_proba),
                    'timestamp': timestamp_str
                }

                text_representation = (
                    f"Event ID: {enriched_data['event_id']}, "
                    f"Source IP: {enriched_data['src_ip']}, "
                    f"Destination IP: {enriched_data['dst_ip']}, "
                    f"Protocol: {enriched_data['protocol']}, "
                    f"Service: {enriched_data['service']}, "
                    f"State: {enriched_data['state']}, "
                    f"Prediction: {'Attack' if enriched_data['prediction'] == 1 else 'Normal'}, "
                    f"Timestamp: {enriched_data['timestamp']}"
                )

                embedding = generate_embedding(text_representation, embedding_model)
                if embedding is not None:
                    store_in_faiss_index(
                        embedding, enriched_data, index, metadata_store,
                        index_lock, faiss_index_path, metadata_store_path
                    )
                else:
                    logger.warning("Embedding generation failed. Skipping storage.")
            else:
                logger.warning("Prediction failed. Skipping.")
    except Exception as e:
        logger.error(f"Error in prediction loop: {e}")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
