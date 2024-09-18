# scripts/prediction_service.py

import json
import yaml
from kafka import KafkaConsumer
import logging
import sys
import signal
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import threading

def setup_logging(config):
    log_level = config.get('logging', {}).get('level', 'INFO').upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger.setLevel(numeric_level)

logger = logging.getLogger(__name__)

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

def load_model(model_path='../models/predictive_model/model.joblib'):
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

def predict(model, data):
    try:
        # Convert data (dict) to numpy array
        feature_values = np.array(list(data['features'].values())).reshape(1, -1)
        # Make prediction
        prediction = model.predict(feature_values)
        # Get prediction probability if needed
        prediction_proba = model.predict_proba(feature_values)
        return prediction[0], prediction_proba[0][1]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, None

def generate_embedding(text, embedding_model):
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return None

def store_in_faiss_index(embedding, metadata, index, metadata_store, lock):
    try:
        with lock:
            # Add the embedding to the index
            index.add(np.array([embedding]).astype('float32'))

            # Add metadata to the metadata store
            metadata_store.append(metadata)

            # Save index and metadata to disk
            faiss.write_index(index, 'faiss_index/index.faiss')
            with open('faiss_index/metadata.json', 'w') as f:
                json.dump(metadata_store, f)
            logger.info(f"Stored event ID {metadata['event_id']} in FAISS index")
    except Exception as e:
        logger.error(f"Error storing data in FAISS index: {e}")

def load_faiss_index(embedding_dimension):
    try:
        index_file = 'faiss_index/index.faiss'
        metadata_file = 'faiss_index/metadata.json'

        if os.path.exists(index_file) and os.path.exists(metadata_file):
            index = faiss.read_index(index_file)
            with open(metadata_file, 'r') as f:
                metadata_store = json.load(f)
            logger.info("FAISS index and metadata loaded from disk")
        else:
            os.makedirs('faiss_index', exist_ok=True)
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
    config = load_config()
    setup_logging(config)
    kafka_bootstrap = config.get('kafka', {}).get('bootstrap', 'localhost:9092')
    processed_topic = config.get('kafka', {}).get('processed_topic', 'processed-traffic-data')
    embedding_model_name = config.get('embedding_model', {}).get('name', 'all-MiniLM-L6-v2')

    consumer = create_kafka_consumer(kafka_bootstrap, processed_topic)
    model = load_model()

    # Load embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    logger.info(f"Embedding model '{embedding_model_name}' loaded")

    # Initialize FAISS index and metadata store
    embedding_dimension = embedding_model.get_sentence_embedding_dimension()
    index, metadata_store = load_faiss_index(embedding_dimension)

    # Lock for thread safety when updating index and metadata
    index_lock = threading.Lock()

    # Register signal handlers
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, consumer))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, consumer))

    logger.info("Starting prediction service...")

    try:
        for message in consumer:
            data = message.value
            logger.debug(f"Consumed processed data: {data}")

            # Extract features and original data
            features = data['features']
            original_data = data['original_data']

            prediction, prediction_proba = predict(model, data)
            if prediction is not None:
                # Enrich data with original information and prediction results
                enriched_data = {
                    'event_id': original_data.get('event_id', None),
                    'src_ip': original_data.get('src_ip', None),
                    'dst_ip': original_data.get('dst_ip', None),
                    'protocol': original_data.get('proto', None),
                    'service': original_data.get('service', None),
                    'state': original_data.get('state', None),
                    'prediction': int(prediction),
                    'prediction_proba': float(prediction_proba),
                    # Include other relevant fields as needed
                }

                # Create a text representation for embedding
                text_representation = (
                    f"Event ID: {enriched_data['event_id']}, "
                    f"Source IP: {enriched_data['src_ip']}, "
                    f"Destination IP: {enriched_data['dst_ip']}, "
                    f"Protocol: {enriched_data['protocol']}, "
                    f"Service: {enriched_data['service']}, "
                    f"State: {enriched_data['state']}, "
                    f"Prediction: {'Attack' if enriched_data['prediction'] == 1 else 'Normal'}"
                )

                # Generate embedding
                embedding = generate_embedding(text_representation, embedding_model)

                if embedding is not None:
                    # Store in FAISS index and metadata store
                    store_in_faiss_index(embedding, enriched_data, index, metadata_store, index_lock)
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
