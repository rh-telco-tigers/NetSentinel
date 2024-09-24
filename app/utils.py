# app/utils.py

import logging
import os
import json
import faiss
import numpy as np

def setup_logging(log_level='INFO', log_file=None):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def load_faiss_index_and_metadata(faiss_index_path, metadata_store_path):
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
    if not os.path.exists(metadata_store_path):
        raise FileNotFoundError(f"Metadata store not found at {metadata_store_path}")

    # Load FAISS index
    faiss_index = faiss.read_index(faiss_index_path)

    # Load metadata store
    with open(metadata_store_path, 'r') as f:
        metadata_store = json.load(f)

    return faiss_index, metadata_store
