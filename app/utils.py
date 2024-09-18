# app/utils.py

import logging
import os
import faiss
import json

logger = logging.getLogger()

def setup_logging(log_level='INFO', log_file=None):
    
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level, logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level, logging.INFO))
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def load_faiss_index_and_metadata(index_file='faiss_index/index.faiss', metadata_file='faiss_index/metadata.json'):
    try:
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            index = faiss.read_index(index_file)
            with open(metadata_file, 'r') as f:
                metadata_store = json.load(f)
            logger.info("FAISS index and metadata store loaded successfully.")
            return index, metadata_store
        else:
            logger.error(f"FAISS index or metadata file not found at '{index_file}' or '{metadata_file}'.")
            return None, None
    except Exception as e:
        logger.error(f"Error loading FAISS index and metadata: {e}", exc_info=True)
        return None, None