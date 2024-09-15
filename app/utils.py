# app/utils.py

import logging

def setup_logging(log_level='INFO', log_file=None):
    logger = logging.getLogger()
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
