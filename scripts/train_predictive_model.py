# scripts/train_predictive_model.py

import os
import sys
import joblib
import yaml
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
import numpy as np

def setup_logging(log_level, log_file=None):
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_processed_data(input_dir):
    """
    Loads the processed data and preprocessor.

    Args:
        input_dir (str): Directory where processed data is stored.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    try:
        X_train = joblib.load(os.path.join(input_dir, 'X_train.pkl'))
        X_test = joblib.load(os.path.join(input_dir, 'X_test.pkl'))
        y_train = joblib.load(os.path.join(input_dir, 'y_train.pkl'))
        y_test = joblib.load(os.path.join(input_dir, 'y_test.pkl'))
        preprocessor = joblib.load(os.path.join(input_dir, 'preprocessor.pkl'))

        logging.info("Processed data loaded successfully.")
        return X_train, X_test, y_train, y_test, preprocessor
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        sys.exit(1)

def train_model(X_train, y_train, n_estimators, random_state, n_jobs):
    """
    Trains a Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_estimators: Number of trees in the forest.
        random_state: Seed for reproducibility.
        n_jobs: Number of jobs to run in parallel.

    Returns:
        model: Trained model.
    """
    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )
        model.fit(X_train, y_train)
        logging.info("Model training completed successfully.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        sys.exit(1)

def evaluate_model(model, X_test, y_test, evaluation_config):
    """
    Evaluates the trained model on the test set.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        evaluation_config: Dictionary with evaluation settings.
    """
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        if evaluation_config.get('enable_classification_report', False):
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

        if evaluation_config.get('enable_confusion_matrix', False):
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

        if evaluation_config.get('enable_roc_auc', False):
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"ROC AUC Score: {roc_auc:.4f}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        sys.exit(1)

def save_model(model, output_dir, model_filename):
    """
    Saves the trained model to disk.

    Args:
        model: Trained model.
        output_dir (str): Directory to save the model.
        model_filename (str): Filename for the saved model.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, model_filename)
        joblib.dump(model, model_path)
        logging.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logging.error(f"Error saving the model: {e}")
        sys.exit(1)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Predictive Model using Random Forest.")
    parser.add_argument(
        '--config_file',
        type=str,
        default='config.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)

    # Setup logging
    setup_logging(config['logging_config']['level'], None)
    logger = logging.getLogger(__name__)

    logger.info("Starting Predictive Model Training Script.")
    logger.debug(f"Configuration Loaded: {config}")

    # Load processed data
    X_train, X_test, y_train, y_test, preprocessor = load_processed_data(config['predictive_model_config']['input_dir'])

    # Train the model
    model = train_model(
        X_train,
        y_train,
        n_estimators=config['predictive_model_config']['n_estimators'],
        random_state=config['predictive_model_config']['random_state'],
        n_jobs=config['predictive_model_config']['n_jobs']
    )

    # Evaluate the model
    evaluate_model(
        model,
        X_test,
        y_test,
        config['predictive_model_config']['evaluation']
    )

    # Save the model
    save_model(
        model,
        config['predictive_model_config']['model_dir'],
        config['predictive_model_config']['model_filename']
    )

if __name__ == "__main__":
    main()
