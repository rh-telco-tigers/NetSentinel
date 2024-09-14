# scripts/train_predictive_model.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

def load_processed_data(input_dir):
    """
    Loads the processed data and preprocessor.

    Args:
        input_dir (str): Directory where processed data is stored.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    X_train = joblib.load(os.path.join(input_dir, 'X_train.pkl'))
    X_test = joblib.load(os.path.join(input_dir, 'X_test.pkl'))
    y_train = joblib.load(os.path.join(input_dir, 'y_train.pkl'))
    y_test = joblib.load(os.path.join(input_dir, 'y_test.pkl'))
    preprocessor = joblib.load(os.path.join(input_dir, 'preprocessor.pkl'))

    print("Processed data loaded.")
    return X_train, X_test, y_train, y_test, preprocessor

def train_model(X_train, y_train):
    """
    Trains a Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        model: Trained model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Model Evaluation Metrics:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

def save_model(model, output_dir):
    """
    Saves the trained model to disk.

    Args:
        model: Trained model.
        output_dir (str): Directory to save the model.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Paths
    INPUT_DIR = os.path.join('data', 'processed')
    MODEL_DIR = os.path.join('models', 'predictive_model')

    # Load processed data
    X_train, X_test, y_train, y_test, preprocessor = load_processed_data(INPUT_DIR)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, MODEL_DIR)
