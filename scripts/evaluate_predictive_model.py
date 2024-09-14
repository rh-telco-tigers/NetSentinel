# scripts/evaluate_predictive_model.py

import os
import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data(input_dir):
    """
    Loads the processed test data and preprocessor.

    Args:
        input_dir (str): Directory where the processed data is stored.

    Returns:
        X_test, y_test, preprocessor
    """
    X_test = joblib.load(os.path.join(input_dir, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(input_dir, 'y_test.pkl'))
    preprocessor = joblib.load(os.path.join(input_dir, 'preprocessor.pkl'))

    print("Processed test data loaded.")
    return X_test, y_test, preprocessor

def load_model(model_path):
    """
    Loads the trained model from disk.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model: Trained model.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
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

    # Additional Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.close()
    print("ROC curve saved to roc_curve.png")

    # Plot Confusion Matrix
    plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix plot saved to confusion_matrix.png")

if __name__ == "__main__":
    # Paths
    INPUT_DIR = os.path.join('data', 'processed')
    MODEL_PATH = os.path.join('models', 'predictive_model', 'model.joblib')

    # Load processed test data and preprocessor
    X_test, y_test, preprocessor = load_processed_data(INPUT_DIR)

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
