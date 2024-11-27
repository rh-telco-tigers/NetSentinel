# n05_evaluate_model_component.py

import kfp
from kfp import dsl
from kfp.dsl import component, InputPath

@component(
    packages_to_install=['scikit-learn', 'joblib', 'pandas', 'matplotlib', 'seaborn'],
)
def evaluate_model_component(
    model_input_path: InputPath(),
    processed_data_path: InputPath(),
):
    import os
    import joblib
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    # Load model and test data
    model = joblib.load(os.path.join(model_input_path, 'model.joblib'))
    X_test = joblib.load(os.path.join(model_input_path, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(model_input_path, 'y_test.pkl'))

    # Load label encoder
    label_encoder = joblib.load(os.path.join(processed_data_path, 'label_encoder.joblib'))

    # Predict on test data
    y_pred = model.predict(X_test)

    # Decode labels back to original categories
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Save confusion matrix plot
    plt.savefig(os.path.join(model_input_path, 'confusion_matrix.png'))
    plt.close()
    print("Evaluation completed and confusion matrix saved.")
