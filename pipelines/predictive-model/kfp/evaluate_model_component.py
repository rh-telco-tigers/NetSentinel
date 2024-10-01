import kfp
from kfp import dsl
from kfp.dsl import component, Input, Dataset, Model

@component(
    packages_to_install=['scikit-learn', 'joblib', 'pandas'],
)
def evaluate_model_component(
    model_input: Input[Model],
    processed_data_path: Input[Dataset],
):
    import os
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    X_test = joblib.load(os.path.join(processed_data_path.path, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(processed_data_path.path, 'y_test.pkl'))
    model = joblib.load(os.path.join(model_input.path, 'model.joblib'))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
