import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    packages_to_install=['scikit-learn', 'joblib', 'pandas'],
)
def train_model_component(
    processed_data_path: Input[Dataset],
    model_output: Output[Model],
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
):
    import os
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    X_train = joblib.load(os.path.join(processed_data_path.path, 'X_train.pkl'))
    y_train = joblib.load(os.path.join(processed_data_path.path, 'y_train.pkl'))

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    print("Model training completed successfully.")

    os.makedirs(model_output.path, exist_ok=True)
    joblib.dump(model, os.path.join(model_output.path, 'model.joblib'))
    print(f"Model saved at {model_output.path}")
