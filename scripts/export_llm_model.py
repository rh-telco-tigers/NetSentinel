# scripts/export_predictive_model.py

import os
import joblib
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

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

def load_preprocessor(preprocessor_path):
    """
    Loads the preprocessor from disk.

    Args:
        preprocessor_path (str): Path to the saved preprocessor file.

    Returns:
        preprocessor: Fitted preprocessor.
    """
    preprocessor = joblib.load(preprocessor_path)
    print(f"Preprocessor loaded from {preprocessor_path}")
    return preprocessor

def export_model_to_onnx(pipeline, output_path):
    """
    Exports the pipeline (preprocessor and model) to ONNX format.

    Args:
        pipeline: The scikit-learn Pipeline object.
        output_path (str): Path to save the ONNX model.
    """
    # Get the number of features before preprocessing
    num_features = len(pipeline.named_steps['preprocessor'].transformers_[0][2]) + len(pipeline.named_steps['preprocessor'].transformers_[1][2])

    # Define initial types
    initial_type = [('float_input', FloatTensorType([None, num_features]))]

    # Convert to ONNX
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model exported to ONNX format at {output_path}")

if __name__ == "__main__":
    # Paths
    MODEL_PATH = os.path.join('models', 'predictive_model', 'model.joblib')
    PREPROCESSOR_PATH = os.path.join('data', 'processed', 'preprocessor.pkl')
    OUTPUT_PATH = os.path.join('models', 'predictive_model', 'model.onnx')

    # Load model and preprocessor
    model = load_model(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)

    # Create a pipeline with preprocessor and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Export to ONNX
    export_model_to_onnx(pipeline, OUTPUT_PATH)
