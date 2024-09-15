# scripts/export_predictive_model.py

import os
import joblib
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

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

def export_model_to_onnx(pipeline, output_path, initial_type):
    """
    Exports the pipeline (preprocessor and model) to ONNX format.

    Args:
        pipeline: The scikit-learn Pipeline object.
        output_path (str): Path to save the ONNX model.
        initial_type (list): List of tuples specifying input feature names and their data types.
    """
    # Convert the sklearn pipeline to ONNX format
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)
    
    # Save the ONNX model to the specified path
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model exported to ONNX format at {output_path}")

def get_feature_details(preprocessor):
    """
    Extracts feature names and their data types from the preprocessor.

    Args:
        preprocessor: The preprocessor object (e.g., ColumnTransformer).

    Returns:
        List of tuples with feature names and their data types.
    """
    feature_details = []
    for transformer in preprocessor.transformers_:
        # Each transformer is a tuple: (name, transformer_object, columns)
        transformer_name, transformer_object, columns = transformer
        for column in columns:
            if column in ['proto', 'service', 'state']:
                feature_details.append((column, StringTensorType([None, 1])))
            else:
                feature_details.append((column, FloatTensorType([None, 1])))
    return feature_details

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

    # Extract feature details for initial_type
    initial_type = get_feature_details(preprocessor)

    # Export to ONNX
    export_model_to_onnx(pipeline, OUTPUT_PATH, initial_type)
