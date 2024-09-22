# scripts/export_predictive_model.py

import os
import joblib
import yaml
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def load_preprocessor(preprocessor_path):
    preprocessor = joblib.load(preprocessor_path)
    print(f"Preprocessor loaded from {preprocessor_path}")
    return preprocessor

def export_model_to_onnx(pipeline, output_path, initial_type):
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model exported to ONNX format at {output_path}")

def get_feature_details(preprocessor):
    feature_details = []
    # Assuming the preprocessor is a ColumnTransformer
    for transformer in preprocessor.transformers_:
        transformer_name, transformer_object, columns = transformer
        for column in columns:
            if column in ['proto', 'service', 'state']:
                feature_details.append((column, StringTensorType([None, 1])))
            else:
                feature_details.append((column, FloatTensorType([None, 1])))
    return feature_details

if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Get predictive model config
    predictive_model_config = config.get('predictive_model_config', {})
    model_dir = predictive_model_config.get('model_dir', 'models/predictive_model')
    model_filename = predictive_model_config.get('model_filename', 'model.joblib')
    onnx_model_filename = predictive_model_config.get('onnx_model_filename', 'model.onnx')
    preprocessor_path = predictive_model_config.get('preprocessor_path', 'data/processed/preprocessor.pkl')

    # Construct paths
    MODEL_PATH = os.path.join(model_dir, model_filename)
    OUTPUT_PATH = os.path.join(model_dir, onnx_model_filename)
    PREPROCESSOR_PATH = preprocessor_path

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
