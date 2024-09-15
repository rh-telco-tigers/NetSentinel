# scripts/validate_predictive_model.py

import onnx
import onnxruntime as ort
import numpy as np
import os

def inspect_onnx_model(onnx_model_path):
    """
    Inspects the ONNX model and returns the input feature names and their types.

    Args:
        onnx_model_path (str): Path to the ONNX model file.

    Returns:
        List of tuples with feature names and data types.
    """
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    input_features = []
    for input_tensor in model.graph.input:
        # Exclude initializer tensors (constants)
        if input_tensor.name not in [init.name for init in model.graph.initializer]:
            name = input_tensor.name
            elem_type = input_tensor.type.tensor_type.elem_type
            input_features.append((name, elem_type))

    return input_features

def prepare_sample_input(input_features):
    """
    Prepares a sample input dictionary based on the expected features.

    Args:
        input_features (list): List of tuples with feature names and data types.

    Returns:
        Dictionary with feature names as keys and numpy arrays as values.
    """
    sample_data = {
        'proto': 'HTTP',
        'service': 'web',
        'state': 'S0',
        'sbytes': 1024.0,
        'dbytes': 2048.0,
        'sttl': 64.0,
        'dttl': 128.0,
        'sloss': 0.0,
        'dloss': 0.0,
        'sload': 0.0,
        'dload': 0.0,
        'spkts': 10.0,
        'dpkts': 20.0
    }

    # One-Hot Encoding for Categorical Features
    # Note: The encoding should match the preprocessor's encoding during training
    proto_categories = ['HTTP', 'FTP', 'SSH']        # Replace with actual categories
    service_categories = ['web', 'ftp', 'ssh', 'dns'] # Replace with actual categories
    state_categories = ['S0', 'S1']                  # Replace with actual categories

    # Function to one-hot encode a categorical feature
    def one_hot_encode(value, categories):
        return [1 if value == cat else 0 for cat in categories]

    # Encode each categorical feature
    proto_encoded = one_hot_encode(sample_data['proto'], proto_categories)
    service_encoded = one_hot_encode(sample_data['service'], service_categories)
    state_encoded = one_hot_encode(sample_data['state'], state_categories)

    # Combine all features
    numerical_features = [
        sample_data['sbytes'],
        sample_data['dbytes'],
        sample_data['sttl'],
        sample_data['dttl'],
        sample_data['sloss'],
        sample_data['dloss'],
        sample_data['sload'],
        sample_data['dload'],
        sample_data['spkts'],
        sample_data['dpkts']
    ]

    # Flatten the input features
    # Adjust the order and number of encoded features based on your preprocessor
    input_array = np.array([proto_encoded + service_encoded + state_encoded + numerical_features], dtype=np.float32)

    return {'classifier_input': input_array}  # Replace 'classifier_input' with the actual input name if different

def validate_onnx_model(onnx_model_path):
    """
    Validates the ONNX model by running a sample inference.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
    """
    # Load the ONNX model
    session = ort.InferenceSession(onnx_model_path)

    # Get expected input features
    input_features = inspect_onnx_model(onnx_model_path)

    # Prepare sample input
    sample_input = prepare_sample_input(input_features)

    # Run inference
    try:
        outputs = session.run(None, sample_input)
        print("Inference Successful. Output:")
        for output in outputs:
            print(output)
    except Exception as e:
        print(f"Inference Failed: {e}")

if __name__ == "__main__":
    # Path to the ONNX model
    onnx_model_path = os.path.join('models', 'predictive_model', 'model.onnx')

    # Validate the model
    validate_onnx_model(onnx_model_path)
