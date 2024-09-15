import onnx

def inspect_onnx_model(onnx_model_path):
    """
    Inspects the ONNX model and prints the input feature names and their types.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
    """
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    input_names = []
    for input_tensor in model.graph.input:
        # Exclude initializer tensors (constants)
        if input_tensor.name not in [init.name for init in model.graph.initializer]:
            input_names.append((input_tensor.name, input_tensor.type.tensor_type.elem_type))

    print("ONNX Model Inputs:")
    for name, elem_type in input_names:
        # Map ONNX tensor types to human-readable types
        type_map = {
            1: 'float32',
            2: 'float64',
            3: 'int32',
            4: 'int64',
            5: 'string',
            # Add more mappings if necessary
        }
        readable_type = type_map.get(elem_type, 'unknown')
        print(f" - {name}: {readable_type}")

if __name__ == "__main__":
    import os
    onnx_model_path = os.path.join('models', 'predictive_model', 'model.onnx')
    inspect_onnx_model(onnx_model_path)
