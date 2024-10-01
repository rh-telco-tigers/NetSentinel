import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    packages_to_install=['scikit-learn', 'joblib', 'skl2onnx'],
)
def export_model_to_onnx_component(
    model_input: Input[Model],
    processed_data_path: Input[Dataset],
    onnx_model_output: Output[Model],
):
    import os
    import joblib
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    from sklearn.pipeline import Pipeline

    model = joblib.load(os.path.join(model_input.path, 'model.joblib'))
    preprocessor = joblib.load(os.path.join(processed_data_path.path, 'preprocessor.pkl'))

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    def get_feature_details(preprocessor):
        feature_details = []
        for transformer in preprocessor.transformers_:
            transformer_name, transformer_object, columns = transformer
            for column in columns:
                if column in ['proto', 'service', 'state']:
                    feature_details.append((column, StringTensorType([None, 1])))
                else:
                    feature_details.append((column, FloatTensorType([None, 1])))
        return feature_details

    initial_type = get_feature_details(preprocessor)

    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)
    os.makedirs(onnx_model_output.path, exist_ok=True)
    onnx_model_path = os.path.join(onnx_model_output.path, 'model.onnx')
    with open(onnx_model_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model exported to ONNX format at {onnx_model_path}")
