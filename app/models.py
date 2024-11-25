# app/models.py

import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)

class PredictiveModel:
    def __init__(self, model_path):
        try:
            self.session = ort.InferenceSession(model_path)
            logger.info(f"Predictive model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load predictive model from {model_path}: {e}")
            raise e

    def predict(self, input_data):
        try:
            # Prepare inputs as per the model's expected format
            # Example input_data: {'proto': 'TCP', 'src_port': 80, 'dst_port': 443}
            inputs = [
                input_data.get('proto', ''),
                float(input_data.get('src_port', 0)),
                float(input_data.get('dst_port', 0))
            ]
            ort_inputs = {self.session.get_inputs()[0].name: [inputs]}
            ort_out = self.session.run(None, ort_inputs)

            # Assuming the model outputs probabilities or labels
            prediction = ort_out[0][0].tolist()  # Convert numpy array to list
            logger.debug(f"Prediction result: {prediction}")
            return {"prediction": prediction}
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise e
