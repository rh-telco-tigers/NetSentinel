# app/models.py

import joblib
import onnxruntime as ort
from transformers import pipeline
import logging
import re
import torch

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

class LLMModel:
    def __init__(self, model_path):
        try:
            # Consider using 'text2text-generation' or 'question-answering' for better results
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model_path,
                tokenizer=model_path,
                device=0 if torch.cuda.is_available() else -1  # Utilize GPU if available
            )
            logger.info(f"LLM loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load LLM from {model_path}: {e}")
            raise e

    def generate_response(self, question):
        try:
            response = self.llm_pipeline(
                question,
                max_length=100,            # Adjust based on expected response length
                num_return_sequences=1,    # Ensure only one response is generated
                no_repeat_ngram_size=3,    # Prevent repeating phrases
                do_sample=False,           # Use deterministic output
                temperature=0.7,           # Adjust for creativity vs. determinism
                eos_token_id=self.llm_pipeline.tokenizer.eos_token_id  # Ensure proper termination
            )
            generated_text = response[0]['generated_text']
            logger.debug(f"Raw LLM response: {generated_text}")

            # Post-process to extract only the first complete sentence
            processed_response = self._extract_first_sentence(generated_text)
            logger.debug(f"Processed LLM response: {processed_response}")
            return processed_response
        except Exception as e:
            logger.error(f"Error during LLM response generation: {e}")
            raise e

    def _extract_first_sentence(self, text):
        """
        Extracts the first complete sentence from the generated text.
        """
        # Use regex to split the text into sentences more accurately
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        if sentences:
            return sentences[0]
        return text.strip()
