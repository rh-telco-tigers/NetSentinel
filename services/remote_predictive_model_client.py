# remote_predictive_model_client.py

import requests
import logging

logger = logging.getLogger(__name__)

class RemotePredictiveModelClient:
    def __init__(self, url, token=None, verify_ssl=True, timeout=30):
        self.url = url
        self.token = token
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.headers = {'Content-Type': 'application/json'}
        if self.token:
            self.headers['Authorization'] = f'Bearer {self.token}'

    def predict(self, features):
        try:
            # Prepare the payload as per the required format
            payload = {
                "inputs": [
                    {
                        "name": "float_input",
                        "shape": [1, len(features)],
                        "datatype": "FP32",
                        "data": features
                    }
                ]
            }

            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            # Extract the prediction and probabilities from the response
            outputs = result.get('outputs', [])
            label_output = next((output for output in outputs if output.get('name') == 'label'), None)
            probabilities_output = next((output for output in outputs if output.get('name') == 'probabilities'), None)

            if label_output is None or probabilities_output is None:
                logger.error("Missing expected outputs in response")
                return None, None

            prediction = label_output.get('data', [None])[0]
            probabilities = probabilities_output.get('data', [])

            return prediction, probabilities

        except requests.exceptions.RequestException as e:
            logger.error(f"Error while communicating with remote predictive model: {e}")
            raise e
