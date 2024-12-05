# app/remote_llm_client.py

import requests
import logging

logger = logging.getLogger(__name__)

class RemoteLLMClient:
    def __init__(self, url, model_name, token=None, verify_ssl=True):
        self.url = url
        self.model_name = model_name
        self.token = token
        self.verify_ssl = verify_ssl
        self.headers = {'Content-Type': 'application/json'}
        if self.token:
            self.headers['Authorization'] = f'Bearer {self.token}'
        if not verify_ssl:
            logger.warning("SSL verification is disabled. This is insecure and should not be used in production.")

    def generate_response(self, input_text,  max_length=150,):
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': input_text}
            ],
            'max_tokens': max_length
        }
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=30  # Adjust as needed
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            if not generated_text:
                logger.error("Remote LLM returned an empty response.")
                raise ValueError("Empty response from remote LLM.")
            return generated_text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error while communicating with remote LLM: {e}")
            raise e
        except ValueError as ve:
            logger.error(f"Value error: {ve}")
            raise ve
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise e
