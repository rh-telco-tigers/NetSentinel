# app/slack_integration.py

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging

logger = logging.getLogger(__name__)

class SlackClient:
    def __init__(self, bot_token):
        try:
            self.client = WebClient(token=bot_token)
            logger.info("Slack client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Slack client: {e}")
            raise e

    def send_message(self, channel, text):
        try:
            response = self.client.chat_postMessage(channel=channel, text=text)
            logger.info(f"Message sent to {channel}: {text}")
            return response
        except SlackApiError as e:
            logger.error(f"Error sending message to Slack: {e.response['error']}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error sending message to Slack: {e}")
            raise e
