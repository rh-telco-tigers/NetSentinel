# scripts/cleanup_kafka_topics.py

from kafka.admin import KafkaAdminClient, ConfigResource, ConfigResourceType
import logging

def cleanup_kafka_topics(bootstrap_servers, topics, retention_ms=0):
    """
    Cleans up all data from the specified Kafka topics by setting their retention time.

    Args:
        bootstrap_servers (str): The Kafka bootstrap servers (e.g., 'localhost:9092').
        topics (list): List of topic names to clean up.
        retention_ms (int, optional): The retention time in milliseconds to set. Defaults to 0.
    """
    admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)

    # Prepare the configurations
    configs = {}
    for topic in topics:
        resource = ConfigResource(ConfigResourceType.TOPIC, topic)
        configs[resource] = {'retention.ms': str(retention_ms)}

    # Alter the topic configurations
    try:
        admin_client.alter_configs(configs)
        logging.info(f"Set retention.ms to {retention_ms} for topics: {topics}")
    except Exception as e:
        logging.error(f"Failed to alter topic configurations: {e}")
    finally:
        admin_client.close()

if __name__ == "__main__":
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Kafka bootstrap servers
    bootstrap_servers = 'netsentenial-kafka-kafka-bootstrap.openshift-operators:9092'  # Replace with your Kafka bootstrap servers

    # List of topics to clean up
    topics_to_clean = ['processed-traffic-data', 'raw-traffic-data']  # Replace with your topic names

    # Clean up topics by setting retention.ms to 0
    cleanup_kafka_topics(bootstrap_servers, topics_to_clean)

    # Optionally, reset retention.ms back to 7 days (default)
    reset_retention_ms = 604800000  # 7 days in milliseconds
    cleanup_kafka_topics(bootstrap_servers, topics_to_clean, retention_ms=reset_retention_ms)
