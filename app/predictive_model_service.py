# app/predictive_model_service.py

from kafka import KafkaConsumer, KafkaProducer
import json
import os
from app.models import PredictiveModel

class PredictiveModelService:
    def __init__(self, kafka_bootstrap_servers, input_topic, output_topic, model_path):
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.predictive_model = PredictiveModel(model_path)

    def process_messages(self):
        for message in self.consumer:
            raw_data = message.value
            prediction = self.predictive_model.predict(raw_data)
            processed_data = {**raw_data, "prediction": prediction["prediction"]}
            self.producer.send(self.output_topic, processed_data)
            self.producer.flush()
            print(f"Processed and published data to {self.output_topic}")

if __name__ == "__main__":
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "netsentenial-kafka-kafka-bootstrap.openshift-operators:9092")
    input_topic = "raw-traffic-data"
    output_topic = "processed-traffic-data"
    model_path = "path/to/predictive_model.onnx"
    service = PredictiveModelService(kafka_bootstrap_servers, input_topic, output_topic, model_path)
    service.process_messages()
