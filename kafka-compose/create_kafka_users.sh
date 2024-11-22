#!/bin/bash

set -e

echo "Waiting for ZooKeeper to be ready..."
while ! echo ruok | nc -w 2 zookeeper 2181 | grep imok > /dev/null; do
    sleep 1
done

echo "Waiting for Kafka to be ready..."
while ! kafka-topics --bootstrap-server kafka:9092 --command-config /etc/kafka/client.properties --list > /dev/null 2>&1; do
    sleep 1
done

echo "Creating Kafka user for SCRAM-SHA-512..."
kafka-configs --bootstrap-server kafka:9092 --alter --add-config 'SCRAM-SHA-512=[password=secret-password]' --entity-type users --entity-name admin --command-config /etc/kafka/client.properties
echo "Kafka user created successfully."
