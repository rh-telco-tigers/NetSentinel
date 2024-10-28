@load base/frameworks/cluster
@load base/frameworks/logging

# Load any required packages
@load packages/zeek-kafka

redef ignore_checksums = T;

redef Kafka::logs_to_send = set(Conn::LOG, HTTP::LOG, DNS::LOG);
redef Kafka::topic_name = "zeek_logs";
redef Kafka::tag_json = T;
redef Kafka::kafka_conf = table(
    ["metadata.broker.list"] = "kafka:29092"
);
