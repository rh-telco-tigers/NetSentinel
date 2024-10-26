# zeek/zeek-config/local.zeek

@load packages/zeek-kafka
@load /usr/local/zeek_scripts/feature_extraction.zeek

redef Kafka::tag_json = T;
redef Kafka::send_all_active_logs = F;
redef Kafka::topic_name = "";  # Use log path as topic name

redef Kafka::kafka_conf = table(
    ["metadata.broker.list"] = "kafka:9092",
    ["client.id"] = "zeek-client"
);

event zeek_init() &priority=-10
    {
    # Send Conn logs to 'conn' topic
    Log::add_filter(Conn::LOG, [$name="kafka-conn", $writer=Log::WRITER_KAFKAWRITER,
        $path="conn", $config=table(["topic_name"] = "conn")]);

    # Send HTTP logs to 'http' topic
    Log::add_filter(HTTP::LOG, [$name="kafka-http", $writer=Log::WRITER_KAFKAWRITER,
        $path="http", $config=table(["topic_name"] = "http")]);

    # Send DNS logs to 'dns' topic
    Log::add_filter(DNS::LOG, [$name="kafka-dns", $writer=Log::WRITER_KAFKAWRITER,
        $path="dns", $config=table(["topic_name"] = "dns")]);

    # Add other logs as needed
    }
