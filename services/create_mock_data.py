# create_mock_data.py

import json
import os
import time
import yaml
from kafka import KafkaProducer
import logging
import random
from datetime import datetime, timedelta
import ipaddress
import uuid
import signal
import sys

# Initialize logger
logger = logging.getLogger(__name__)

SCANNING_TOOL_CONFIG = {
    "publish_interval_seconds": 10,
    "subnets": {
        "src_subnets": ["192.168.1.0/24", "10.0.0.0/24"],
        "dst_subnets": ["172.16.0.0/24", "10.1.1.0/24"],
    },
    "protocols": {
        "TCP": {"ports": [80, 443, 22, 21]},
        "UDP": {"ports": [53, 67, 68]},
        "ICMP": [],
        "HTTP": {
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "status_codes": [200, 201, 400, 401, 403, 404, 500],
            "urls": [
                "/home", "/login", "/dashboard", "/api/data", "/logout",
                "/register", "/profile", "/settings", "/search", "/contact",
                "/products", "/cart", "/checkout", "/help", "/about",
                "/terms", "/privacy", "/blog", "/news", "/support"
            ],
        },
        "DNS": {"query_types": ["A", "AAAA", "MX", "CNAME", "TXT"]},
    },
}

def setup_logging(config):
    """
    Configure logging based on the configuration.
    """
    log_level = config.get("logging", {}).get("level", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(numeric_level)

def load_config(config_path="config.yaml"):
    """
    Load configuration from the YAML file.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def create_kafka_producer(kafka_bootstrap_servers):
    """
    Create and return a Kafka producer.
    """
    try:
        use_ssl = os.path.exists("/usr/local/share/ca-certificates/ca.crt")

        producer_args = {
            "bootstrap_servers": kafka_bootstrap_servers,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "retries": 5,
            "acks": "all",
            "security_protocol": "SASL_SSL" if use_ssl else "SASL_PLAINTEXT",
            "sasl_mechanism": "SCRAM-SHA-512" if use_ssl else "PLAIN",
            "sasl_plain_username": os.getenv("KAFKA_USERNAME", "admin"),
            "sasl_plain_password": os.getenv("KAFKA_PASSWORD", "secret-password"),
        }

        if use_ssl:
            producer_args["ssl_cafile"] = "/usr/local/share/ca-certificates/ca.crt"
            producer_args["ssl_check_hostname"] = False

        producer = KafkaProducer(**producer_args)
        logger.info(f"Connected to Kafka at {kafka_bootstrap_servers}")
        return producer
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        sys.exit(1)

def get_random_ip(subnet_list):
    """
    Generate a random IP address from the provided list of subnets.
    """
    try:
        subnet = ipaddress.ip_network(random.choice(subnet_list))
        return str(random.choice(list(subnet.hosts())))
    except Exception as e:
        logger.error(f"Error generating IP address: {e}")
        return "0.0.0.0"

def generate_mock_data(config):
    """
    Generate a single mock network traffic data record based on the configuration.
    """
    try:
        # Define possible values for categorical features
        protocols = ["tcp", "udp", "icmp", "arp"]
        states = ["CON", "INT", "FIN", "REQ", "RST", "URH", "URN", "PAR", "ECO", "ECR", "MAS", "OTH"]
        services = ["-", "http", "ftp", "smtp", "ssh", "dns", "ftp-data", "irc"]

        # Generate feature values
        srcip = get_random_ip(SCANNING_TOOL_CONFIG["subnets"]["src_subnets"])
        sport = str(random.randint(1024, 65535))
        dstip = get_random_ip(SCANNING_TOOL_CONFIG["subnets"]["dst_subnets"])
        dsport = str(random.randint(1, 1024))
        proto = random.choice(protocols)
        state = random.choice(states)
        dur = round(random.uniform(0.0, 1000.0), 6)
        sbytes = random.randint(0, 1000000)
        dbytes = random.randint(0, 1000000)
        sttl = random.randint(1, 255)
        dttl = random.randint(1, 255)
        sloss = random.randint(0, 10)
        dloss = random.randint(0, 10)
        service = random.choice(services)
        sload = round(random.uniform(0.0, 1000000.0), 6)
        dload = round(random.uniform(0.0, 1000000.0), 6)
        spkts = random.randint(1, 100000)
        dpkts = random.randint(1, 100000)
        swin = random.randint(0, 65535)
        dwin = random.randint(0, 65535)
        stcpb = random.randint(0, 4294967295)
        dtcpb = random.randint(0, 4294967295)
        smeansz = random.randint(0, 10000)
        dmeansz = random.randint(0, 10000)
        trans_depth = random.randint(0, 100)
        res_bdy_len = random.randint(0, 500000)
        sjit = round(random.uniform(0.0, 1000.0), 6)
        djit = round(random.uniform(0.0, 1000.0), 6)
        stime = datetime.utcnow() - timedelta(seconds=random.randint(0, 3600))
        ltime = stime + timedelta(seconds=dur)
        sintpkt = round(random.uniform(0.0, 1000.0), 6)
        dintpkt = round(random.uniform(0.0, 1000.0), 6)
        tcprtt = round(random.uniform(0.0, 1.0), 6)
        synack = round(random.uniform(0.0, 1.0), 6)
        ackdat = round(random.uniform(0.0, 1.0), 6)
        is_sm_ips_ports = random.randint(0, 1)
        ct_state_ttl = random.randint(0, 10)
        ct_flw_http_mthd = random.randint(0, 5)
        is_ftp_login = random.randint(0, 1)
        ct_ftp_cmd = str(random.randint(0, 10))
        ct_srv_src = random.randint(0, 1000)
        ct_srv_dst = random.randint(0, 1000)
        ct_dst_ltm = random.randint(0, 1000)
        ct_src_ltm = random.randint(0, 1000)
        ct_src_dport_ltm = random.randint(0, 1000)
        ct_dst_sport_ltm = random.randint(0, 1000)
        ct_dst_src_ltm = random.randint(0, 1000)

        # Build the mock data record
        mock_data = {
            "srcip": srcip,
            "sport": sport,
            "dstip": dstip,
            "dsport": dsport,
            "proto": proto,
            "state": state,
            "dur": dur,
            "sbytes": sbytes,
            "dbytes": dbytes,
            "sttl": sttl,
            "dttl": dttl,
            "sloss": sloss,
            "dloss": dloss,
            "service": service,
            "Sload": sload,
            "Dload": dload,
            "Spkts": spkts,
            "Dpkts": dpkts,
            "swin": swin,
            "dwin": dwin,
            "stcpb": stcpb,
            "dtcpb": dtcpb,
            "smeansz": smeansz,
            "dmeansz": dmeansz,
            "trans_depth": trans_depth,
            "res_bdy_len": res_bdy_len,
            "Sjit": sjit,
            "Djit": djit,
            "Stime": stime.isoformat(),
            "Ltime": ltime.isoformat(),
            "Sintpkt": sintpkt,
            "Dintpkt": dintpkt,
            "tcprtt": tcprtt,
            "synack": synack,
            "ackdat": ackdat,
            "is_sm_ips_ports": is_sm_ips_ports,
            "ct_state_ttl": ct_state_ttl,
            "ct_flw_http_mthd": ct_flw_http_mthd,
            "is_ftp_login": is_ftp_login,
            "ct_ftp_cmd": ct_ftp_cmd,  # Now a string
            "ct_srv_src": ct_srv_src,
            "ct_srv_dst": ct_srv_dst,
            "ct_dst_ltm": ct_dst_ltm,
            "ct_src_ ltm": ct_src_ltm,
            "ct_src_dport_ltm": ct_src_dport_ltm,
            "ct_dst_sport_ltm": ct_dst_sport_ltm,
            "ct_dst_src_ltm": ct_dst_src_ltm,
            "event_id": str(uuid.uuid4()),
        }

        return mock_data

    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        return None

def validate_mock_data(data):
    """
    Comprehensive validation to ensure data integrity.
    """
    required_fields = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
        "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
        "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
        "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
        "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "event_id"
    ]
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing field: {field}")
            return False

    # Validate IP addresses
    try:
        ipaddress.ip_address(data["srcip"])
        ipaddress.ip_address(data["dstip"])
    except ValueError:
        logger.warning("Invalid IP address format.")
        return False

    # Validate numerical fields
    numerical_fields = [
        "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
        "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz",
        "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Sintpkt", "Dintpkt",
        "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
        "is_ftp_login", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
        "ct_src_ ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
    ]
    for field in numerical_fields:
        if not isinstance(data[field], (int, float)):
            logger.warning(f"Invalid type for field {field}: Expected int or float.")
            return False

    # Validate timestamps
    try:
        datetime.fromisoformat(data["Stime"])
        datetime.fromisoformat(data["Ltime"])
    except ValueError:
        logger.warning("Invalid timestamp format.")
        return False

    return True

def publish_mock_data(producer, topic, config):
    """
    Generate, validate, and publish mock data to Kafka.
    """
    mock_data = generate_mock_data(config)
    if not mock_data:
        logger.error("Mock data generation returned None. Skipping.")
        return
    if not validate_mock_data(mock_data):
        logger.error("Generated mock data failed validation. Skipping.")
        return
    try:
        producer.send(topic, mock_data)
        producer.flush()
        logger.info(f"Published mock data to {topic}: {mock_data}")
    except Exception as e:
        logger.error(f"Failed to publish mock data: {e}")

def signal_handler(sig, frame, producer):
    """
    Handle termination signals for graceful shutdown.
    """
    logger.info("Shutting down gracefully...")
    producer.close()
    sys.exit(0)

def main():
    """
    Main function to run the mock data generator.
    """
    config = load_config()
    setup_logging(config)

    kafka_bootstrap = config.get("kafka", {}).get("bootstrap_servers", "localhost:9092")
    raw_topic = config.get("kafka", {}).get("topics", {}).get("raw", "raw-traffic-data")
    
    producer = create_kafka_producer(kafka_bootstrap)

    publish_interval = SCANNING_TOOL_CONFIG.get("publish_interval_seconds", 10)

    signal.signal(
        signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, producer)
    )
    signal.signal(
        signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, producer)
    )

    logger.info("Starting mock data publishing...")
    while True:
        publish_mock_data(producer, raw_topic, config)
        time.sleep(publish_interval)

if __name__ == "__main__":
    main()
