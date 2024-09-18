# scripts/create_mock_data.py

import json
import os
import time
import yaml
from kafka import KafkaProducer
import logging
import random
from datetime import datetime
import ipaddress
import uuid
import signal
import sys


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


# Initialize logger
logger = logging.getLogger(__name__)


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


def create_producer(kafka_bootstrap_servers):
    """
    Create and return a Kafka producer with retries and acks.
    """
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,  # Retry up to 5 times on failure
            acks="all",  # Wait for all replicas to acknowledge
        )
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
        protocols = [
            "TCP",
            "UDP",
            "ICMP",
            "HTTP",
            "HTTPS",
            "FTP",
            "SMTP",
            "DNS",
            "SSH",
            "TELNET",
        ]
        services = [
            "HTTP",
            "HTTPS",
            "FTP",
            "SSH",
            "DNS",
            "SMTP",
            "TELNET",
            "SNMP",
            "POP3",
            "IMAP",
        ]
        states = ["S0", "S1", "S2", "S3", "SF", "REJ", "RSTO", "RSTR", "SH", "OTH"]

        # Select protocol
        proto = random.choice(protocols)

        # Select service based on protocol
        if proto in ["TCP", "UDP", "ICMP"]:
            service = random.choice(services)
        elif proto == "HTTP":
            service = "HTTP"
        elif proto == "HTTPS":
            service = "HTTPS"
        elif proto == "FTP":
            service = "FTP"
        elif proto == "SMTP":
            service = "SMTP"
        elif proto == "DNS":
            service = "DNS"
        elif proto == "SSH":
            service = "SSH"
        elif proto == "TELNET":
            service = "TELNET"
        else:
            service = "OTHER"

        # Select state
        state = random.choice(states)

        # Get source and destination IPs
        src_ip = get_random_ip(config["scanning_tool"]["subnets"]["src_subnets"])
        dst_ip = get_random_ip(config["scanning_tool"]["subnets"]["dst_subnets"])

        # Select ports based on protocol
        if proto in ["TCP", "UDP"]:
            ports = config["scanning_tool"]["protocols"][proto]["ports"]
            src_port = random.choice(ports)
            dst_port = random.choice(ports)
        else:
            src_port = None
            dst_port = None

        # Initialize protocol-specific details
        protocol_details = {}
        if proto == "HTTP":
            methods = config["scanning_tool"]["protocols"]["HTTP"]["methods"]
            status_codes = config["scanning_tool"]["protocols"]["HTTP"]["status_codes"]
            urls = config["scanning_tool"]["protocols"]["HTTP"]["urls"]
            protocol_details["method"] = random.choice(methods)
            protocol_details["status_code"] = random.choice(status_codes)
            protocol_details["url"] = random.choice(urls)
        elif proto == "DNS":
            query_types = config["scanning_tool"]["protocols"]["DNS"]["query_types"]
            domains = [
                "example.com",
                "testsite.org",
                "mydomain.net",
                "sample.co",
                "website.io",
                "service.biz",
                "application.ai",
                "platform.dev",
                "network.tech",
                "cloudservice.com",
                "securemail.net",
                "fastdns.org",
                "reliableftp.com",
                "smartsmtp.io",
                "deepdns.tech",
            ]
            protocol_details["query_type"] = random.choice(query_types)
            protocol_details["domain"] = random.choice(domains)
        elif proto == "FTP":
            ftp_commands = ["LIST", "RETR", "STOR", "DELE", "PWD", "CWD"]
            protocol_details["command"] = random.choice(ftp_commands)
        elif proto == "SSH":
            auth_methods = ["password", "publickey"]
            protocol_details["auth_method"] = random.choice(auth_methods)
        elif proto == "SMTP":
            mail_from = random.choice(
                [
                    "alice@example.com",
                    "bob@testsite.org",
                    "carol@mydomain.net",
                    "dave@sample.co",
                    "eve@website.io",
                    "frank@service.biz",
                ]
            )
            mail_to = random.choice(
                [
                    "mallory@example.com",
                    "trent@testsite.org",
                    "peggy@mydomain.net",
                    "victor@sample.co",
                    "walter@website.io",
                    "sybil@service.biz",
                ]
            )
            subject_lines = [
                "Meeting Schedule",
                "Project Update",
                "Invoice Attached",
                "Welcome to the Team",
                "Your Order Confirmation",
                "Security Alert",
            ]
            protocol_details["mail_from"] = mail_from
            protocol_details["mail_to"] = mail_to
            protocol_details["subject"] = random.choice(subject_lines)
        # Add more protocol-specific details as needed

        # Generate random numerical values based on protocol type
        if proto in ["TCP", "UDP"]:
            sbytes = random.randint(500, 10000)
            dbytes = random.randint(500, 10000)
            sttl = random.randint(30, 128)
            dttl = random.randint(30, 128)
            sloss = (
                random.randint(0, 10) if random.random() < 0.1 else 0
            )  # 10% chance of loss
            dloss = random.randint(0, 10) if random.random() < 0.1 else 0
            sload = round(random.uniform(0.0, 1.0), 2)
            dload = round(random.uniform(0.0, 1.0), 2)
            spkts = random.randint(1, 100)
            dpkts = random.randint(1, 100)
        elif proto == "ICMP":
            sbytes = random.randint(100, 1000)
            dbytes = random.randint(100, 1000)
            sttl = random.randint(30, 128)
            dttl = random.randint(30, 128)
            sloss = 0
            dloss = 0
            sload = 0.0
            dload = 0.0
            spkts = random.randint(1, 50)
            dpkts = random.randint(1, 50)
        else:
            # For application-layer protocols like HTTP, HTTPS, etc.
            sbytes = random.randint(1000, 50000)
            dbytes = random.randint(1000, 50000)
            sttl = random.randint(30, 128)
            dttl = random.randint(30, 128)
            sloss = (
                random.randint(0, 5) if random.random() < 0.05 else 0
            )  # 5% chance of loss
            dloss = random.randint(0, 5) if random.random() < 0.05 else 0
            sload = round(random.uniform(0.0, 1.0), 2)
            dload = round(random.uniform(0.0, 1.0), 2)
            spkts = random.randint(10, 500)
            dpkts = random.randint(10, 500)

        # Generate payload sizes based on bytes
        payload_size = {"sbytes": sbytes, "dbytes": dbytes}

        # Generate unique session ID and connection duration
        session_id = str(uuid.uuid4())
        connection_duration = round(
            random.uniform(0.1, 300.0), 2
        )  # Duration in seconds

        mock_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "event_id": session_id,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "proto": proto,
            "service": service,
            "state": state,
            "sbytes": sbytes,
            "dbytes": dbytes,
            "sttl": sttl,
            "dttl": dttl,
            "sloss": sloss,
            "dloss": dloss,
            "sload": sload,
            "dload": dload,
            "spkts": spkts,
            "dpkts": dpkts,
            "connection_duration": connection_duration,  # Duration in seconds
        }

        # Add ports if applicable
        if src_port:
            mock_data["src_port"] = src_port
        if dst_port:
            mock_data["dst_port"] = dst_port
            

        # Add protocol-specific details
        mock_data.update(protocol_details)

        return mock_data

    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        return None


def validate_mock_data(data):
    """
    Comprehensive validation to ensure data integrity.
    """
    required_fields = [
        "timestamp",
        "session_id",
        "src_ip",
        "dst_ip",
        "proto",
        "service",
        "state",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        "sload",
        "dload",
        "spkts",
        "dpkts",
        "connection_duration",
    ]
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing field: {field}")
            return False

    # Validate IP addresses
    try:
        ipaddress.ip_address(data["src_ip"])
        ipaddress.ip_address(data["dst_ip"])
    except ValueError:
        logger.warning("Invalid IP address format.")
        return False

    # Validate numerical fields
    numerical_fields = [
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        "sload",
        "dload",
        "spkts",
        "dpkts",
        "connection_duration",
    ]
    for field in numerical_fields:
        if not isinstance(data[field], (int, float)):
            logger.warning(f"Invalid type for field {field}: Expected int or float.")
            return False
        if field in ["sload", "dload"] and not (0.0 <= data[field] <= 1.0):
            logger.warning(
                f"Invalid value for {field}: {data[field]} (Expected between 0.0 and 1.0)"
            )
            return False

    # Validate protocol-specific fields
    if data["proto"] == "HTTP":
        http_fields = ["method", "status_code", "url"]
        for field in http_fields:
            if field not in data:
                logger.warning(f"Missing HTTP-specific field: {field}")
                return False
    elif data["proto"] == "DNS":
        dns_fields = ["query_type", "domain"]
        for field in dns_fields:
            if field not in data:
                logger.warning(f"Missing DNS-specific field: {field}")
                return False
    elif data["proto"] == "FTP":
        ftp_fields = ["command"]
        for field in ftp_fields:
            if field not in data:
                logger.warning(f"Missing FTP-specific field: {field}")
                return False
    elif data["proto"] == "SSH":
        ssh_fields = ["auth_method"]
        for field in ssh_fields:
            if field not in data:
                logger.warning(f"Missing SSH-specific field: {field}")
                return False
    elif data["proto"] == "SMTP":
        smtp_fields = ["mail_from", "mail_to", "subject"]
        for field in smtp_fields:
            if field not in data:
                logger.warning(f"Missing SMTP-specific field: {field}")
                return False
    # Add more protocol-specific validations as needed

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
    Main function to run the scanning tool.
    """
    config = load_config()
    setup_logging(config)
    kafka_bootstrap = config.get("kafka", {}).get("bootstrap", "localhost:9092")
    producer = create_producer(kafka_bootstrap)
    topic = "raw-traffic-data"  # Ensure this topic exists in Kafka
    publish_interval = config.get("scanning_tool", {}).get(
        "publish_interval_seconds", 10
    )

    # Register signal handlers for graceful shutdown
    signal.signal(
        signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, producer)
    )
    signal.signal(
        signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, producer)
    )

    logger.info("Starting mock data publishing...")
    while True:
        publish_mock_data(producer, topic, config)
        time.sleep(publish_interval)  # Publish based on configured interval


if __name__ == "__main__":
    main()
