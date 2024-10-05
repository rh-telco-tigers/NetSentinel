# app/intent_handlers.py

import logging
from typing import Dict

from .utils import (
    generate_response,
    get_event_by_id,
    get_all_attack_event_ids,
    get_events_by_src_ip,
    get_events_by_dst_ip,
    build_context_from_event_data,
    get_recent_events,
    extract_namespace
)

logger = logging.getLogger(__name__)

def log_extracted_entities(entities: Dict):
    """
    Logs all the extracted entities for debugging purposes.
    """
    logger.info(f"Extracted entities: {entities}")

# -------------------------------
# General Intent Handlers
# -------------------------------

def handle_greet(entities: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    return "Hello! How can I assist you today?"

def handle_goodbye(entities: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    return "Goodbye! If you have more questions, feel free to ask."

def handle_get_event_info(entities: Dict, event_id_index: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    event_id = entities.get('event_id')
    if not event_id:
        return "Please provide a valid event ID."

    event_data = get_event_by_id(event_id, event_id_index)
    if not event_data:
        return "No data found for the provided event ID."

    text = entities.get('text', '').lower()
    if "source ip" in text:
        return f"Source IP: {event_data.get('src_ip', 'N/A')}"
    elif "destination ip" in text:
        return f"Destination IP: {event_data.get('dst_ip', 'N/A')}"
    elif "attack" in text:
        prediction = event_data.get('prediction')
        return "Yes, it is an attack." if prediction == 1 else "No, it is not an attack."
    elif "prediction probability" in text:
        proba = event_data.get('prediction_proba', 'N/A')
        return f"Prediction Probability: {proba}"
    elif "what kind of traffic" in text or "traffic type" in text:
        protocol = event_data.get('protocol', 'N/A')
        service = event_data.get('service', 'N/A')
        return f"Protocol: {protocol}, Service: {service}"
    else:
        return build_context_from_event_data(event_data)

def handle_list_attack_events(entities: Dict, metadata_store: list, **kwargs) -> str:
    log_extracted_entities(entities)
    event_ids = get_all_attack_event_ids(metadata_store)
    if event_ids:
        return "Attack Event IDs:\n" + "\n".join(event_ids)
    else:
        return "No attack events found."

def handle_list_recent_attack_events(entities: Dict, metadata_store: list, **kwargs) -> str:
    """
    Handle intent to list recent attack events.
    Accepts an optional 'number' entity to specify how many events to list.
    Defaults to 10 if not provided.
    """
    log_extracted_entities(entities)

    number = entities.get('number')
    if not number:
        logger.warning(f"No 'number' entity found in: {entities}. Defaulting to 10.")
    try:
        limit = int(number) if number else 10
    except ValueError:
        logger.warning(f"Invalid number provided: {number}. Defaulting to 10.")
        limit = 10

    logger.info(f"Listing {limit} recent attack events.")
    # Fetch recent attack events
    recent_attack_events = get_recent_events(metadata_store, event_type='attack', limit=limit)
    
    if recent_attack_events:
        event_list = "\n".join([f"- Event ID: {event['event_id']} at {event['timestamp']}" for event in recent_attack_events])
        final_message = f"Here are the last {limit} attack events:\n{event_list}"
    else:
        final_message = "No recent attack events found."

    return final_message

def handle_list_recent_normal_events(entities: Dict, metadata_store: list, **kwargs) -> str:
    log_extracted_entities(entities)
    """
    Handle intent to list recent normal events.
    Accepts an optional 'number' entity to specify how many events to list.
    Defaults to 10 if not provided.
    """
    number = entities.get('number')
    try:
        limit = int(number) if number else 10
    except ValueError:
        logger.warning(f"Invalid number provided: {number}. Defaulting to 10.")
        limit = 10

    # Fetch recent normal events
    recent_normal_events = get_recent_events(metadata_store, event_type='normal', limit=limit)
    
    if recent_normal_events:
        event_list = "\n".join([f"- Event ID: {event['event_id']} at {event['timestamp']}" for event in recent_normal_events])
        final_message = f"Here are the last {limit} normal events:\n{event_list}"
    else:
        final_message = "No recent normal events found."

    return final_message

def handle_get_events_by_ip(entities: Dict, metadata_store: list, **kwargs) -> str:
    log_extracted_entities(entities)
    ip_address = entities.get('ip_address')
    if not ip_address:
        return "Please provide a valid IP address."

    text = entities.get('text', '').lower()
    if "source ip" in text:
        events = get_events_by_src_ip(ip_address, metadata_store)
    elif "destination ip" in text:
        events = get_events_by_dst_ip(ip_address, metadata_store)
    else:
        return "Please specify whether you are interested in source IP or destination IP."

    if events:
        return "\n".join([f"Event ID: {e['event_id']}" for e in events])
    else:
        return "No events found for the specified IP."

def handle_ask_who_are_you(entities: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    return "I am a network monitoring assistant designed to help you analyze events, attacks, and IP traffic."

def handle_ask_how_are_you(entities: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    return "I'm just a program, but thank you for asking! How can I help you?"

def handle_ask_help(entities: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    return ("I can help you with queries regarding network events, attacks, IP information, "
            "and interact with your OpenShift cluster for networking and security metrics.\n"
            "For example:\n"
            "- Get details about an event by its ID\n"
            "- List attack events\n"
            "- Find events related to a specific IP address\n"
            "- List network policies in a namespace\n"
            "- Check network traffic metrics\n"
            "- Review user access levels\n"
            "- And more!\n"
            "How can I assist you today?")

def handle_thank_you(entities: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    return "You're welcome! I'm here to help whenever you need me."

def handle_ask_farewell(entities: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    return "Goodbye! Looking forward to helping you again."

def handle_ask_joke(entities: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    return "Why don't computers get tired? Because they have chips to keep them going!"

def handle_ask_capabilities(entities: Dict, **kwargs) -> str:
    log_extracted_entities(entities)
    return ("I can help you with the following:\n"
            "- Look up network events by event ID\n"
            "- List events by IP address (source/destination)\n"
            "- Identify attack events\n"
            "- Provide IP-related event details\n"
            "- Interact with your OpenShift cluster for networking metrics\n"
            "- Monitor security policies and compliance\n"
            "How can I assist you today?")

def handle_general_question(entities: Dict, **kwargs) -> str:
    """
    Handle general technical questions by generating a response using the LLM.
    """
    log_extracted_entities(entities)
    user_text = entities.get('text', '')
    tokenizer = kwargs.get('tokenizer')
    llm_model = kwargs.get('llm_model')
    llm_model_type = kwargs.get('llm_model_type', 'seq2seq')

    if not all([user_text, tokenizer, llm_model]):
        logger.error("Missing components for generating LLM response.")
        return "Sorry, I couldn't process your request at the moment."

    # Generate a response using the LLM
    input_text = f"User asked: {user_text}\nPlease provide a helpful and accurate response."
    try:
        response_text = generate_response(
            input_text,
            tokenizer,
            llm_model,
            llm_model_type,
            max_context_length=512,
            max_answer_length=150
        )
        return response_text
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return "Sorry, I couldn't generate a response at the moment."

def handle_fallback(entities: Dict, **kwargs) -> str:
    """
    Handle fallback by generating a response using the LLM.
    """
    log_extracted_entities(entities)
    user_text = entities.get('text', '')
    tokenizer = kwargs.get('tokenizer')
    llm_model = kwargs.get('llm_model')
    llm_model_type = kwargs.get('llm_model_type', 'seq2seq')

    if not all([user_text, tokenizer, llm_model]):
        logger.error("Missing components for generating LLM response.")
        return "Sorry, I couldn't process your request at the moment."

    # Generate a response using the LLM
    input_text = f"User asked: {user_text}\nPlease provide a helpful and accurate response."
    try:
        response_text = generate_response(
            input_text,
            tokenizer,
            llm_model,
            llm_model_type,
            max_context_length=512,
            max_answer_length=150
        )
        return response_text
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return "Sorry, I couldn't generate a response at the moment."

# -------------------------------
# Networking Intent Handlers
# -------------------------------
def handle_list_network_policies(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle intent to list network policies in a specific namespace or cluster-wide.
    """
    namespace = extract_namespace(entities)
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": [],
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        # List policies in the specific namespace or across all namespaces if 'namespace' is not specified
        if namespace and namespace.lower() != 'all':
            query = f"list_network_policies(namespace='{namespace}')"
            policies_result = ocp_client.list_network_policies(namespace)
        else:
            query = "list_network_policies(all namespaces)"
            policies_result = ocp_client.list_network_policies()  # Fetch across all namespaces if 'namespace' is None or 'all'

        policy_names = policies_result.get("output", [])
        final_message = policies_result.get("final_message", "No network policies found.")

        return {
            "query": query,
            "output": policy_names,
            "final_message": final_message
        }

    except Exception as e:
        logger.error(f"Error fetching network policies: {e}")
        return {
            "query": "",
            "output": [],
            "final_message": f"Error retrieving network policies for namespace '{namespace if namespace else 'all'}'"
        }


def handle_check_network_traffic(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle intent to check current network traffic metrics.
    """
    log_extracted_entities(entities)
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": {},
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        traffic_metrics = ocp_client.check_network_traffic()

        # Check if 'throughput', 'latency', or 'packet_loss' are available
        if traffic_metrics and 'throughput' in traffic_metrics and 'packet_loss' in traffic_metrics:
            query = "Check network traffic metrics."
            output = traffic_metrics  # This is the metrics dictionary
            final_message = (
                f"Network Traffic Metrics:\n"
                f"Throughput: {traffic_metrics['throughput']}\n"
                f"Packet Loss: {traffic_metrics['packet_loss']}\n"
                f"TCP Retransmission: {traffic_metrics.get('tcp_retransmission', 'N/A')}"
            )
        else:
            # If any required metric is missing, log an error and respond accordingly
            logger.error("Missing required metrics from traffic metrics.")
            final_message = "No network traffic metrics available at the moment."

        return {
            "query": query if 'query' in locals() else "Check network traffic metrics.",
            "output": output if 'output' in locals() else {},
            "final_message": final_message
        }
    except Exception as e:
        logger.error(f"Error checking network traffic: {e}")
        return {
            "query": "",
            "output": {},
            "final_message": "Sorry, I couldn't retrieve network traffic metrics at the moment."
        }

def handle_list_services(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle the intent to list services in a specific namespace or across all namespaces.
    """
    namespace = extract_namespace(entities)
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": [],
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        # List services in the specific namespace or across all namespaces
        if namespace and namespace.lower() != 'all':
            query = f"list_services(namespace='{namespace}')"
            services_result = ocp_client.list_services(namespace)
        else:
            query = "list_services(all namespaces)"
            services_result = ocp_client.list_services()  # Fetch across all namespaces if 'namespace' is None or 'all'

        service_names = services_result.get("output", [])
        final_message = services_result.get("final_message", "No services found.")

        return {
            "query": query,
            "output": service_names,
            "final_message": final_message
        }

    except Exception as e:
        logger.error(f"Error listing services: {e}")
        return {
            "query": "",
            "output": [],
            "final_message": f"Error retrieving services for namespace '{namespace if namespace else 'all'}'"
        }


def handle_check_pod_connectivity(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle intent to check connectivity between two pods.
    """
    log_extracted_entities(entities)
    pod_a = entities.get('pod_a')
    pod_b = entities.get('pod_b')
    namespace = entities.get('namespace', 'default')
    ocp_client = kwargs.get('ocp_client')

    if not all([pod_a, pod_b]):
        return {
            "query": "",
            "output": [],
            "final_message": "Please provide both pod names to check connectivity."
        }

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": [],
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        connectivity = ocp_client.check_pod_connectivity(namespace, pod_a, pod_b)
        query = f"Check connectivity between {pod_a} and {pod_b} in namespace '{namespace}'."
        output = []
        final_message = (
            f"Pod '{pod_a}' can communicate with Pod '{pod_b}'."
            if connectivity.get('final_message').startswith("Pod")
            else f"Pod '{pod_a}' cannot communicate with Pod '{pod_b}'."
        )
        return {
            "query": query,
            "output": [pod_a, pod_b],
            "final_message": final_message
        }
    except Exception as e:
        logger.error(f"Error checking pod connectivity: {e}")
        return {
            "query": "",
            "output": [],
            "final_message": "Sorry, I couldn't verify pod connectivity at the moment."
        }

def handle_check_dns_health(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle intent to check DNS health.
    """
    log_extracted_entities(entities)
    namespace = entities.get('namespace', 'kube-system')
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": {},
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        dns_status = ocp_client.check_dns_health(namespace)
        query = f"Check DNS health in namespace '{namespace}'."
        output = dns_status
        final_message = (
            f"DNS Health: {dns_status['healthy']}. {dns_status['issues']}"
            if dns_status['issues'] else "DNS is healthy."
        )
        return {
            "query": query,
            "output": output,
            "final_message": final_message
        }
    except Exception as e:
        logger.error(f"Error checking DNS health: {e}")
        return {
            "query": "",
            "output": {},
            "final_message": "Sorry, I couldn't verify DNS health at the moment."
        }

# -------------------------------
# List Pods Intent Handler
# -------------------------------

def handle_list_pods(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle the intent to list pods in a given namespace or across all namespaces.
    """
    namespace = extract_namespace(entities)
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": [],
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        # List pods in the specific namespace or across all namespaces
        if namespace and namespace.lower() != 'all':
            query = f"list_pods(namespace='{namespace}')"
            pods_result = ocp_client.list_pods(namespace)
        else:
            query = "list_pods(all namespaces)"
            pods_result = ocp_client.list_pods()  # Fetch across all namespaces if 'namespace' is None or 'all'

        pod_names = pods_result.get("output", [])
        final_message = pods_result.get("final_message", "No pods found.")

        return {
            "query": query,
            "output": pod_names,
            "final_message": final_message
        }

    except Exception as e:
        logger.error(f"Error listing pods: {e}")
        return {
            "query": "",
            "output": [],
            "final_message": f"Error retrieving pods for namespace '{namespace if namespace else 'all'}'"
        }


# -------------------------------
# Security Intent Handlers
# -------------------------------

def handle_list_security_policies(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle intent to list security policies in a specific namespace or cluster-wide.
    """
    log_extracted_entities(entities)
    namespace = entities.get('namespace', 'all')
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": [],
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        policies = ocp_client.list_security_policies(namespace)
        query = policies.get("query", "")
        output = policies.get("output", [])
        final_message = policies.get("final_message", "No security policies found.")
        return {
            "query": query,
            "output": output,
            "final_message": final_message
        }
    except Exception as e:
        logger.error(f"Error fetching security policies: {e}")
        return {
            "query": "",
            "output": [],
            "final_message": "Sorry, I couldn't retrieve security policies at the moment."
        }

def handle_check_pod_security_compliance(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle intent to check pod security compliance.
    """
    log_extracted_entities(entities)
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": [],
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        non_compliant_pods = ocp_client.check_pod_security_compliance()
        query = "Check pod security compliance."
        output = non_compliant_pods
        if non_compliant_pods:
            final_message = f"Non-compliant pods: {', '.join(non_compliant_pods)}"
        else:
            final_message = "All pods comply with the security policies."
        return {
            "query": query,
            "output": output,
            "final_message": final_message
        }
    except Exception as e:
        logger.error(f"Error checking pod security compliance: {e}")
        return {
            "query": "",
            "output": [],
            "final_message": "Sorry, I couldn't verify pod security compliance at the moment."
        }

def handle_review_user_access(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle intent to review user access levels.
    """
    log_extracted_entities(entities)
    namespace = entities.get('namespace', 'all')
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": [],
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        user_access = ocp_client.review_user_access(namespace)
        query = f"Review user access in namespace '{namespace}'."
        output = user_access
        if user_access:
            response = f"User Access Information: {', '.join(user_access)}"
        else:
            response = f"No user access information found in namespace '{namespace}'."
        return {
            "query": query,
            "output": output,
            "final_message": response
        }
    except Exception as e:
        logger.error(f"Error reviewing user access: {e}")
        return {
            "query": "",
            "output": [],
            "final_message": "Sorry, I couldn't retrieve user access information at the moment."
        }

def handle_retrieve_audit_logs(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle intent to retrieve audit logs.
    """
    log_extracted_entities(entities)
    time_range = entities.get('time_range', 'last 24 hours')
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": [],
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        audit_logs = ocp_client.retrieve_audit_logs(time_range)
        query = f"Retrieve audit logs for {time_range}."
        output = audit_logs.get("output", []) if isinstance(audit_logs, dict) else []
        final_message = audit_logs.get("final_message", f"No security audit logs found for {time_range}.")
        return {
            "query": query,
            "output": output,
            "final_message": final_message
        }
    except Exception as e:
        logger.error(f"Error retrieving audit logs: {e}")
        return {
            "query": "",
            "output": [],
            "final_message": "Sorry, I couldn't retrieve audit logs at the moment."
        }

def handle_run_vulnerability_scan(entities: Dict, **kwargs) -> Dict[str, str]:
    """
    Handle intent to run a vulnerability scan.
    """
    log_extracted_entities(entities)
    ocp_client = kwargs.get('ocp_client')

    if not ocp_client:
        logger.error("OCP client not provided.")
        return {
            "query": "",
            "output": [],
            "final_message": "Internal error: Unable to access OpenShift cluster."
        }

    try:
        vulnerabilities = ocp_client.run_vulnerability_scan()
        query = "Run vulnerability scan in the cluster."
        output = vulnerabilities.get("output", []) if isinstance(vulnerabilities, dict) else []
        final_message = vulnerabilities.get("final_message", "No vulnerabilities detected in the cluster.")
        return {
            "query": query,
            "output": output,
            "final_message": final_message
        }
    except Exception as e:
        logger.error(f"Error running vulnerability scan: {e}")
        return {
            "query": "",
            "output": [],
            "final_message": "Sorry, I couldn't perform a vulnerability scan at the moment."
        }

# -------------------------------
# Mapping of Intents to Handlers
# -------------------------------

INTENT_HANDLERS = {
    # General Intents
    'greet': handle_greet,
    'goodbye': handle_goodbye,
    'get_event_info': handle_get_event_info,
    'list_attack_events': handle_list_attack_events,
    'list_recent_attack_events': handle_list_recent_attack_events,
    'list_recent_normal_events': handle_list_recent_normal_events,
    'get_events_by_ip': handle_get_events_by_ip,
    'ask_who_are_you': handle_ask_who_are_you,
    'ask_how_are_you': handle_ask_how_are_you,
    'ask_help': handle_ask_help,
    'thank_you': handle_thank_you,
    'ask_farewell': handle_ask_farewell,
    'ask_joke': handle_ask_joke,
    'ask_capabilities': handle_ask_capabilities,
    'general_question': handle_general_question,
    'fallback': handle_fallback,

    # Networking Intents
    'list_network_policies': handle_list_network_policies,
    'check_network_traffic': handle_check_network_traffic,
    'list_services': handle_list_services,
    'check_pod_connectivity': handle_check_pod_connectivity,
    'check_dns_health': handle_check_dns_health,

    # Security Intents
    'list_security_policies': handle_list_security_policies,
    'check_pod_security_compliance': handle_check_pod_security_compliance,
    'review_user_access': handle_review_user_access,
    'retrieve_audit_logs': handle_retrieve_audit_logs,
    'run_vulnerability_scan': handle_run_vulnerability_scan,

    # List pods
    'list_pods': handle_list_pods,
}