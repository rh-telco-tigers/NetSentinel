import os
import logging
from typing import List, Dict
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from prometheus_api_client import PrometheusConnect
from kubernetes.client import V1NetworkPolicy, V1ObjectMeta, V1NetworkPolicySpec, V1LabelSelector, V1NetworkPolicyIngressRule, V1NetworkPolicyPeer, V1IPBlock


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OCPClient:
    def __init__(self, kubeconfig_path: str = None, prometheus_url: str = None):
        """
        Initialize the OCP Client using kubeconfig or in-cluster configuration.
        Also initializes the Prometheus client for fetching metrics.
        """
        try:
            # Authentication
            if kubeconfig_path and os.path.exists(kubeconfig_path):
                config.load_kube_config(config_file=kubeconfig_path)
                logger.info(f"Loaded kubeconfig from {kubeconfig_path}")
            else:
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            
            # Initialize Kubernetes APIs
            self.network_api = client.NetworkingV1Api()
            self.rbac_api = client.RbacAuthorizationV1Api()
            self.core_api = client.CoreV1Api()
            self.apps_api = client.AppsV1Api()
            
            # Initialize Prometheus client with Bearer Token
            if prometheus_url:
                token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
                
                if os.path.exists(token_path):
                    with open(token_path, 'r') as token_file:
                        bearer_token = token_file.read().strip()
                    headers = {"Authorization": f"Bearer {bearer_token}"}
                    
                    self.prom = PrometheusConnect(url=prometheus_url, headers=headers, disable_ssl=True)
                    logger.info(f"Connected to Prometheus at {prometheus_url}")
                else:
                    # Allow setting a manual token if running outside of the cluster
                    bearer_token = os.getenv('PROMETHEUS_BEARER_TOKEN', None)
                    if bearer_token:
                        headers = {"Authorization": f"Bearer {bearer_token}"}
                        self.prom = PrometheusConnect(url=prometheus_url, headers=headers, disable_ssl=True)
                        logger.info(f"Connected to Prometheus at {prometheus_url} using a manually provided token.")
                    else:
                        logger.error("Bearer token not found. Unable to authenticate to Prometheus.")
                        self.prom = None
            else:
                logger.warning("Prometheus URL not provided. Network metrics will be unavailable.")
                self.prom = None

        except Exception as e:
            logger.error(f"Failed to initialize OCP Client: {e}")
            raise e

    def test_prometheus_connection(self):
        """
        Test if the connection to Prometheus is working by running a simple query.
        """
        if not self.prom:
            logger.error("Prometheus client not initialized. Cannot perform test query.")
            return {
                "query": "Prometheus client not initialized",
                "output": {},
                "final_message": "Prometheus client not initialized. Cannot perform test query."
            }

        try:
            test_query = 'up'
            test_result = self.prom.custom_query(query=test_query)
            
            if test_result:
                logger.info(f"Test query successful. Result: {test_result}")
                return {
                    "query": test_query,
                    "output": test_result,
                    "final_message": "Successfully connected to Prometheus and retrieved metrics."
                }
            else:
                logger.warning("Test query returned no results.")
                return {
                    "query": test_query,
                    "output": test_result,
                    "final_message": "Test query returned no results."
                }
        except Exception as e:
            logger.error(f"Error performing test query on Prometheus: {e}")
            return {
                "query": test_query,
                "output": {},
                "final_message": f"Error performing test query: {e}"
            }

    # -------------------------
    # Networking-Related Functions
    # -------------------------

    def list_network_policies(self, namespace: str = None) -> Dict[str, str]:
        """
        List all network policies in a given namespace or across the cluster.
        Returns a dictionary with 'query', 'output', and 'final_message'.
        """
        try:
            if namespace and namespace.lower() != 'all':
                query = f"list_namespaced_network_policy(namespace='{namespace}')"
                policies = self.network_api.list_namespaced_network_policy(namespace)
            else:
                query = "list_network_policies_for_all_namespaces()"
                policies = self.network_api.list_network_policy_for_all_namespaces()

            # Access policies.items, which is an iterable
            policy_names = [f"{policy.metadata.namespace}/{policy.metadata.name}" for policy in policies.items]
            logger.debug(f"Network policies: {policy_names}")

            # Prepare the final response message
            if policy_names:
                if namespace and namespace.lower() != 'all':
                    final_message = f"Active network policies in namespace '{namespace}':\n" + "\n".join(policy_names)
                else:
                    final_message = "Active network policies across the cluster:\n" + "\n".join(policy_names)
            else:
                if namespace and namespace.lower() != 'all':
                    final_message = f"No network policies found in namespace '{namespace}'."
                else:
                    final_message = "No network policies found across the cluster."

            return {
                "query": query,
                "output": policy_names,
                "final_message": final_message
            }

        except ApiException as e:
            logger.error(f"API exception when listing network policies: {e}")
            return {
                "query": query,
                "output": [],
                "final_message": f"Error retrieving network policies: {e}"
            }
        except Exception as e:
            logger.error(f"Unexpected error when listing network policies: {e}")
            return {
                "query": query,
                "output": [],
                "final_message": f"Unexpected error: {e}"
            }


    def generate_network_policy_yaml(self, namespace: str, source_ips: List[str], destination_ips: List[str]) -> Dict[str, str]:
        import yaml

        try:
            # Construct the NetworkPolicy object
            network_policy = V1NetworkPolicy(
                api_version='networking.k8s.io/v1',
                kind='NetworkPolicy',
                metadata=V1ObjectMeta(
                    name='block-traffic-policy',
                    namespace=namespace
                ),
                spec=V1NetworkPolicySpec(
                    pod_selector=V1LabelSelector(match_labels={}),  # Empty selector matches all pods
                    policy_types=['Ingress'],
                    ingress=[]
                )
            )

            if source_ips:
                ip_block = V1IPBlock(
                    cidr='0.0.0.0/0'
                )
                # Since 'except' is a reserved keyword, use setattr to set it
                setattr(ip_block, 'except', [ip + '/32' for ip in source_ips])

                ingress_rule = V1NetworkPolicyIngressRule(
                    _from=[
                        V1NetworkPolicyPeer(
                            ip_block=ip_block
                        )
                    ]
                )
                network_policy.spec.ingress.append(ingress_rule)
            else:
                # Deny all ingress traffic
                network_policy.spec.ingress = []
                network_policy.spec.policy_types = ['Ingress']

            if destination_ips:
                # Since NetworkPolicy cannot specify destination IPs directly unless they are pod IPs,
                # we need to inform the user that this is not possible
                raise ValueError("NetworkPolicy cannot specify destination IPs unless they correspond to pod IPs.")

            # Attempt to apply the policy
            try:
                self.network_api.create_namespaced_network_policy(
                    namespace=namespace,
                    body=network_policy
                )
                message = f"NetworkPolicy 'block-traffic-policy' has been successfully applied to namespace '{namespace}'."
                applied = True
            except ApiException as e:
                if e.status == 403:
                    message = ("You do not have permission to apply this NetworkPolicy due to RBAC restrictions. "
                            "You can however create the NetworkPolicy using the following YAML:\n"
                            "```yaml\n{yaml}\n```")
                    applied = False
                else:
                    message = f"An error occurred while applying the NetworkPolicy: {e}"
                    applied = False

            # Convert the policy to YAML
            api_client = client.ApiClient()
            policy_dict = api_client.sanitize_for_serialization(network_policy)
            network_policy_yaml = yaml.dump(policy_dict, sort_keys=False)

            if not applied and '{yaml}' in message:
                message = message.format(yaml=network_policy_yaml)

            return {
                "applied": applied,
                "message": message,
                "yaml": network_policy_yaml
            }

        except ValueError as ve:
            logger.error(f"Value error: {ve}")
            raise ve
        except Exception as e:
            message = f"An unexpected error occurred: {e}"
            logger.error(message)
            raise e


    def check_network_traffic(self) -> Dict[str, str]:
        """
        Check current network traffic metrics like throughput and packet loss.
        Returns the query executed, its output, and the final message.
        """
        if not self.prom:
            logger.error("Prometheus client not initialized")
            return {
                "query": "Prometheus client not initialized",
                "output": {},
                "final_message": "Prometheus client not initialized. Cannot fetch network traffic metrics."
            }

        try:
            # Prometheus queries
            throughput_query = 'sum(rate(container_network_receive_bytes_total[5m])) + sum(rate(container_network_transmit_bytes_total[5m]))'
            packet_loss_query = 'sum(rate(container_network_receive_errors_total[5m])) / sum(rate(container_network_receive_bytes_total[5m])) * 100'
            tcp_retransmission_query = 'sum(rate(node_netstat_Tcp_RetransSegs[5m])) / sum(rate(node_netstat_Tcp_OutSegs[5m])) * 100'

            # Execute queries
            throughput = self.prom.custom_query(query=throughput_query)
            packet_loss = self.prom.custom_query(query=packet_loss_query)
            tcp_retransmission = self.prom.custom_query(query=tcp_retransmission_query)

            # Log the raw results
            logger.info(f"Throughput query raw result: {throughput}")
            logger.info(f"Packet loss query raw result: {packet_loss}")
            logger.info(f"TCP retransmission query raw result: {tcp_retransmission}")

            # Prepare metrics dictionary
            metrics = {}

            # Process throughput
            if throughput and len(throughput) > 0 and 'value' in throughput[0]:
                metrics['throughput'] = f"{float(throughput[0]['value'][1]) / 1_000_000:.2f} Mbps"  # Convert to Mbps
            else:
                logger.error("Throughput query returned no valid data.")
                metrics['throughput'] = "N/A"

            # Process packet loss
            if packet_loss and len(packet_loss) > 0 and 'value' in packet_loss[0]:
                metrics['packet_loss'] = f"{float(packet_loss[0]['value'][1]):.2f}%"
            else:
                logger.error("Packet loss query returned no valid data.")
                metrics['packet_loss'] = "N/A"

            # Process TCP retransmission
            if tcp_retransmission and len(tcp_retransmission) > 0 and 'value' in tcp_retransmission[0]:
                metrics['tcp_retransmission'] = f"{float(tcp_retransmission[0]['value'][1]):.2f}%"
            else:
                logger.error("TCP retransmission query returned no valid data.")
                metrics['tcp_retransmission'] = "N/A"

            # Log final metrics
            logger.debug(f"Final network metrics: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"Error fetching network traffic metrics: {e}")
            raise e


    def list_services(self, namespace: str = None) -> Dict[str, str]:
        """
        List services in a given namespace or across all namespaces.
        """
        try:
            if namespace and namespace.lower() != 'all':
                query = f"list_namespaced_service(namespace='{namespace}')"
                services = self.core_api.list_namespaced_service(namespace)
            else:
                query = "list_service_for_all_namespaces()"
                services = self.core_api.list_service_for_all_namespaces()

            service_names = [f"{service.metadata.namespace}/{service.metadata.name}" for service in services.items]
            logger.debug(f"Services: {service_names}")

            return {
                "query": query,
                "output": service_names,
                "final_message": f"Services: {', '.join(service_names) if service_names else 'No services found'}"
            }
        except ApiException as e:
            logger.error(f"API exception when listing services: {e}")
            return {
                "query": "",
                "output": [],
                "final_message": f"Error retrieving services: {e}"
            }
        except Exception as e:
            logger.error(f"Unexpected error when listing services: {e}")
            return {
                "query": "",
                "output": [],
                "final_message": f"Unexpected error: {e}"
            }


    def list_pods(self, namespace: str = None) -> Dict[str, str]:
        """
        List pods in a given namespace or across all namespaces.
        """
        try:
            if namespace and namespace.lower() != 'all':
                query = f"list_namespaced_pod(namespace='{namespace}')"
                pods = self.core_api.list_namespaced_pod(namespace)
            else:
                query = "list_pod_for_all_namespaces()"
                pods = self.core_api.list_pod_for_all_namespaces()

            pod_names = [f"{pod.metadata.namespace}/{pod.metadata.name}" for pod in pods.items]
            logger.debug(f"Pods: {pod_names}")

            return {
                "query": query,
                "output": pod_names,
                "final_message": f"Pods: {', '.join(pod_names) if pod_names else 'No pods found'}"
            }
        except ApiException as e:
            logger.error(f"API exception when listing pods: {e}")
            return {
                "query": "",
                "output": [],
                "final_message": f"Error retrieving pods: {e}"
            }
        except Exception as e:
            logger.error(f"Unexpected error when listing pods: {e}")
            return {
                "query": "",
                "output": [],
                "final_message": f"Unexpected error: {e}"
            }


    def check_pod_connectivity(self, namespace: str, pod_a: str, pod_b: str) -> Dict[str, str]:
        """
        Check connectivity between two pods using exec commands.
        Returns the query executed, its output, and the final message.
        """
        try:
            # Execute ping from pod_a to pod_b
            query = f"Ping from {pod_a} to {pod_b}"
            exec_command = [
                '/bin/sh',
                '-c',
                f'ping -c 3 {pod_b}'
            ]
            resp = self.core_api.connect_get_namespaced_pod_exec(
                name=pod_a,
                namespace=namespace,
                command=exec_command,
                stderr=True, stdin=False,
                stdout=True, tty=False,
                _preload_content=False
            )

            success = False
            output = ""
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stdout():
                    output = resp.read_stdout()
                    logger.debug(f"Ping Output: {output}")
                    if "0% packet loss" in output:
                        success = True
                if resp.peek_stderr():
                    error = resp.read_stderr()
                    logger.error(f"Ping Error: {error}")
            final_message = f"Pod '{pod_a}' can communicate with Pod '{pod_b}'." if success else f"Pod '{pod_a}' cannot communicate with Pod '{pod_b}'."
            
            return {
                "query": query,
                "output": output,
                "final_message": final_message
            }
        except ApiException as e:
            logger.error(f"API exception when checking pod connectivity: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error when checking pod connectivity: {e}")
            raise e

    def check_dns_health(self, namespace: str = 'kube-system') -> Dict[str, str]:
        """
        Check DNS health by verifying the status of CoreDNS pods.
        Returns the query executed, its output, and the final message.
        """
        try:
            query = f"list_namespaced_pod(namespace={namespace}, label_selector='k8s-app=kube-dns')"
            pods = self.core_api.list_namespaced_pod(namespace=namespace, label_selector='k8s-app=kube-dns')
            coredns_pods = [pod for pod in pods.items if 'coredns' in pod.metadata.name.lower()]
            healthy = all(pod.status.phase == 'Running' for pod in coredns_pods)
            issues = []
            if not healthy:
                issues.append("Some CoreDNS pods are not running.")
            metrics = {
                'healthy': 'Yes' if healthy else 'No',
                'issues': "; ".join(issues) if issues else "No issues detected."
            }
            final_message = f"DNS Health: {metrics['healthy']}. {metrics['issues']}" if metrics['issues'] else "DNS is healthy."

            return {
                "query": query,
                "output": metrics,
                "final_message": final_message
            }
        except ApiException as e:
            logger.error(f"API exception when checking DNS health: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error when checking DNS health: {e}")
            raise e

    # -------------------------
    # Security Functions
    # -------------------------

    def list_security_policies(self, namespace: str = None) -> Dict[str, str]:
        """
        List security policies (e.g., PodSecurityPolicies) in a given namespace or cluster-wide.
        Returns the query executed, its output, and the final message.
        """
        try:
            if namespace and namespace.lower() != 'all':
                query = f"list_namespaced_role_binding(namespace={namespace})"
                role_bindings = self.rbac_api.list_namespaced_role_binding(namespace=namespace)
            else:
                query = "list_cluster_role_binding()"
                role_bindings = self.rbac_api.list_cluster_role_binding()

            policy_names = [rb.metadata.name for rb in role_bindings.items]
            logger.debug(f"Security policies: {policy_names}")

            return {
                "query": query,
                "output": policy_names,
                "final_message": f"Security policies: {', '.join(policy_names) if policy_names else 'No security policies found'}"
            }
        except ApiException as e:
            logger.error(f"API exception when listing security policies: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error when listing security policies: {e}")
            raise e

    def check_pod_security_compliance(self) -> Dict[str, str]:
        """
        Check pods for security compliance based on security policies.
        Returns the query executed, its output, and the final message.
        """
        try:
            query = "list_pod_for_all_namespaces()"
            pods = self.core_api.list_pod_for_all_namespaces()
            non_compliant_pods = []
            for pod in pods.items:
                for container in pod.spec.containers:
                    if container.security_context and container.security_context.privileged:
                        non_compliant_pods.append(f"{pod.metadata.namespace}/{pod.metadata.name}")
                        break  # Avoid duplicate entries for the same pod
            logger.debug(f"Non-compliant Pods: {non_compliant_pods}")

            final_message = f"Non-compliant pods: {', '.join(non_compliant_pods)}" if non_compliant_pods else "All pods comply with the security policies."

            return {
                "query": query,
                "output": non_compliant_pods,
                "final_message": final_message
            }
        except ApiException as e:
            logger.error(f"API exception when checking pod security compliance: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error when checking pod security compliance: {e}")
            raise e

    def review_user_access(self, namespace: str = None) -> Dict[str, str]:
        """
        Review user roles and permissions within a namespace or cluster-wide.
        Returns the query executed, its output, and the final message.
        """
        try:
            if namespace and namespace.lower() != 'all':
                query = f"list_namespaced_role_binding(namespace={namespace})"
                role_bindings = self.rbac_api.list_namespaced_role_binding(namespace=namespace)
            else:
                query = "list_cluster_role_binding()"
                role_bindings = self.rbac_api.list_cluster_role_binding()

            access_info = []
            for rb in role_bindings.items:
                role_name = rb.role_ref.name
                subjects = [subject.name for subject in rb.subjects]
                access_info.append(f"Role '{role_name}': {', '.join(subjects)}")

            logger.debug(f"User Access Information: {access_info}")

            return {
                "query": query,
                "output": access_info,
                "final_message": f"User Access Information: {', '.join(access_info)}" if access_info else "No user access information found."
            }
        except ApiException as e:
            logger.error(f"API exception when reviewing user access: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error when reviewing user access: {e}")
            raise e

    def retrieve_audit_logs(self, time_range: str = 'last 24 hours') -> Dict[str, str]:
        """
        Retrieve recent security audit logs.
        Returns the query executed, its output, and the final message.
        """
        try:
            # Placeholder for real query
            query = f"Retrieve logs for {time_range}"
            # Simulate log retrieval
            audit_logs = [
                "2024-09-24T14:00:00Z - User john.doe accessed pod frontend.",
                "2024-09-24T14:05:00Z - User jane.smith attempted unauthorized access.",
                "2024-09-24T14:10:00Z - Security policy violation detected in namespace 'production'."
            ]
            logger.debug(f"Audit Logs Retrieved: {audit_logs}")

            final_message = f"Audit logs for {time_range}: {', '.join(audit_logs)}"

            return {
                "query": query,
                "output": audit_logs,
                "final_message": final_message
            }
        except Exception as e:
            logger.error(f"Error retrieving audit logs: {e}")
            raise e

    def run_vulnerability_scan(self) -> Dict[str, str]:
        """
        Perform a vulnerability scan within the cluster.
        Returns the query executed, its output, and the final message.
        """
        try:
            # Placeholder for real query
            query = "Vulnerability scan"
            # Simulate a vulnerability scan
            vulnerabilities = [
                "Vulnerability CVE-2024-XXXX in image nginx:1.19.0",
                "Vulnerability CVE-2024-YYYY in image redis:6.0.9"
            ]
            logger.debug(f"Vulnerabilities Found: {vulnerabilities}")

            final_message = f"Vulnerabilities found: {', '.join(vulnerabilities)}"

            return {
                "query": query,
                "output": vulnerabilities,
                "final_message": final_message
            }
        except Exception as e:
            logger.error(f"Error running vulnerability scan: {e}")
            raise e
