## NetSentinel

NetSentinel is a next-generation network intrusion detection system designed specifically for telecom environments. It combines predictive AI and generative AI (`google-flan-t5` or `Mistral-7B`, support both), observability, and agent-based architecture to report core network events. Deployed on Red Hat OpenShift, NetSentinel integrates multiple specialized agents and Slack-based chatbot functionality, allowing telecom operators to interact with the system in real-time.

## Key Components and Features:

- **Agent-Based Architecture:** NetSentinel’s modular design enables seamless scalability and customization, with four primary agents:

   - **NLU Agent:** Interprets human intent and extracts key information, enabling operators to engage with NetSentinel through natural language on Slack.
   - **Predictive Analysis and Generative Model:** Uses AI-powered classification to detect network anomalies and handle network-related queries, offering telecom-specific security insights.
   - **OpenShift API Agent:** Executes operational commands on OpenShift (list/create network policies, check pods compliance, etc), ensuring a swift response to network issues.
   - **Prometheus Agent:** Provides observability, running PromQL queries to monitor traffic and health metrics across RAN and core networks.

- **Slack Chatbot Integration:** NetSentinel’s chatbot allows telecom operators to ask questions like "List all attacks from the last hour" or "Is there suspicious activity from IP 192.168.1.1?" and receive immediate real-time responses.

### Demo Highlights:

- Real-time AI-driven network classification and anomaly detection
- Interactive Slack-based querying for seamless operator interaction
- Comprehensive observability and traffic monitoring via Prometheus and OpenShift integration

NetSentinel delivers an adaptable security solution for telecom providers, blending observability, predictive AI, and hands-on network management to protect RAN and core infrastructure.


## Deploy NetSentinel on OpenShift

### Create a new project 

Ensure that you update the namespace in the Kustomize file if you are using a namespace other than `netsentinel`. This adjustment may need to be made in several locations within the configuration.

For example, to create the `netsentinel` namespace, you can use:

```
oc new-project netsentinel
```

### Deploy Operators

The Kafka Operator is required for this setup to function properly. To deploy it, use the following command:

```
oc apply -k k8s/operators/overlays/rhlab/
```

Currently, we are using the `amq-streams-2.7.x` version. Older versions of Kafka exhibited different behavior, so it is important to use this version for consistency.


### Copy overlays
We are using Kustomize to deploy the NetSentinel application. To get started, copy the example overlays and modify them as needed:

```
cp -R k8s/apps/overlays/example k8s/apps/overlays/rhdemo-netsentinel
```

After copying, make the necessary adjustments to the new overlay files to match your environment and requirements.


### Create PVC
To patch the storageClassName for an existing PersistentVolumeClaim, use the following patch file `k8s/apps/overlays/rhdemo-netsentinel/pvc/storageclass-patch.yaml`. Update it with the desired storage class:

```
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: storageclass
spec:
  storageClassName: gp3-csi
```

To apply the patch, use the following command:

```
oc apply -k k8s/apps/overlays/rhdemo-netsentinel/pvc/
```

Make sure the `storageClassName` matches the desired storage class for your environment and that the patch file aligns with your PVC's namespace and name.


### Deploy kafka instance

To deploy the Kafka instance, follow these steps:

- Generate Secrets

```
./k8s/apps/base/kafka/generate-console-secrets.sh
mv console-ui-secrets.yaml ./k8s/apps/overlays/rhdemo-netsentinel/kafka/.
```

- Update Cluster DNS

Replace `<CLUSTER_NAME_WITH_BASE_DOMAIN>` with your cluster's DNS name. Ensure the DNS is:

  - Publicly resolvable.
  - Not using a self-signed certificate. Certificates must be valid.

> Note: This is required for communication with Slack channels.
If deploying in an OpenShift cluster where the DNS is not publicly resolvable and uses self-signed certificates, you can use tools like ngrok as a workaround. Refer to `k8s/apps/overlays/telcolab` for an example of this approach.

- Apply Kafka Configuration

Deploy the Kafka instance using the following command:

```
oc apply -k k8s/apps/overlays/rhdemo-netsentinel/kafka/
```

- Wait for Kafka to Start

It may take some time for Kafka to be fully operational. The `CreateContainerConfigError` status for certain pods (e.g., Kafka console) will resolve automatically once kafkausers are created and the necessary secrets are available.

Check the pods status 

```
oc get pods
```

Example output during initialization:

```
NAME                        READY   STATUS                       RESTARTS   AGE
console-5c498fb9c4-ffm6v    1/2     CreateContainerConfigError   0          67s
console-kafka-kafka-0       1/1     Running                      0          22s
console-kafka-kafka-1       0/1     Running                      0          22s
console-kafka-kafka-2       0/1     Running                      0          22s
console-kafka-zookeeper-0   1/1     Running                      0          57s
console-kafka-zookeeper-1   1/1     Running                      0          57s
console-kafka-zookeeper-2   1/1     Running                      0          57s
```

- Verify Kafka Users

```
oc get kafkausers
NAME                     CLUSTER         AUTHENTICATION   AUTHORIZATION   READY
console-kafka-user1      console-kafka   scram-sha-512    simple          True
netsentinel-kafka-user   console-kafka   scram-sha-512    simple          True
```

- Confirm All Pods are Running

After a few minutes, verify that all pods are running as expected:


```
Balkrishnas-MacBook-Pro:NetSentinel bpandey$ oc get pods
NAME                                             READY   STATUS    RESTARTS   AGE
console-5c498fb9c4-ffm6v                         2/2     Running   0          2m39s
console-kafka-entity-operator-74f8599b68-mmrq6   2/2     Running   0          81s
console-kafka-kafka-0                            1/1     Running   0          114s
console-kafka-kafka-1                            1/1     Running   0          114s
console-kafka-kafka-2                            1/1     Running   0          114s
console-kafka-zookeeper-0                        1/1     Running   0          2m29s
console-kafka-zookeeper-1                        1/1     Running   0          2m29s
console-kafka-zookeeper-2                        1/1     Running   0          2m29s
```

Your Kafka instance is now ready to use!


### Deploy NetSentinel Application
```
oc apply -k k8s/apps/overlays/telcolab/netsentinel/
```