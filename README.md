## NetSentinel

NetSentinel is a next-generation network intrusion detection system designed specifically for telecom environments. It combines predictive AI and generative AI (`google-flan-t5` or `Mistral-7B`, support both), observability, and agent-based architecture to report core network events. Currently a proof of concept using simulated data, NetSentinel is designed to integrate with tools like Zeek and Suricata in the future. Deployed on Red Hat OpenShift, NetSentinel integrates multiple specialized agents and Slack-based chatbot functionality, allowing telecom operators to interact with the system in real-time.

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

## Order OpenShift Environment

- Any OpenShift environment should work technically, provided there are no operator conflicts. To avoid issues, it’s recommended to start with a clean environment since the project requires installing multiple operators and configurations.
- For testing purposes, we are using the following environment.
  - Order an OCP demo cluster via this [URL](https://catalog.demo.redhat.com/catalog?item=babylon-catalog-prod/sandboxes-gpte.ocp-wksp.prod&utm_source=webapp&utm_medium=share-link)
  - Select **OpenShift Version 4.16** during setup.
  - Only a single control plane is sufficient.
  - If you are using **Model as a Service** for the LLM model, a CPU-only setup is adequate for deploying this project.

## Deploy NetSentinel on OpenShift

### 1. Create new openshift projects

Apply the pre-defined namespace configurations using Kustomize:

```
kubectl apply -k k8s/namespaces/base
```

Ensure the namespace in the Kustomize configuration matches your desired namespace (e.g., `netsentinel`). Update it in all relevant locations if needed.

### 2. Deploy Operators

```
oc apply -k k8s/operators/overlays/common
```

Currently, we are using the `amq-streams-2.7.x` version. Older versions of Kafka exhibited different behavior, so it is important to use this version for consistency.

### 3. Deploy Instances of Operators

#### Deploy common opertors

```
oc apply -k k8s/instances/overlays/common
```

#### Deploy kafka instance

To deploy the Kafka instance, follow these steps:

- Update Cluster DNS

Replace `<CLUSTER_NAME_WITH_BASE_DOMAIN>` with your cluster's DNS name.

Example you can run following command

```
find ./k8s/instances/overlays/rhlab/kafka/ -type f -exec sed -i '' 's/<CLUSTER_NAME_WITH_BASE_DOMAIN>/cluster-bbgs4.bbgs4.sandbox592.opentlc.com/g' {} +
```

> Ensure the DNS is:
> Publicly resolvable.
> Domain is Not using a self-signed certificate. Certificates must be valid.
> Note: This is required for communication with Slack channels.
> If deploying in an OpenShift cluster where the DNS is not publicly resolvable and uses self-signed certificates, you can use tools like ngrok as a workaround.

- Apply Kafka Configuration

Deploy the Kafka instance using the following command:

```
oc apply -k k8s/instances/overlays/rhlab/
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

Your Kafka instance is now ready to use. You can browse kafka console using following url.

```
URL=$(oc get routes console-ui-route -o jsonpath='{.spec.host}')
echo "https://$URL"
open "https://$URL"
```

### 4. Upload model to s3 bucket

Follow guide [Upload model to buckets](./docs/upload-models-to-bucket.md)

### 5. Create a New API Token for "Models as a Service" on OpenShift AI

Follow guide [Model as a service](./docs/model-as-a-service.md)

### 6. Deploy NetSentinel Application

This process configures the Titan model server using OpenShift ServingRuntime and deploys the predictive model in `netsentinel` namespace. Note that there are hardcoded references to MinIO object storage and specific paths, so ensure the model is available in the correct MinIO location by following Step 4. This step also deploys the NetSentinel application along with its components, including the mock data generator, mock data processor, and prediction service. If the model is not properly deployed, the installation will fail.

Also, ensure that `<YOUR_API_KEY_HERE>` is replaced with the actual API key in the `models.llm.token` section of the file `k8s/apps/overlays/rhlab/netsentinel/app-config.yaml` as part of Step 5. The URL and model name should remain consistent across all MaaS-deployed services. If they differ, adjust those values accordingly to maintain consistency.

Now execute following.

```
oc apply -k  k8s/apps/overlays/rhlab/
```

### 6. Configure SLACK for communication with the bot

Follow doc [Slack Configuration](./docs/configure-slack.md)

## Cleanup

Execute the commands in the specified sequence to ensure proper deletion, as Kafka topics may not be deleted if the order is not followed:

```
oc delete -k k8s/apps/overlays/rhlab/netsentinel/
oc delete -k k8s/instances/overlays/common
oc delete kafkatopics --all -n netsentinel
oc delete -k k8s/instances/overlays/rhlab

oc delete pvc --all -n netsentinel

oc delete deployment --all -n milvus-operator
oc delete sts --all -n milvus-operator
oc delete pvc --all -n milvus-operator
oc delete -k k8s/namespaces/base
```
