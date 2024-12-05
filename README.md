### NetSentinel

NetSentinel is an AI-powered chatbot platform integrated with Slack, showcasing the potential of predictive analysis, generative AI and Natural Language Understanding (NLU) for advanced use cases. Designed as a versatile proof of concept, it demonstrates attack detection and response using mock data. While initially focused on network intrusion detection, and communication with OCP API and Prometheus API, its flexible architecture supports use cases across industries, from customer support to billing inquiries.

NetSentinel's future vision includes integrating human agents into workflows, allowing users to escalate queries that the AI cannot answer. Agents could range from document processors to knowledge base managers or balance inquiry handlers, showcasing the system's adaptability to various operational needs.

### Key Components and Features:

- **Generative and Predictive AI:** Powered by `granite-8b-code-instruct-128k`, the system identifies and classifies potential anomalies while handling complex queries.

- **Slack Chatbot Integration:** Users can interact with the chatbot through Slack, asking queries like "List all network events identified as attacks" or "Check network traffic metrics" and receiving instant, AI-driven or API-driven responses.

- **Future-Ready Human-in-the-Loop Design:** Though not currently implemented, the architecture envisions seamless escalation to human agents for cases requiring specialized intervention.

- **Multi-Agent Architecture:**
  - **NLU Agent:** Processes natural language queries to extract user intent and actionable data.
  - **Predictive and Generative Agents:** Detect anomalies and respond to network or general queries.
  - **OpenShift API Agent:** Executes operational commands like listing network policies or checking pod compliance in an OpenShift environment.
  - **Prometheus Agent:** Enables observability, running PromQL queries to monitor metrics.

### Use Case Versatility:

NetSentinel's design extends beyond telecom. The framework can be adapted to create:

- Customer support chatbots
- Billing inquiry handlers
- Document processing agents
- Knowledge base assistants
- And more, with minimal customization.

### Demo Highlights:

- Showcases AI-driven attack detection using simulated data
- Demonstrates interactive Slack-based querying for various scenarios
- Highlights adaptability for industry-specific use cases, such as customer service or network operations

NetSentinel offers a glimpse into the future of AI-enhanced operational workflows, emphasizing adaptability, scalability, and real-time response for diverse applications.

## Order OpenShift Environment

- Any OpenShift environment should work technically, provided there are no operator conflicts. To avoid issues, it’s recommended to start with a clean environment since the project requires installing multiple operators and configurations.
- For testing purposes, we are using the following environment.
  - Order an OCP demo cluster via this [URL](https://catalog.demo.redhat.com/catalog?item=babylon-catalog-prod/sandboxes-gpte.ocp-wksp.prod&utm_source=webapp&utm_medium=share-link)
  - Select **OpenShift Version 4.16** during setup.
  - Only a single control plane is sufficient.
  - If you are using **Model as a Service** for the LLM model, a CPU-only setup is adequate for deploying this project.

## Clone NetSentinel locally

```
git clone git@github.com:pandeybk/NetSentenial.git
cd NetSentenial
```

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

> Note: Currently, we are using the `amq-streams-2.7.x` version. Older versions of Kafka exhibited different behavior, so it is important to use this version for consistency.
> Note: The demo environment we are using above already contains cert manager operator. If you are using different demo environment you may have to install this operator separtely. Check following files to include this operator as well, `k8s/operators/overlays/common/kustomization.yaml` and `k8s/instances/overlays/common/kustomization.yaml`
> Note: Ensure all operators are active and running before proceeding. Navigate to Operators > Installed Operators in the OpenShift Console to verify their status.
> Note: Verify that the Red Hat OpenShift AI Operator is properly configured and that all components are active and functioning correctly. You can check their status on the Operator page under "All Instances" in the OpenShift Console.

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

Ensure the following files have been modified as expected:

```
git status
On branch cleanup
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   k8s/instances/overlays/rhlab/kafka/patches/console-kafka.kafka.yaml
	modified:   k8s/instances/overlays/rhlab/kafka/patches/console-ui.route.yaml
	modified:   k8s/instances/overlays/rhlab/kafka/patches/console.deployment.yaml

```

To further validate the changes, you can run the following command:

```
git diff .
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
oc get pods -n netsentinel
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
oc get kafkausers -n netsentinel
NAME                     CLUSTER         AUTHENTICATION   AUTHORIZATION   READY
console-kafka-user1      console-kafka   scram-sha-512    simple          True
netsentinel-kafka-user   console-kafka   scram-sha-512    simple          True
```

- Confirm All Pods are Running

After a few minutes, verify that all pods are running as expected:

```
oc get pods -n netsentinel

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
URL=$(oc get routes console-ui-route -o jsonpath='{.spec.host}' -n netsentinel)
echo "https://$URL"
open "https://$URL"
```

### 4. Upload model to s3 bucket

Follow guide [Upload model to buckets](./docs/upload-models-to-bucket.md)

### 5. Create a New API Token for "Models as a Service" on OpenShift AI

Follow guide [Model as a service](./docs/model-as-a-service.md)

### 6. Deploy NetSentinel Application

This process configures the "NVIDIA Triton Inference Server" using OpenShift ServingRuntime and deploys the predictive model in `netsentinel` namespace. Note that there are hardcoded references to MinIO object storage and specific paths, so ensure the model is available in the correct MinIO location by following Step 4. This step also deploys the NetSentinel application along with its components, including the mock data generator, mock data processor, and prediction service. If the model is not properly deployed, the installation will fail.

Also, ensure that `<YOUR_API_KEY_HERE>` is replaced with the actual API key in the `models.llm.token` section of the file `k8s/apps/overlays/rhlab/netsentinel/app-config.yaml` as part of Step 5. The URL and model name should remain consistent across all MaaS-deployed services. If they differ, adjust those values accordingly to maintain consistency.

Now execute following.

```
oc apply -k  k8s/apps/overlays/rhlab/
```

Validate:

```
oc get pods -n netsentinel
```

Output:

```
NAME                                             READY   STATUS              RESTARTS   AGE
console-68c9df5c59-6qfff                         2/2     Running             0          16m
console-kafka-entity-operator-7fc848f9fc-bnncl   2/2     Running             0          15m
console-kafka-kafka-0                            1/1     Running             0          16m
console-kafka-kafka-1                            1/1     Running             0          16m
console-kafka-kafka-2                            1/1     Running             0          16m
console-kafka-zookeeper-0                        1/1     Running             0          16m
console-kafka-zookeeper-1                        1/1     Running             0          16m
console-kafka-zookeeper-2                        1/1     Running             0          16m
create-mock-data-98df85fdc-7kxx7                 0/1     ContainerCreating   0          31s
minio-6c95767dc6-642zv                           1/1     Running             0          24m
netsentinel-5d9896664b-twblx                     0/1     ContainerCreating   0          31s
prediction-service-d6b56cd88-2bd2r               0/1     ContainerCreating   0          32s
process-mock-data-575bd9bd64-5vdf5               0/1     ContainerCreating   0          31s
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
