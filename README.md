### NetSentinel

NetSentinel is an AI-powered chatbot platform integrated with Slack, showcasing the potential of predictive analysis, generative AI and Natural Language Understanding (NLU) for advanced use cases. Designed as a versatile proof of concept, it demonstrates attack detection and response using mock data. While initially focused on network intrusion detection, and communication with OCP API and Prometheus API, its flexible architecture supports use cases across industries, from customer support to billing inquiries.

NetSentinel's future vision includes integrating human agents into workflows, allowing users to escalate queries that the AI cannot answer. Agents could range from document processors to knowledge base managers or balance inquiry handlers, showcasing the system's adaptability to various operational needs.

![NetSentinel Architecture](./docs/images/netsentinel-architecture.jpg)

### Demo Video
To see NetSentinel in action, check out the [demo video on YouTube](https://youtu.be/jxnB854MqH0?si=vYtpjuv-ogjInwet). The video showcases the platform's capabilities, including real-time query handling through Slack, AI-powered attack detection, and integration with OpenShift components. It provides a step-by-step walkthrough of how NetSentinel's multi-agent architecture processes queries and delivers insights, offering a comprehensive view of its potential applications.

### Key Components and Features:

- **Generative and Predictive AI:** The system leverages `granite-8b-code-instruct-128k` for generative capabilities, a predictive model trained on the `UNSW-NB15` dataset for anomaly detection, and an NLU component to interpret human intent as the entry point for all interactions.

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
- Leverages various OpenShift components to demonstrate Red Hat OpenShift AI functionality:
  - **Kafka** as a middleware service for event streaming
  - **NVIDIA Triton** model server to deploy the predictive model
  - Hosted **Model-as-a-Service** integration to interact with `granite-8b-code-instruct-128k`
  - **OpenShift Tekton Pipeline** to automate the image-building process
  - **RHOAI Kubeflow Pipeline** to showcase the predictive model training process

NetSentinel offers a glimpse into the future of AI-enhanced operational workflows, emphasizing adaptability, scalability, and real-time response for diverse applications.

## Order OpenShift Environment

- Any OpenShift environment should work technically, provided there are no operator conflicts. To avoid issues, it’s recommended to start with a clean environment since the project requires installing multiple operators and configurations.
- For testing purposes, we are using the following environment.
  - Order an OCP demo cluster via this [URL](https://catalog.demo.redhat.com/catalog?item=babylon-catalog-prod/sandboxes-gpte.ocp-wksp.prod&utm_source=webapp&utm_medium=share-link)
  - Select **OpenShift Version 4.17** during setup.
  - Only a single control plane is sufficient.
  - If you are using **Model as a Service** for the LLM model, a CPU-only setup is adequate for deploying this project.

## Running the Application
To test this application, follow the appropriate instructions based on your environment:

- On RHDP: Refer to the instructions [here] .

- On On-Premises OpenShift Cluster: Follow the steps mentioned in [steps-to-run-on-prem.md] (./steps-to-run-on-prem.md) file.