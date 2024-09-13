# NetSentenial

NetSentenial is a proof-of-concept (POC) network intrusion detection system that leverages both predictive models and Large Language Models (LLMs) to detect intrusions and interact with users via Slack. The project demonstrates how machine learning models can be integrated with a chatbot to provide insights into network security events.

## Table of Contents

- [NetSentenial](#netsentenial)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Folder Structure](#folder-structure)
  - [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Virtual Environment Setup](#virtual-environment-setup)
      - [Using `venv`](#using-venv)
      - [Using `conda`](#using-conda)
    - [Install Dependencies](#install-dependencies)
    - [Data Preparation](#data-preparation)
      - [Download the UNSW-NB15 Dataset](#download-the-unsw-nb15-dataset)
      - [Preprocess the Data](#preprocess-the-data)
    - [Model Training](#model-training)
      - [Train the Predictive Model](#train-the-predictive-model)
      - [Fine-Tune the LLM](#fine-tune-the-llm)
    - [Running the Backend API](#running-the-backend-api)
    - [Running the Slack Bot](#running-the-slack-bot)
  - [Deployment on OpenShift](#deployment-on-openshift)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

---

## Project Overview

NetSentenial aims to:

- Build a predictive model to detect network intrusions using the UNSW-NB15 dataset.
- Fine-tune an LLM (GPT-2) to handle security-related queries.
- Integrate both models into a backend API.
- Develop a Slack bot that allows users to interact with the system using natural language.

## Features

- **Predictive Intrusion Detection Model**: Detects potential network intrusions based on network traffic data.
- **Fine-Tuned LLM Chatbot**: Responds to user queries about network security events.
- **Slack Integration**: Allows users to interact with the system through a Slack bot.
- **Modular Architecture**: Separates concerns into different components for scalability and maintainability.
- **Deployment on OpenShift**: Supports deployment on OpenShift for container orchestration and management.

## Folder Structure

```
NetSentenial/
├── app/                    # Backend Flask API
├── data/                   # Data files
│   ├── processed/          # Processed data
│   └── raw/                # Raw data
├── deployment/             # Deployment configurations
├── models/                 # Trained models
│   ├── llm_model/          # Fine-tuned LLM
│   └── predictive_model/   # Predictive model
├── scripts/                # Scripts for data processing and training
├── slack_bot/              # Slack bot application
├── tests/                  # Test suite
├── LICENSE                 # License file
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

---

## Setup Instructions

### Prerequisites

- **Python 3.8 or higher**
- **Git**
- **Virtual Environment tool**: `venv` or `conda` (choose one)
- **Slack Workspace**: Access to a Slack workspace where you can add a custom app
- **(Optional) OpenShift Cluster**: For deployment

### Clone the Repository

```
git clone https://github.com/pandeybk/NetSentenial.git
cd NetSentenial
```

### Virtual Environment Setup

It's recommended to use a virtual environment to manage project dependencies.

#### Using `venv`

1. **Create a Virtual Environment**

   ```
   python3 -m venv venv
   ```

2. **Activate the Virtual Environment**

   - On Linux/Mac:

     ```
     source venv/bin/activate
     ```

   - On Windows:

     ```
     venv\Scripts\activate
     ```

#### Using `conda`

1. **Create a Conda Environment**

   ```
   conda create -n netsentenial_env python=3.8
   ```

2. **Activate the Conda Environment**

   ```
   conda activate netsentenial_env
   ```

### Install Dependencies

```
pip install -r requirements.txt
```

### Data Preparation

#### Download the UNSW-NB15 Dataset

1. **Register on Kaggle** (if you haven't already).

2. **Set Up Kaggle API Credentials**

   - Place your `kaggle.json` file in the appropriate directory:

     ```
     mkdir ~/.kaggle
     cp path_to_kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download the Dataset**

   ```
   python scripts/download_data.py
   ```

   *Note: Ensure that the `download_data.py` script is properly configured to download the UNSW-NB15 dataset.*

#### Preprocess the Data

```
python scripts/preprocess_data.py
```

- This script will process the raw data and save it to `data/processed/`.

### Model Training

#### Train the Predictive Model

```
python scripts/train_predictive_model.py
```

- Trains the intrusion detection model and saves it to `models/predictive_model/`.

#### Fine-Tune the LLM

```
python scripts/prepare_llm_data.py
python scripts/train_llm.py
```

- Prepares the data and fine-tunes the GPT-2 model.
- The fine-tuned model is saved to `models/llm_model/`.

### Running the Backend API

1. **Navigate to the `app/` Directory**

   ```
   cd app
   ```

2. **Ensure Dependencies are Installed**

   - If dependencies differ, install them:

     ```
     pip install -r requirements.txt
     ```

3. **Set Environment Variables**

   - Create a `.env` file or set environment variables as needed.

4. **Run the Flask App**

   ```
   flask run
   ```

   - By default, the app runs on `http://localhost:5000`.

### Running the Slack Bot

1. **Navigate to the `slack_bot/` Directory**

   ```
   cd slack_bot
   ```

2. **Ensure Dependencies are Installed**

   ```
   pip install -r requirements.txt
   ```

3. **Set Up Slack App**

   - **Create a Slack App** in your workspace.
   - **Configure Bot Tokens and Signing Secret**.
   - **Set Environment Variables**:

     - `SLACK_BOT_TOKEN`
     - `SLACK_SIGNING_SECRET`
     - Optionally, store these in a `.env` file.

4. **Run the Slack Bot**

   ```
   python bot.py
   ```

5. **Expose the Slack Bot (for local development)**

   - Use a tool like `ngrok` to expose your local server to the internet.

     ```
     ngrok http 3000
     ```

   - Update the Slack app's **Event Subscriptions** with the `ngrok` URL.

---

## Deployment on OpenShift

*Note: Ensure you have access to an OpenShift cluster and the necessary permissions.*

1. **Build Docker Images**

   - Build images for the backend API and Slack bot.

     ```
     # From the root directory
     docker build -t quay.io/bpandey/netsentenial/backend-api:latest -f app/Dockerfile app/
     docker build -t quay.io/bpandey/netsentenial/slack-bot:latest -f slack_bot/Dockerfile slack_bot/
     ```

2. **Push Images to a Registry**

   - Push the images to a container registry accessible by OpenShift.

     ```
     docker push quay.io/bpandey/netsentenial/backend-api:latest
     docker push quay.io/bpandey/netsentenial/slack-bot:latest
     ```

3. **Deploy on OpenShift**

   - Use the deployment files in `deployment/openshift/`.

     ```
     oc apply -f deployment/openshift/backend-deployment.yaml
     oc apply -f deployment/openshift/backend-service.yaml
     oc apply -f deployment/openshift/slackbot-deployment.yaml
     oc apply -f deployment/openshift/slackbot-service.yaml
     oc apply -f deployment/openshift/route.yaml
     ```

4. **Configure OpenShift Routes**

   - Ensure the services are accessible externally if needed.

5. **Update Slack App Configuration**

   - Update the Slack app's URLs to point to the OpenShift routes.

---

## Usage

- **Interacting with the Slack Bot**

  - Send messages or commands to the Slack bot in your workspace.
  - Example queries:

    - "What are the recent attacks?"
    - "List suspicious IPs."

- **API Endpoints**

  - The backend API exposes endpoints (e.g., `/predict`, `/chat`) that can be accessed programmatically if needed.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```
   git commit -m "Description of your changes"
   ```

4. **Push to Your Fork**

   ```
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **UNSW-NB15 Dataset**: [Link to dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Hugging Face Transformers**: [GitHub Repository](https://github.com/huggingface/transformers)
- **Slack Developer Tools**: [Slack API](https://api.slack.com/)

---

**Additional Tips:**

- **Environment Variables Management**:

  - Use the `python-dotenv` package to manage environment variables.
  - Create a `.env` file in the root directory and add it to `.gitignore` to prevent sensitive information from being committed.

- **Testing**:

  - Run tests located in the `tests/` directory to ensure everything is working as expected.

    ```
    pytest tests/
    ```

- **Code Style**:

  - Follow PEP 8 guidelines for Python code.
  - Use linters like `flake8` or formatters like `black` to maintain code quality.

---
