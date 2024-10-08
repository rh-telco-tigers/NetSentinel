# Makefile

# Define directories and log files
DATA_DIR = data
MODEL_DIR = models
LOG_DIR = logs
ENV_DIR = venv

DOCKER_IMAGE = quay.io/bpandey/netsentenial:0.0.1
# PLATFORMS = linux/amd64,linux/arm64
PLATFORMS = linux/amd64


# Define commands
PYTHON = $(ENV_DIR)/bin/python
PIP = $(ENV_DIR)/bin/pip

# Define files
CONFIG_FILE = config.yaml
REQUIREMENTS_FILE = requirements.txt

# ngrok Variables
NGROK_URL = https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
NGROK_TGZ = ngrok.tgz
NGROK_BIN = ngrok

# Define directories and model paths
MISTRAL_MODEL_DIR = models/Mistral-7B-v0.1-demo
MISTRAL_REPO = git@hf.co:mistralai/Mistral-7B-v0.1

FLAN_MODEL_DIR = models/flan-t5-large
FLAN_REPO = git@hf.co:google/flan-t5-large

rasa_folder := rasa
rasa_output_model_dir := models/rasa
rasa_config_file := $(rasa_folder)/config.yml
rasa_nlu_file := $(rasa_folder)/nlu.yml
rasa_domain_file := $(rasa_folder)/domain.yml
rasa_data_dir := $(rasa_folder)/data

# Phony targets to avoid conflicts with files named like these
.PHONY: setup_dirs ensure_setuptools setup_env download_data preprocess_data train_predictive_model evaluate_predictive_model export_predictive_model train_llm export_llm run_app start_services run_tests clean start-ngrok

# Create log and model directories if they don't exist
setup_dirs:
	@mkdir -p $(LOG_DIR)
	@mkdir -p $(MODEL_DIR)
	@echo "✅ Directories for logs and models are set up."

# Ensure setuptools is installed
ensure_setuptools:
	@echo "🔧 Ensuring setuptools is installed..."
	python3 -m ensurepip --upgrade
	python3 -m pip install --upgrade setuptools
	@echo "✅ setuptools is installed."

# Docker Build Section
build_docker:
	@echo "🐳 Building Docker image $(DOCKER_IMAGE)..."
	@docker buildx create --use || true
	@docker buildx inspect --bootstrap
	@docker buildx build --platform $(PLATFORMS) -t $(DOCKER_IMAGE) --push .
	@echo "✅ Docker image $(DOCKER_IMAGE) built and pushed."

# Set up the Python virtual environment
setup_env: ensure_setuptools setup_dirs
	@echo "🔧 Setting up Python virtual environment..."
	python3 -m venv $(ENV_DIR)
	@echo "🔧 Installing dependencies from $(REQUIREMENTS_FILE)..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "✅ Python environment setup complete."

# Download the data
download_data:
	@echo "📥 Downloading data..."
	$(PYTHON) scripts/download_data.py
	@echo "✅ Data downloaded."

# Preprocess the downloaded data
preprocess_data: download_data
	@echo "🛠️ Preprocessing data..."
	$(PYTHON) scripts/preprocess_data.py
	@echo "✅ Data preprocessed."

.PHONY: setup-monitoring-access
setup-monitoring-access:
	@echo "🔐 Assigning cluster-monitoring-view role and generating token for ServiceAccount..."
	oc adm policy add-scc-to-user privileged -z dev-netsentenial-sa
	oc adm policy add-cluster-role-to-user cluster-monitoring-view -z dev-netsentenial-sa
	@TOKEN=$$(oc create token dev-netsentenial-sa); \
	echo "✅ Token generated."; \
	echo ""; \
	echo "Run the following command to export the token as an environment variable:"; \
	echo ""; \
	echo 'export PROMETHEUS_BEARER_TOKEN="'$$TOKEN'"'; \
	echo ""; \
	echo "To test Prometheus connection, use the following steps:"; \
	echo "1. Port forward Prometheus service using:"; \
	echo "   oc port-forward svc/thanos-querier 9091 -n openshift-monitoring"; \
	echo "2. Run the following curl command to validate the connection:"; \
	echo '   curl -k -H "Authorization: Bearer $$PROMETHEUS_BEARER_TOKEN" \'; \
	echo '      "https://localhost:9091/api/v1/query?query=up" | jq .'; \
	echo ""; \
	echo "3. Make sure you have 'jq' installed to parse the response."


# Task to download Mistral 7B model using git clone
.PHONY: download_mistral
download_mistral:
	@echo "🔍 Checking if Mistral-7B model exists..."
	@if [ ! -d $(MISTRAL_MODEL_DIR) ]; then \
		echo "📥 Cloning Mistral-7B model from Hugging Face..."; \
		brew install git-lfs; \
		git lfs install; \
		git clone $(MISTRAL_REPO) $(MISTRAL_MODEL_DIR); \
		echo "✅ Mistral-7B model downloaded successfully."; \
	else \
		echo "✅ Mistral-7B model already exists."; \
	fi


# Task to download Mistral 7B model using git clone
.PHONY: download_flan
download_flan:
	@echo "🔍 Checking if flan-t5-large model exists..."
	@if [ ! -d $(FLAN_MODEL_DIR) ]; then \
		echo "📥 Cloning flan-t5-large model from Hugging Face..."; \
		brew install git-lfs; \
		git lfs install; \
		git clone $(FLAN_REPO) $(FLAN_MODEL_DIR); \
		echo "✅ flan-t5-large model downloaded successfully."; \
	else \
		echo "✅ flan-t5-large model already exists."; \
	fi


# Task to download all-MiniLM-L6-v2 model using git clone
.PHONY: download_minilm
download_minilm:
	@echo "🔍 Checking if all-MiniLM-L6-v2 model exists..."
	@if [ ! -d $(MINILM_MODEL_DIR) ]; then \
		echo "📥 Cloning all-MiniLM-L6-v2 model from Hugging Face..."; \
		brew install git-lfs || echo "git-lfs already installed"; \
		git lfs install; \
		git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 $(MINILM_MODEL_DIR); \
		echo "✅ all-MiniLM-L6-v2 model downloaded successfully."; \
	else \
		echo "✅ all-MiniLM-L6-v2 model already exists."; \
	fi

# Train the predictive model
train_predictive_model: preprocess_data
	@echo "🧠 Training the predictive model..."
	$(PYTHON) scripts/train_predictive_model.py --config_file $(CONFIG_FILE) > $(LOG_DIR)/train_predictive.log 2>&1
	@echo "✅ Predictive model training completed."

# Evaluate the predictive model
evaluate_predictive_model: train_predictive_model
	@echo "📊 Evaluating the predictive model..."
	$(PYTHON) scripts/evaluate_predictive_model.py --config_file $(CONFIG_FILE) > $(LOG_DIR)/evaluate_predictive.log 2>&1
	@echo "✅ Predictive model evaluation completed."

# Export the predictive model
export_predictive_model: evaluate_predictive_model
	@echo "📤 Exporting the predictive model..."
	$(PYTHON) scripts/export_predictive_model.py --config_file $(CONFIG_FILE)
	@echo "✅ Predictive model exported."

# Generate QA Pairs for LLM
prepare_llm_data: download_data
	@echo "🛠️ Creating QA pairs for LLM..."
	$(PYTHON) scripts/prepare_llm_data.py
	@echo "✅ QA Pairs generated."

# Train the LLM model
train_llm: prepare_llm_data
	@echo "🧠 Training the LLM model..."
	$(PYTHON) scripts/train_llm.py --config_file $(CONFIG_FILE) > $(LOG_DIR)/train_llm.log 2>&1
	@echo "✅ LLM model training completed."

# Train the LLM model
train_llm_mistral:
	@echo "🧠 Training the Mistral LLM model..."
	$(PYTHON) scripts/train_llm_mistral.py --config_file $(CONFIG_FILE) --model_config_section llm_model_config_mistral > $(LOG_DIR)/train_llm_mistral.log 2>&1
	@echo "✅ LLM model training completed."

# Export the LLM model
export_llm: train_llm
	@echo "📤 Exporting the LLM model..."
	$(PYTHON) scripts/export_llm_model.py --config_file $(CONFIG_FILE)
	@echo "✅ LLM model exported."

train_rasa:
	@echo "Training Rasa model..."
	@rasa train --config $(rasa_config_file) --domain $(rasa_domain_file) --data $(rasa_folder) --out $(rasa_output_model_dir)
	@echo "Model training complete. The trained model is saved in $(rasa_output_model_dir)"


# Start services
start_services:
	@echo "🚀 Starting services..."
	@echo "Starting create_mock_data.py..."
	nohup bash -c 'source $(ENV_DIR)/bin/activate && python scripts/create_mock_data.py' > $(LOG_DIR)/create_mock_data.log 2>&1 &
	@echo "Starting process_mock_data.py..."
	nohup bash -c 'source $(ENV_DIR)/bin/activate && python scripts/process_mock_data.py' > $(LOG_DIR)/process_mock_data.log 2>&1 &
	@echo "Starting prediction_service.py..."
	nohup bash -c 'source $(ENV_DIR)/bin/activate && python scripts/prediction_service.py' > $(LOG_DIR)/prediction_service.log 2>&1 &
	@echo "✅ Services are running."

# Run the application
run_app: export_llm start_services
	@echo "🚀 Starting the application..."
	$(PYTHON) -m app.run
	@echo "✅ Application is running."

# Run all steps in order
start: run_app
	@echo "🎉 Project setup completed and application started."

# Run tests
run_tests: setup_env
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest --cov=app app/tests/
	@echo "✅ Tests completed."

# Start ngrok
start-ngrok:
	@echo "🔍 Checking if ngrok exists..."
	@if [ ! -f $(NGROK_BIN) ]; then \
		echo "📥 Downloading ngrok..."; \
		wget -O $(NGROK_TGZ) $(NGROK_URL); \
		tar -xzf $(NGROK_TGZ); \
		chmod +x $(NGROK_BIN); \
		echo "✅ ngrok downloaded and ready."; \
	else \
		echo "✅ ngrok already exists."; \
	fi
	@echo "🚀 Starting ngrok..."
	$(NGROK_BIN) http --domain=newly-advanced-dane.ngrok-free.app 5001

# Clean the log, model directories, and virtual environment
clean:
	@echo "🧹 Cleaning logs, models, and virtual environment..."
	rm -rf $(LOG_DIR)/*
	rm -rf $(MODEL_DIR)/*
	rm -rf $(ENV_DIR)
	@echo "✅ Cleanup completed."

# Stop services
.PHONY: stop
stop:
	@echo "🛑 Stopping services..."
	@pkill -f create_mock_data.py || true
	@pkill -f process_mock_data.py || true
	@pkill -f prediction_service.py || true
	@echo "✅ Services have been stopped."