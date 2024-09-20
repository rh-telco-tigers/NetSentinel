# Makefile

# Define directories and log files
DATA_DIR = data
MODEL_DIR = models
LOG_DIR = log
ENV_DIR = venv

# Define commands
PYTHON = $(ENV_DIR)/bin/python
PIP = $(ENV_DIR)/bin/pip

# Define files
CONFIG_FILE = config.yaml
REQUIREMENTS_FILE = requirements.txt

# Create log and model directories if they don't exist
setup_dirs:
	@mkdir -p $(LOG_DIR)
	@mkdir -p $(MODEL_DIR)
	@echo "Directories for logs and models are set up."

# Set up the Python environment
setup_env: setup_dirs
	@echo "Setting up Python virtual environment..."
	python3 -m venv $(ENV_DIR)
	@echo "Installing dependencies from $(REQUIREMENTS_FILE)..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "Python environment setup complete."

# Download the data
download_data: setup_env
	@echo "Downloading data..."
	$(PYTHON) scripts/download_data.py

# Preprocess the downloaded data
preprocess_data: download_data
	@echo "Preprocessing data..."
	$(PYTHON) scripts/preprocess_data.py

# Train the predictive model
train_predictive_model: preprocess_data
	@echo "Training the predictive model..."
	$(PYTHON) scripts/train_predictive_model.py > $(LOG_DIR)/train_predictive.log 2>&1

# Evaluate the predictive model
evaluate_predictive_model: train_predictive_model
	@echo "Evaluating the predictive model..."
	$(PYTHON) scripts/evaluate_predictive_model.py > $(LOG_DIR)/evaluate_predictive.log 2>&1

# Export the predictive model
export_predictive_model: evaluate_predictive_model
	@echo "Exporting the predictive model..."
	$(PYTHON) scripts/export_predictive_model.py

# Set up LLM (RAG flow, no need for LLM training)
setup_llm: export_predictive_model
	@echo "Setting up the LLM model (RAG)..."
	$(PYTHON) scripts/export_llm_model.py

# Run the application
run_app: setup_llm
	@echo "Starting the application..."
	$(PYTHON) -m app.run

# Run all steps in order
start: setup_env download_data preprocess_data train_predictive_model evaluate_predictive_model export_predictive_model setup_llm run_app
	@echo "Project setup completed and application started."

# Run tests
run_tests:
	@echo "Running tests..."
	$(PYTHON) -m pytest --cov=app app/tests/

# Clean the log, model directories, and virtual environment
clean:
	@echo "Cleaning logs, models, and virtual environment..."
	rm -rf $(LOG_DIR)/*
	rm -rf $(MODEL_DIR)/*
	rm -rf $(ENV_DIR)
