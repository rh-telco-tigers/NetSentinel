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

# Phony targets to avoid conflicts with files named like these
.PHONY: setup_dirs ensure_setuptools setup_env download_data preprocess_data train_predictive_model evaluate_predictive_model export_predictive_model train_llm export_llm run_app start run_tests clean

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

# Set up the Python virtual environment
setup_env: ensure_setuptools setup_dirs
	@echo "🔧 Setting up Python virtual environment..."
	python3 -m venv $(ENV_DIR)
	@echo "🔧 Installing dependencies from $(REQUIREMENTS_FILE)..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "✅ Python environment setup complete."

# Download the data
download_data: setup_env
	@echo "📥 Downloading data..."
	$(PYTHON) scripts/download_data.py
	@echo "✅ Data downloaded."

# Preprocess the downloaded data
preprocess_data: download_data
	@echo "🛠️ Preprocessing data..."
	$(PYTHON) scripts/preprocess_data.py
	@echo "✅ Data preprocessed."

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
	@echo "🛠️ Create QA pairs for LLM..."
	$(PYTHON) scripts/prepare_llm_data.py
	@echo "✅ QA Pairs generated."

# Train the LLM model
train_llm: prepare_llm_data
	@echo "🧠 Training the LLM model..."
	$(PYTHON) scripts/train_llm.py --config_file $(CONFIG_FILE) > $(LOG_DIR)/train_llm.log 2>&1
	@echo "✅ LLM model training completed."

# Export the LLM model
export_llm: train_llm
	@echo "📤 Exporting the LLM model..."
	$(PYTHON) scripts/export_llm_model.py --config_file $(CONFIG_FILE)
	@echo "✅ LLM model exported."

# Run the application
run_app: export_llm
	@echo "🚀 Starting the application..."
	$(PYTHON) -m app.run
	@echo "✅ Application is running."

# Run all steps in order
start: setup_env download_data preprocess_data train_predictive_model evaluate_predictive_model export_predictive_model train_llm export_llm run_app
	@echo "🎉 Project setup completed and application started."

# Run tests
run_tests: setup_env
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest --cov=app app/tests/
	@echo "✅ Tests completed."

# Clean the log, model directories, and virtual environment
clean:
	@echo "🧹 Cleaning logs, models, and virtual environment..."
	rm -rf $(LOG_DIR)/*
	rm -rf $(MODEL_DIR)/*
	rm -rf $(ENV_DIR)
	@echo "✅ Cleanup completed."
