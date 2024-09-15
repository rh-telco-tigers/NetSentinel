# Makefile

.PHONY: train download_data preprocess train_predictive evaluate_predictive export_predictive

train:
	# Start the training process with nohup and redirect output to train_llm.log
	nohup python scripts/train_llm.py --config_file config.yaml --log_file log/train_llm.log > log/train_llm.out 2>&1 & disown

download_data:
	python scripts/download_data.py

preprocess:
	python scripts/preprocess_data.py

train_predictive:
	python scripts/train_predictive_model.py

evaluate_predictive:
	python scripts/evaluate_predictive_model.py

export_predictive:
	python scripts/export_predictive_model.py
