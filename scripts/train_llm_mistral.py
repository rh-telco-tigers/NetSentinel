# train_llm_mistral.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import glob
import sys
import argparse
import logging
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
import torch
from sklearn.model_selection import train_test_split

def setup_logging(log_level, log_file=None):
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config):
    required_fields = [
        'model_path', 'data_file', 'tokenizer_name', 'model_name',
        'num_train_epochs', 'learning_rate', 'per_device_train_batch_size',
        'logging_steps', 'save_steps', 'save_total_limit',
        'gradient_accumulation_steps', 'max_length',
        'early_stopping', 'early_stopping_patience',
        'eval_strategy', 'eval_steps', 'save_strategy',
        'load_best_model_at_end', 'metric_for_best_model',
        'greater_is_better', 'subset_size', 'resume_from_checkpoint', 'use_cpu'
    ]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing field '{field}' in model configuration.")

    # Additional type checks
    if not isinstance(config['num_train_epochs'], (float, int)):
        raise TypeError("num_train_epochs must be a float or int.")
    if not isinstance(config['learning_rate'], float):
        raise TypeError("learning_rate must be a float.")
    if not isinstance(config['early_stopping'], bool):
        raise TypeError("early_stopping must be a boolean.")
    if not isinstance(config['use_cpu'], bool):
        raise TypeError("use_cpu must be a boolean.")

def log_config_types(config, logger):
    for key, value in config.items():
        logger.debug(f"Config Param - {key}: {value} (type: {type(value).__name__})")

def load_dataset_for_fine_tuning(data_file, subset_size=None):
    try:
        dataset = load_dataset('json', data_files=data_file)['train']
        logging.info(f"Dataset loaded successfully from {data_file}")
        if subset_size is not None:
            logging.info(f"Selecting a subset of size {subset_size}")
            dataset = dataset.shuffle(seed=42).select(range(subset_size))
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

def tokenize_function(examples, tokenizer, max_length):
    inputs = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(
        inputs,
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )

def get_last_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1]

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B on custom QA pairs.")
    parser.add_argument(
        '--config_file',
        type=str,
        default='config.yaml',
        help='Path to a YAML configuration file.'
    )
    parser.add_argument(
        '--model_config_section',
        type=str,
        default='llm_model_config_mistral',
        help='Section in the config file for Mistral model configuration.'
    )
    args = parser.parse_args()

    # Load configuration from file
    config = load_config(args.config_file)
    model_config = config.get(args.model_config_section)
    if model_config is None:
        logging.error(f"Model configuration section '{args.model_config_section}' not found in config file.")
        sys.exit(1)

    # Setup logging
    setup_logging(config.get('logging_config', {}).get('level', 'INFO'))
    logger = logging.getLogger(__name__)

    logger.info("Starting Mistral 7B training script.")

    # Validate configuration
    try:
        validate_config(model_config)
    except (ValueError, TypeError) as e:
        logger.error(f"Configuration validation error: {e}")
        sys.exit(1)

    # Log configuration parameter types
    log_config_types(model_config, logger)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() and not model_config['use_cpu'] else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset_for_fine_tuning(model_config['data_file'], subset_size=model_config['subset_size'])

    # Split the dataset into training and evaluation sets
    train_size = 0.9
    test_size = 1 - train_size
    dataset_df = dataset.to_pandas()
    train_df, eval_df = train_test_split(
        dataset_df, test_size=test_size, random_state=42
    )
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer_name'])
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token if necessary
        model = AutoModelForCausalLM.from_pretrained(model_config['model_name'])
        model.to(device)
        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

    # Tokenize the datasets
    try:
        tokenized_train_dataset = train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, model_config['max_length']),
            batched=True,
            remove_columns=['question', 'answer']
        )

        tokenized_eval_dataset = eval_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, model_config['max_length']),
            batched=True,
            remove_columns=['question', 'answer']
        )

        logger.info("Dataset tokenization completed.")
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        sys.exit(1)

    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_config['model_path'],
        num_train_epochs=model_config['num_train_epochs'],
        per_device_train_batch_size=model_config['per_device_train_batch_size'],
        learning_rate=model_config['learning_rate'],
        logging_steps=model_config['logging_steps'],
        gradient_accumulation_steps=model_config['gradient_accumulation_steps'],
        logging_dir=os.path.join(model_config['model_path'], 'logs'),
        report_to="none",
        load_best_model_at_end=model_config['load_best_model_at_end'],
        eval_strategy=model_config['eval_strategy'],
        save_strategy=model_config['save_strategy'],
        metric_for_best_model=model_config['metric_for_best_model'],
        greater_is_better=model_config['greater_is_better'],
        save_total_limit=model_config['save_total_limit'],
        save_steps=model_config.get('save_steps', None),
        eval_steps=model_config.get('eval_steps', None),
        fp16=False,  # FP16 not supported on CPU
    )

    # Log training arguments
    logger.debug(f"Training arguments: {training_args}")

    # Callbacks
    callbacks = []
    if model_config['early_stopping']:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=model_config['early_stopping_patience']
        )
        callbacks.append(early_stopping_callback)
        logger.info("Early stopping enabled.")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Fine-tune the model
    try:
        if model_config['resume_from_checkpoint']:
            logger.info(f"Resuming training from checkpoint: {model_config['resume_from_checkpoint']}")
            trainer.train(resume_from_checkpoint=model_config['resume_from_checkpoint'])
        else:
            last_checkpoint = get_last_checkpoint(model_config['model_path'])
            if last_checkpoint:
                logger.info(f"No checkpoint specified. Resuming from last checkpoint: {last_checkpoint}")
                trainer.train(resume_from_checkpoint=last_checkpoint)
            else:
                logger.info("No checkpoints found. Starting training from scratch.")
                trainer.train()
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)

    # Save the model and tokenizer
    try:
        trainer.save_model(model_config['model_path'])
        tokenizer.save_pretrained(model_config['model_path'])
        logger.info(f"Fine-tuned model and tokenizer saved to {model_config['model_path']}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
