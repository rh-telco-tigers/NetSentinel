# scripts/train_llm.py

import os
import sys
import argparse
import logging
import json
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset
import torch

def setup_logging(log_level, log_file=None):
    """
    Sets up logging with the specified level and optional file handler.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (str): Optional path to a log file.
    """
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on custom QA pairs.")

    parser.add_argument(
        '--data_file',
        type=str,
        default=os.path.join('data', 'processed', 'qa_pairs.jsonl'),
        help='Path to the JSONL file containing QA pairs.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join('models', 'llm_model'),
        help='Directory to save the fine-tuned model.'
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=3,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=4,
        help='Batch size per device during training.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate.'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=100,
        help='Log every X updates steps.'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='Save checkpoint every X updates steps.'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Optional path to a log file.'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default=None,
        help='Path to a JSON configuration file.'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Flag to enable GPU usage if available.'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps.'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length.'
    )
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        help='Enable early stopping based on evaluation loss.'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=1,
        help='Number of evaluation steps with no improvement after which training will be stopped.'
    )
    args = parser.parse_args()
    return args

def load_config(config_file):
    """
    Loads configuration from a JSON file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_dataset_for_fine_tuning(data_file):
    """
    Loads the dataset for fine-tuning.

    Args:
        data_file (str): Path to the JSONL file with QA pairs.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    try:
        dataset = load_dataset('json', data_files=data_file)['train']
        logging.info(f"Dataset loaded successfully from {data_file}")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenizes the input examples.

    Args:
        examples (dict): Examples from the dataset.
        tokenizer: The tokenizer.
        max_length (int): Maximum sequence length.

    Returns:
        dict: Tokenized inputs.
    """
    inputs = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(
        inputs,
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )

def main():
    # Parse arguments
    args = parse_args()

    # Load configuration from file if provided
    if args.config_file:
        config = load_config(args.config_file)
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)

    # Setup logging
    setup_logging(args.log_level.upper(), args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("Starting training script.")
    logger.debug(f"Arguments: {args}")

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset_for_fine_tuning(args.data_file)

    # Load tokenizer and model
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.to(device)
        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

    # Tokenize the dataset
    try:
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, args.max_length),
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available() and args.use_gpu,
        no_cuda=not torch.cuda.is_available() or not args.use_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        report_to="none",  # Disable default logging to avoid duplicate logs
    )

    # Callbacks
    callbacks = []
    if args.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        callbacks.append(early_stopping_callback)
        logger.info("Early stopping enabled.")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Fine-tune the model
    try:
        logger.info("Starting model training.")
        trainer.train()
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)

    # Save the model and tokenizer
    try:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Fine-tuned model and tokenizer saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
