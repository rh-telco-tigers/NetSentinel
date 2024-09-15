# scripts/train_llm.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import glob
import sys
import argparse
import logging
import yaml  # Import yaml module
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
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

def parse_args():
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
        help='Path to a YAML configuration file.'
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
    parser.add_argument(
        '--evaluation_strategy',
        type=str,
        default='steps',  # or 'epoch'
        help='Evaluation strategy to use.'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help='Number of steps between evaluations (if evaluation_strategy="steps").'
    )
    parser.add_argument(
        '--save_strategy',
        type=str,
        default='steps',  # or 'epoch'
        help='Checkpoint save strategy to use.'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to a checkpoint from which training will be resumed.'
    )
    parser.add_argument(
        '--subset_size',
        type=int,
        default=None,
        help='Number of examples to use from the dataset. If None, use the full dataset.'
    )
    args = parser.parse_args()
    return args

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
    args = parse_args()

    # Load configuration from file if provided
    if args.config_file:
        config = load_config(args.config_file)
        # Update config with args to prioritize command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)

    # Ensure critical parameters are set correctly
    if not hasattr(args, 'load_best_model_at_end') or args.load_best_model_at_end is not True:
        args.load_best_model_at_end = True

    if not hasattr(args, 'evaluation_strategy') or args.evaluation_strategy not in ['steps', 'epoch']:
        args.evaluation_strategy = 'steps'

    if not hasattr(args, 'save_strategy') or args.save_strategy not in ['steps', 'epoch']:
        args.save_strategy = 'steps'

    # Setup logging
    setup_logging(args.log_level.upper(), args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("Starting training script.")
    logger.debug(f"Arguments: {args}")

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset_for_fine_tuning(args.data_file, subset_size=args.subset_size)

    # Split the dataset into training and evaluation sets
    train_size = 0.9
    test_size = 1 - train_size
    # Convert Dataset to pandas DataFrame for splitting
    dataset_df = dataset.to_pandas()
    train_df, eval_df = train_test_split(
        dataset_df, test_size=test_size, random_state=42
    )
    # Convert back to Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

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

    # Tokenize the datasets
    try:
        tokenized_train_dataset = train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, args.max_length),
            batched=True,
            remove_columns=['question', 'answer']
        )

        tokenized_eval_dataset = eval_dataset.map(
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

    # Replace 'no_cuda' with 'use_cpu' (if using Transformers 4.44.2 or later)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available() and args.use_gpu,
        use_cpu=not torch.cuda.is_available() or not args.use_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        report_to="none",
        load_best_model_at_end=args.load_best_model_at_end,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        save_total_limit=2,
        # Include save_steps and eval_steps only if strategies are 'steps'
        **({'save_steps': args.save_steps} if args.save_strategy == 'steps' else {}),
        **({'eval_steps': args.eval_steps} if args.evaluation_strategy == 'steps' else {}),
    )

    # Log training arguments
    logger.debug(f"Training arguments: {training_args}")

    # Callbacks
    callbacks = []
    if args.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        callbacks.append(early_stopping_callback)
        logger.info("Early stopping enabled.")

    # Optional: Define compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss)
        return {'eval_loss': loss.item(), 'perplexity': perplexity.item()}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    try:
        if args.resume_from_checkpoint is None:
            last_checkpoint = get_last_checkpoint(args.output_dir)
            if last_checkpoint:
                logger.info(f"No checkpoint specified. Resuming from last checkpoint: {last_checkpoint}")
                args.resume_from_checkpoint = last_checkpoint
            else:
                logger.info("No checkpoints found. Starting training from scratch.")

        logger.info("Starting model training.")
        if args.resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
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
