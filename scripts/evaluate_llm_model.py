# scripts/evaluate_llm.py

import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import torch

def load_dataset_for_evaluation(data_file):
    """
    Loads the dataset for evaluation.

    Args:
        data_file (str): Path to the JSONL file with QA pairs.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    dataset = load_dataset('json', data_files=data_file)['train']
    return dataset

def tokenize_function(examples, tokenizer):
    """
    Tokenizes the input examples.

    Args:
        examples (dict): Examples from the dataset.
        tokenizer: The tokenizer.

    Returns:
        dict: Tokenized inputs.
    """
    inputs = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(
        inputs,
        truncation=True,
        max_length=512,
        padding='max_length',
    )

if __name__ == "__main__":
    # Paths
    DATA_FILE = os.path.join('data', 'processed', 'qa_pairs.jsonl')
    MODEL_DIR = os.path.join('models', 'llm_model')

    # Load dataset
    dataset = load_dataset_for_evaluation(DATA_FILE)

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True, remove_columns=['question', 'answer'])

    # Define evaluation metric
    metric = load_metric('perplexity')

    # Training arguments (set evaluation batch size)
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=4,
        no_cuda=not torch.cuda.is_available(),
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation Loss: {eval_results['eval_loss']:.4f}")

    # Calculate Perplexity
    import math
    perplexity = math.exp(eval_results['eval_loss'])
    print(f"Perplexity: {perplexity:.2f}")
