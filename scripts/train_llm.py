# scripts/train_llm.py

import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def load_dataset_for_fine_tuning(data_file):
    """
    Loads the dataset for fine-tuning.

    Args:
        data_file (str): Path to the JSONL file with QA pairs.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    dataset = load_dataset('json', data_files=data_file)['train']
    return dataset

def preprocess_examples(examples, tokenizer):
    """
    Preprocesses the examples for training.

    Args:
        examples (dict): Examples from the dataset.
        tokenizer: The tokenizer.

    Returns:
        dict: Tokenized inputs.
    """
    inputs = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
    model_inputs = tokenizer(inputs, truncation=True, max_length=512)
    return model_inputs

if __name__ == "__main__":
    # Paths
    DATA_FILE = os.path.join('data', 'processed', 'qa_pairs.jsonl')
    OUTPUT_DIR = os.path.join('models', 'llm_model')

    # Load dataset
    dataset = load_dataset_for_fine_tuning(DATA_FILE)

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Preprocess dataset
    dataset = dataset.map(lambda examples: preprocess_examples(examples, tokenizer), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        no_cuda=not torch.cuda.is_available(),
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data]),
            'labels': torch.stack([f['input_ids'] for f in data])
        }
    )

    # Fine-tune the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_DIR}")
