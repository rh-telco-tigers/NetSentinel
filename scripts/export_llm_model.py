# scripts/export_llm.py

import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

if __name__ == "__main__":
    # Paths
    MODEL_DIR = os.path.join('models', 'llm_model')
    OUTPUT_PATH = os.path.join('models', 'llm_model', 'model.onnx')

    # Load the model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

    # Dummy input for the model
    dummy_input = tokenizer("This is a dummy input", return_tensors="pt")

    # Export to ONNX
    torch.onnx.export(
        model,
        args=(dummy_input['input_ids'],),
        f=OUTPUT_PATH,
        input_names=['input_ids'],
        output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 'output': {0: 'batch_size', 1: 'sequence'}},
        opset_version=11
    )

    print(f"Model exported to ONNX format at {OUTPUT_PATH}")

    