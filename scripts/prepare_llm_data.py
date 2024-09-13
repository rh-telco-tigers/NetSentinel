# scripts/prepare_llm_data.py

import os
import pandas as pd
import json

def load_dataset(file_path):
    """
    Loads the UNSW-NB15 dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def create_qa_pairs(df):
    """
    Creates question-answer pairs for fine-tuning the LLM.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        List[dict]: List of question-answer pairs.
    """
    qa_pairs = []

    # Example: Get top 5 recent attacks
    recent_attacks = df['attack_cat'].value_counts().head(5).index.tolist()
    attacks_str = ', '.join(recent_attacks)

    qa_pairs.append({
        'question': 'What are the recent attacks?',
        'answer': f'The recent attacks include {attacks_str}.'
    })

    # Example: Get list of suspicious IPs (assuming 'srcip' column exists)
    # For demonstration, select top 5 IPs involved in attacks
    if 'srcip' in df.columns:
        suspicious_ips = df[df['label'] == 1]['srcip'].value_counts().head(5).index.tolist()
        ips_str = ', '.join(suspicious_ips)
        qa_pairs.append({
            'question': 'List suspicious IPs.',
            'answer': f'The suspicious IPs are {ips_str}.'
        })

    # Add more QA pairs as needed

    return qa_pairs

def save_qa_pairs(qa_pairs, output_file):
    """
    Saves the QA pairs to a JSONL file.

    Args:
        qa_pairs (List[dict]): List of QA pairs.
        output_file (str): Output file path.
    """
    with open(output_file, 'w') as f:
        for pair in qa_pairs:
            json.dump(pair, f)
            f.write('\n')
    print(f"QA pairs saved to {output_file}")

if __name__ == "__main__":
    # Paths
    DATA_DIR = os.path.join('data', 'raw')
    DATA_FILE = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')
    OUTPUT_FILE = os.path.join('data', 'processed', 'qa_pairs.jsonl')

    # Load dataset
    df = load_dataset(DATA_FILE)

    # Create QA pairs
    qa_pairs = create_qa_pairs(df)

    # Save QA pairs
    save_qa_pairs(qa_pairs, OUTPUT_FILE)
