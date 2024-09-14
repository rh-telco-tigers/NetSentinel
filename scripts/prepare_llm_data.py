# scripts/prepare_llm_data.py

import os
import pandas as pd
import json
import random

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
    Creates a large number of question-answer pairs for fine-tuning the LLM.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        List[dict]: List of question-answer pairs.
    """
    qa_pairs = []

    # Ensure consistent column names
    df.columns = df.columns.str.lower()

    # Map labels to 'Normal' and 'Attack' for clarity
    df['label_str'] = df['label'].apply(lambda x: 'Attack' if x == 1 else 'Normal')

    # Drop rows with missing 'attack_cat' for attacks
    df.loc[df['label'] == 1, 'attack_cat'] = df.loc[df['label'] == 1, 'attack_cat'].fillna('Unknown')

    # Iterate over the entire dataset
    for idx, row in df.iterrows():
        # Generate QA pairs based on whether the event is an attack or normal traffic
        if row['label'] == 1:
            # Attack event
            attack_cat = row['attack_cat']
            proto = row['proto']
            service = row['service'] if not pd.isnull(row['service']) else 'Unknown'
            srcip = row['srcip'] if 'srcip' in df.columns else 'Unknown'
            dstip = row['dstip'] if 'dstip' in df.columns else 'Unknown'

            # Question templates for attack events
            questions = [
                f"What type of attack occurred in event ID {idx}?",
                f"Which protocol was used in attack event {idx}?",
                f"Which service was targeted in attack event {idx}?",
                f"From which IP did attack event {idx} originate?",
                f"To which IP was attack event {idx} directed?",
                f"Provide details about attack event ID {idx}.",
            ]

            # Corresponding answers
            answers = [
                f"The attack in event ID {idx} was of type {attack_cat}.",
                f"The protocol used in attack event {idx} was {proto}.",
                f"The service targeted in attack event {idx} was {service}.",
                f"The attack originated from IP {srcip}.",
                f"The attack was directed to IP {dstip}.",
                f"Event ID {idx} was an {attack_cat} attack using {proto} protocol targeting {service} service from IP {srcip} to IP {dstip}.",
            ]

        else:
            # Normal traffic event
            proto = row['proto']
            service = row['service'] if not pd.isnull(row['service']) else 'Unknown'
            srcip = row['srcip'] if 'srcip' in df.columns else 'Unknown'
            dstip = row['dstip'] if 'dstip' in df.columns else 'Unknown'

            # Question templates for normal events
            questions = [
                f"Was event ID {idx} an attack?",
                f"Which protocol was used in normal event {idx}?",
                f"Which service was involved in normal event {idx}?",
                f"From which IP did normal event {idx} originate?",
                f"To which IP was normal event {idx} directed?",
                f"Provide details about normal event ID {idx}.",
            ]

            # Corresponding answers
            answers = [
                f"No, event ID {idx} was not an attack.",
                f"The protocol used in normal event {idx} was {proto}.",
                f"The service involved in normal event {idx} was {service}.",
                f"The event originated from IP {srcip}.",
                f"The event was directed to IP {dstip}.",
                f"Event ID {idx} was normal traffic using {proto} protocol with {service} service from IP {srcip} to IP {dstip}.",
            ]

        # Select a subset of questions and answers to prevent data overload
        for question, answer in zip(questions, answers):
            qa_pairs.append({
                'question': question,
                'answer': answer
            })

    # Additional aggregated QA pairs

    # Attack categories and counts
    attack_counts = df[df['label'] == 1]['attack_cat'].value_counts()
    for attack_cat, count in attack_counts.items():
        qa_pairs.append({
            'question': f'How many {attack_cat} attacks have occurred?',
            'answer': f'There have been {count} instances of {attack_cat} attacks.'
        })

    # Protocol usage in attacks
    attack_protocols = df[df['label'] == 1]['proto'].value_counts()
    for proto, count in attack_protocols.items():
        qa_pairs.append({
            'question': f'How often is the {proto} protocol used in attacks?',
            'answer': f'The {proto} protocol was used in {count} attack events.'
        })

    # Services targeted in attacks
    if 'service' in df.columns:
        attack_services = df[df['label'] == 1]['service'].dropna().value_counts()
        for service, count in attack_services.items():
            qa_pairs.append({
                'question': f'How many attacks targeted the {service} service?',
                'answer': f'There have been {count} attacks targeting the {service} service.'
            })

    # Source IPs involved in attacks
    if 'srcip' in df.columns:
        attack_ips = df[df['label'] == 1]['srcip'].value_counts()
        for ip, count in attack_ips.items():
            qa_pairs.append({
                'question': f'How many attacks originated from IP {ip}?',
                'answer': f'There have been {count} attacks originating from IP {ip}.'
            })

    # Overall network statistics
    total_records = len(df)
    total_attacks = df['label'].sum()
    attack_rate = (total_attacks / total_records) * 100
    qa_pairs.append({
        'question': 'What is the overall attack rate on the network?',
        'answer': f'The network has an attack rate of {attack_rate:.2f}%, with {int(total_attacks)} attacks out of {total_records} total events.'
    })

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
    print(f"QA pairs saved to {output_file} with {len(qa_pairs)} entries.")

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
