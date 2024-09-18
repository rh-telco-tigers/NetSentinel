# scripts/visualize_faiss_metadata.py

import json
import os

def load_metadata(metadata_file='faiss_index/metadata.json'):
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata_store = json.load(f)
        return metadata_store
    else:
        print(f"Metadata file '{metadata_file}' not found.")
        return None

def main():
    metadata_store = load_metadata()
    if metadata_store:
        print("Metadata entries:")
        for idx, entry in enumerate(metadata_store):
            print(f"\nEntry {idx+1}:")
            for key, value in entry.items():
                print(f"  {key}: {value}")
    else:
        print("No metadata to display.")

if __name__ == "__main__":
    main()
