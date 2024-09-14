# scripts/download_data.py

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_kaggle_dataset(dataset, download_path):
    """
    Downloads and extracts a Kaggle dataset.

    Args:
        dataset (str): Kaggle dataset identifier.
        download_path (str): Path to download and extract the dataset.
    """
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print(f"Downloading dataset '{dataset}'...")
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print(f"Dataset downloaded and extracted to '{download_path}'.")

if __name__ == "__main__":
    DATASET_NAME = 'mrwellsdavid/unsw-nb15'
    DOWNLOAD_PATH = './data/raw'

    download_and_extract_kaggle_dataset(DATASET_NAME, DOWNLOAD_PATH)
