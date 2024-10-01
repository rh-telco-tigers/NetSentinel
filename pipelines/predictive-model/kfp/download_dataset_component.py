import kfp
from kfp import dsl
from kfp.dsl import component, OutputPath

@component(
    packages_to_install=['kaggle'],
)
def download_dataset_component(
    dataset: str,
    download_path: OutputPath(),
):
    import os
    from kaggle.api.kaggle_api_extended import KaggleApi
    import zipfile

    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print(f"Downloading dataset '{dataset}'...")
    api.dataset_download_files(dataset, path=download_path, unzip=False)

    # Unzip all downloaded zip files
    for file in os.listdir(download_path):
        if file.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(download_path, file), 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print(f"Extracted {file}")
    print(f"Dataset downloaded and extracted to '{download_path}'.")
