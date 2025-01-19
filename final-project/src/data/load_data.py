import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_walmart_data(dataset_name: str, download_path: str):
    # Check if dataset already exists
    if os.path.exists(download_path) and any(os.scandir(download_path)):
        print(f"Dataset already exists in {download_path}. Skipping download.")
        return  # Skip download if dataset already exists
    
    os.makedirs(download_path, exist_ok=True)

    # Configure Kaggle API
    print(f"Downloading dataset {dataset_name} to {download_path}...")
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), 'secrets')
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset downloaded and extracted to {download_path}.")


# Add test code below to test the function when this script is run
if __name__ == "__main__":
    # Define dataset name and download path
    dataset_name = 'ankitrajmishra/walmart'  # Name of the dataset on Kaggle
    download_path = 'final-project/data/raw'  # Path to save the dataset

    # Call the function to download the dataset
    download_walmart_data(dataset_name, download_path)