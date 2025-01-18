import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_walmart_data(dataset_name: str, download_path: str):
    # Configurar API de Kaggle
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), 'secrets')
    api = KaggleApi()
    api.authenticate()

    # Descargar dataset
    print(f"Descargando dataset {dataset_name} en {download_path}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset descargado y descomprimido en {download_path}.")
