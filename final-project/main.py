from src.data.load_data import download_walmart_data
from src.data.preprocess import load_and_clean_data
import os

# Define paths
dataset_name = "ankitrajmishra/walmart"  # Kaggle dataset name
raw_path = "final-project/data/raw"  # Folder to download the raw data
preprocessed_path = "final-project/data/preprocessed"

# Download data if it doesn't exist
download_walmart_data(dataset_name, raw_path)

raw_data_file_name = os.listdir(raw_path)[0]
raw_data_path = raw_path + "/" + raw_data_file_name

# Load and preprocess data if it's not already processed
data = load_and_clean_data(raw_data_path, preprocessed_path)
