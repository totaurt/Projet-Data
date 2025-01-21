from src.data.load_data import download_walmart_data
from src.data.preprocess import load_and_clean_data
import os

# Define paths
dataset_name = "ankitrajmishra/walmart"  # Kaggle dataset name
raw_path = "final-project/data/raw"  # Folder to download the raw data
preprocessed_path = "final-project/data/preprocessed/preprocessed_data.csv"
train_path = "final-project/data/preprocessed/train_data.csv"
test_path = "final-project/data/preprocessed/test_data.csv"

# Download data if it doesn't exist
print("Checking and downloading raw data if necessary...")
download_walmart_data(dataset_name, raw_path)

# Get the path of the raw data file
raw_data_file_name = os.listdir(raw_path)[0]
raw_data_path = os.path.join(raw_path, raw_data_file_name)

# Preprocess data and split into train/test sets
print("Starting preprocessing...")
load_and_clean_data(raw_data_path, preprocessed_path, train_path, test_path)

print("Pipeline completed successfully!")


