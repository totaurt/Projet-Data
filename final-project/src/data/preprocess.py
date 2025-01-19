import os
import pandas as pd
import numpy as np

def load_and_clean_data(raw_data_path, preprocessed_path):
    # Check if the preprocessed file exists
    if os.path.exists(preprocessed_path):
        print(f"Preprocessed data already exists at {preprocessed_path}. Skipping preprocessing.")
        # code have to find the preprocessed data file name before trying to load it
        # return pd.read_csv(preprocessed_data_path)  # Return the preprocessed data without processing

    os.makedirs(preprocessed_path, exist_ok=True)
    
    # Load the dataset
    print(f"Loading data from {raw_data_path}...")
    data = pd.read_csv(raw_data_path)
    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
    
    # Handle missing values
    print("Handling missing values...")
    data = data.dropna()  # Drop rows with missing values, or you could use data.fillna() depending on your strategy
    
    # Drop duplicates (if any)
    print("Removing duplicates...")
    data = data.drop_duplicates()
    
    # Convert date columns to datetime (if applicable)
    print("Converting date columns to datetime...")
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    
    # Feature engineering (Example: Create a new feature based on existing ones)
    print("Creating new features...")
    if 'sales' in data.columns:
        data['log_sales'] = np.log1p(data['sales'])  # Log transformation to stabilize variance
    
    # Encoding categorical variables (if any)
    print("Encoding categorical variables...")
    if 'store' in data.columns:
        data['store'] = data['store'].astype('category').cat.codes  # Encoding categorical variable 'store' to numeric
    
    # Normalize numerical columns (if applicable)
    print("Normalizing numerical columns...")
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()  # Z-score normalization
    
    # Save preprocessed data
    preprocessed_data_path = preprocessed_path + "/preprocessed_data.csv"
    print(f"Saving preprocessed data to {preprocessed_data_path}...")
    data.to_csv(preprocessed_data_path, index=False)
    print("Data saved successfully.")
    
    return data
