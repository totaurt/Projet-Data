import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

def load_and_clean_data(raw_data_path, preprocessed_path, train_path, test_path, test_size=0.2):
    # Check if the preprocessed file exists
    if os.path.exists(preprocessed_path):
        print(f"Preprocessed data already exists at {preprocessed_path}. Skipping preprocessing.")
        return pd.read_csv(preprocessed_path)  # Return the preprocessed data without processing

    os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
    
    # Load the dataset
    print(f"Loading data from {raw_data_path}...")
    data = pd.read_csv(raw_data_path)
    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
    
    data.stockout_indicator = data.stockout_indicator.astype(int)
    data.holiday_indicator = data.holiday_indicator.astype(int)
    data.promotion_applied = data.promotion_applied.astype(int)

    # Handle missing values
    print("Handling missing values...")
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col].fillna(data[col].mean(), inplace=True)
    for col in data.select_dtypes(include=['object', 'category']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    print("Handling outliers...")
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower_bound, upper_bound)
    
    # Drop duplicates (if any)
    print("Removing duplicates...")
    data = data.drop_duplicates()
    
    # Convert date columns to datetime (if applicable)
    print("Converting date columns to datetime...")
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    
    # Feature engineering (Example: Create a new feature based on existing ones)
    print("Creating new features...")
    if 'quantity_sold' in data.columns and 'unit_price' in data.columns:
        data['total_revenue'] = data['quantity_sold'] * data['unit_price']
    
    if 'transaction_date' in data.columns:
        data['transaction_date'] = pd.to_datetime(data['transaction_date'])
        data['year'] = data['transaction_date'].dt.year
        data['month'] = data['transaction_date'].dt.month
        data['day_of_week'] = data['transaction_date'].dt.dayofweek 
        
    if 'inventory_level' in data.columns and 'reorder_point' in data.columns:
        data['inventory_below_reorder'] = (data['inventory_level'] < data['reorder_point']).astype(int)

    print("Selecting features based on correlation...")
    correlation_matrix = data.corr()
    high_corr_features = [
        column for column in correlation_matrix.columns
        if any(abs(correlation_matrix[column]) > 0.9) and column != correlation_matrix.columns[0]
    ]
    print(f"Removing highly correlated features: {high_corr_features}")
    data = data.drop(columns=high_corr_features)

    print("Normalizing numerical columns...")
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # One-Hot 
    print("Applying One-Hot Encoding...")
    data = pd.get_dummies(data, columns=['category', 'promotion_type'], drop_first=True)

    
    print("Applying Ordinal Encoding...")
    loyalty_mapping = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
    if 'customer_loyalty_level' in data.columns:
        data['customer_loyalty_level'] = data['customer_loyalty_level'].map(loyalty_mapping)
    
    # Encoding categorical variables
    print("Encoding categorical variables...")
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    encoder = OrdinalEncoder()
    data[categorical_cols] = encoder.fit_transform(data[categorical_cols])
    
    
    # Save preprocessed data
    preprocessed_data_path = preprocessed_path + "/preprocessed_data.csv"
    print(f"Saving preprocessed data to {preprocessed_data_path}...")
    data.to_csv(preprocessed_data_path, index=False)
    print("Data saved successfully.")
    
    # Split data into training and test sets
    print("Splitting data into training and test sets...")
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # Save training and test sets
    print(f"Saving training data to {train_path}...")
    train_data.to_csv(train_path, index=False)
    print(f"Saving test data to {test_path}...")
    test_data.to_csv(test_path, index=False)

    print("Data split and saved successfully.")
    return train_data, test_data
