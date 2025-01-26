import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import category_encoders as ce
from sklearn.impute import SimpleImputer


def impute_missing_values(data):
    """
    Handles missing values for both numeric and categorical columns.
    """
    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Define imputers for numeric and categorical columns
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Apply imputers
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    return data


def load_and_clean_data(raw_data_path, preprocessed_path, train_path, test_path, test_size=0.2):
    # Load raw data
    print("Loading raw data...")
    data = pd.read_csv(raw_data_path)
    
    # Drop unnecessary columns
    print("Dropping unnecessary columns...")
    columns_to_drop = ['transaction_id', 'forecasted_demand']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    # Create new time-based features
    print("Creating time-based features...")
    data['transaction_month'] = pd.to_datetime(data['transaction_date']).dt.month
    data['transaction_hour'] = pd.to_datetime(data['transaction_date']).dt.hour
    data = data.drop('transaction_date', axis=1)

    # Handle missing values
    print("Handling missing values...")
    data = impute_missing_values(data)

    # One-Hot Encoding for low-cardinality features
    print("Applying one-hot encoding...")
    one_hot_columns = ['weekday', 'weather_conditions', 'payment_method', 'store_location', 
                   'category', 'customer_gender', 'promotion_type']  # Added 'promotion_type'
    data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True)

    # Target Encoding for high-cardinality features
    print("Applying target encoding...")
    target_encoding_columns = ['product_id', 'store_id', 'supplier_id', 'customer_id', 'product_name']  # Added 'product_name'
    target_encoder = ce.TargetEncoder(cols=target_encoding_columns)
    data[target_encoding_columns] = target_encoder.fit_transform(data[target_encoding_columns], data['actual_demand'])


    # Encode ordinal columns
    print("Applying ordinal encoding...")
    ordinal_columns = ['customer_income', 'customer_loyalty_level']
    label_encoder = ce.OrdinalEncoder(cols=ordinal_columns)
    data[ordinal_columns] = label_encoder.fit_transform(data[ordinal_columns])

    # Drop high-cardinality identifiers
    columns_to_drop = ['customer_id', 'product_id', 'supplier_id']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    # Split data into train and test sets
    print("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # Save processed data
    print("Saving processed data...")
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Preprocessing complete. Train and test data saved at {train_path} and {test_path}")
    return train_data, test_data



