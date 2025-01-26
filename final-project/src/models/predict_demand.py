import joblib
import pandas as pd

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)

def predict_demand(model, input_data):
    if isinstance(input_data, pd.DataFrame):
        predictions = model.predict(input_data)
        return predictions
    else:
        raise ValueError("Input data must be a Pandas DataFrame.")
