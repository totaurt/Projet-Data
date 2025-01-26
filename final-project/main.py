from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
from src.data.preprocess import load_and_clean_data
from src.models.hyperparameter_tuning import tune_hyperparameters
from src.models.feature_importance import plot_feature_importance
from src.models.evaluate_models import evaluate_models
import os

# Define the directory for saving plots
plots_dir = "final-project/plots"
os.makedirs(plots_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Paths
raw_data_path = "final-project/data/raw/Walmart.csv"
preprocessed_path = "final-project/data/preprocessed/preprocessed_data.csv"
train_path = "final-project/data/preprocessed/train_data.csv"
test_path = "final-project/data/preprocessed/test_data.csv"
model_path = "final-project/models/best_model.pkl"

# Load and preprocess data
print("Preprocessing data...")
train_data, test_data = load_and_clean_data(raw_data_path, preprocessed_path, train_path, test_path)

# Separate features and target
X_train = train_data.drop('actual_demand', axis=1)
y_train = train_data['actual_demand']
X_test = test_data.drop('actual_demand', axis=1)
y_test = test_data['actual_demand']

# Drop ID columns for model training and testing
id_columns = ['customer_id', 'product_id', 'store_id', 'supplier_id']
X_train = X_train.drop(columns=id_columns, errors='ignore')
X_test = X_test.drop(columns=id_columns, errors='ignore')

# Check for non-numeric columns
non_numeric_columns = X_train.select_dtypes(include=['object']).columns
if not non_numeric_columns.empty:
    print(f"Non-numeric columns in X_train: {non_numeric_columns}")
    raise ValueError("There are non-numeric columns in X_train. Ensure preprocessing handles them.")

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
print("Tuning hyperparameters...")
best_models = tune_hyperparameters(X_train_scaled, y_train)

# Evaluate models
results_df = evaluate_models(best_models, X_test_scaled, y_test)

# Save and deploy the best model
best_model_name = results_df.loc[results_df["R²"].idxmax()]["Model"]
best_model = best_models[best_model_name]
joblib.dump(best_model, model_path)
print(f"Best model saved: {best_model_name}")

# Feature importance
plot_feature_importance(best_model, X_train.columns, save_path=f"{plots_dir}/feature_importance.png")

# Visualize R² scores
plt.figure(figsize=(10, 6))
sns.barplot(y="Model", x="R²", data=results_df, palette="viridis")
plt.xlabel("R² Score")
plt.ylabel("Model Name")
plt.title("Model Performance (R² Score)")
plt.savefig(f"{plots_dir}/model_performance_r2.png")
plt.show()
