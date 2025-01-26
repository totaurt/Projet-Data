from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_models(models, X_test, y_test):
    results = []
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": model_name, "MSE": mse, "MAE": mae, "R²": r2})
        print(f"{model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
    
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    sns.barplot(x="R²", y="Model", data=results_df, palette="viridis")
    plt.title("Model Performance (R² Score)")
    plt.xlabel("R² Score")
    plt.ylabel("Model")
    
    # Save the plot
    plot_path = "final-project/plots/model_performance.png"
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Model performance plot saved at: {plot_path}")
    plt.close()
    
    return results_df

