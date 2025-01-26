import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_feature_importance(model, feature_names, save_path=None):
    """Plot and optionally save feature importance.

    Args:
        model: Trained model with feature importance attributes.
        feature_names: List of feature names.
        save_path: Path to save the plot. If None, the plot will only be displayed.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        raise ValueError("The provided model does not support feature importance extraction.")

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved at: {save_path}")
    
    plt.show()

