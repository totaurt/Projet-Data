from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

# Define a custom wrapper to avoid compatibility issues
class CompatibleXGBRegressor(XGBRegressor):
    def __sklearn_tags__(self):
        return {"binary_only": False, "multioutput": False}

def tune_hyperparameters(X_train, y_train):
    warnings.filterwarnings("ignore", category=UserWarning)

    models = {
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": CompatibleXGBRegressor()
    }
    
    param_grids = {
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    }

    best_models = {}

    for model_name, model in models.items():
        print(f"Tuning hyperparameters for {model_name}...")
        param_grid = param_grids[model_name]
        random_search = RandomizedSearchCV(
            model, param_grid, cv=3, n_iter=10, scoring="neg_mean_squared_error", random_state=42
        )
        try:
            random_search.fit(X_train, y_train)
            print(f"Best parameters for {model_name}: {random_search.best_params_}")
            best_models[model_name] = random_search.best_estimator_
        except Exception as e:
            print(f"Error tuning {model_name}: {e}")

    return best_models
