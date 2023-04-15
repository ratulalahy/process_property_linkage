from .metrics import calculate_r2, calculate_rmse
from typing import Dict, List
import pandas as pd
from omegaconf import DictConfig

class Evaluator:
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def evaluate_models(self, models: Dict, X_test, y_test) -> pd.DataFrame:
        results = []
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            r2 = calculate_r2(y_test, y_pred)
            rmse = calculate_rmse(y_test, y_pred)
            results.append({
                'model': model_name,
                'r2': r2,
                'rmse': rmse
            })
        results_df = pd.DataFrame(results)
        return results_df