from .metrics import Metrics
from typing import Dict, List
import pandas as pd
from omegaconf import DictConfig

class Evaluator:
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def evaluate_model(self, models: Dict, X_test, y_test, metric_names: List[str]) -> pd.DataFrame:
        results = []
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            metrics = Metrics(y_test, y_pred)
            result_dict = {'model': model_name}
            for metric_name in metric_names:
                metric_func = getattr(metrics, metric_name, None)
                if metric_func is not None and callable(metric_func):
                    result_dict[metric_name] = metric_func()
            results.append(result_dict)
        results_df = pd.DataFrame(results)
        return results_df
