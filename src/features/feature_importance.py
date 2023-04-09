import shap
from typing import Dict
import pandas as pd

class FeatureImportance:
    def __init__(self, config: Dict) -> None:
        self.config = config

    def calculate_feature_importance(self, model, X_test) -> pd.DataFrame:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
        return shap_df.abs().mean().sort_values(ascending=False).reset_index().rename(columns={'index': 'feature', 0: 'importance'})
