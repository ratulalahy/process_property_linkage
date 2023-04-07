import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class FeatureImportance:
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        
    def get_feature_importance(self) -> None:
        result = permutation_importance(self.model, self.X_train, self.y_train, n_repeats=10, random_state=42)
        sorted_idx = result.importances_mean.argsort()
        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=self.X_train.columns[sorted_idx])
        ax.set_title("Permutation Importance")
        fig.tight_layout()
        plt.show()
