from typing import Any, Dict
from src.models.base_model import BaseModel
from src.data.dataset import Dataset


class Validate:
    def __init__(self, model: BaseModel, dataset: Dataset):
        self.model = model
        self.dataset = dataset
    
    def validate_model(self) -> Dict[str, Any]:
        """Validate a trained model on a given dataset

        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics
        """
        X_val, y_val = self.dataset.get_validation_data()
        y_pred = self.model.predict(X_val)
        eval_metrics = self.model.evaluate(X_val, y_val)
        eval_metrics["r2_score"] = r2_score(y_val, y_pred)
        eval_metrics["mean_squared_error"] = mean_squared_error(y_val, y_pred)
        return eval_metrics
