from typing import List, Union
from .base_model import BaseModel, ModelConfig
from .dataset import Dataset


class Test:
    def __init__(self, config: ModelConfig):
        self.config = config

    def test_model(self, model: BaseModel, dataset: Dataset) -> Union[float, List[float]]:
        """
        Test a trained model on a given dataset

        Args:
            model (BaseModel): The trained model to test
            dataset (Dataset): The dataset to test the model on

        Returns:
            float or List[float]: The evaluation metric(s) on the test dataset
        """
        X_test, y_test = dataset.get_test_data()

        if self.config.problem_type == 'classification':
            return model.evaluate(X_test, y_test)
        elif self.config.problem_type == 'regression':
            y_pred = model.predict(X_test)
            return model.evaluate(y_test, y_pred)
        else:
            raise ValueError(f"Unsupported problem type: {self.config.problem_type}")
