from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
from typing import Dict, Any
from omegaconf import DictConfig

class LinearModel(BaseModel):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.model = LinearRegression()

    def train(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test) -> Any:
        return self.model.predict(X_test)
