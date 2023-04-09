from abc import ABC, abstractmethod
from typing import Dict
from omegaconf import DictConfig

class BaseModel(ABC):
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    @abstractmethod
    def train(self, X_train, y_train) -> None:
        pass

    @abstractmethod
    def predict(self, X_test) -> None:
        pass
