from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
from typing import Tuple, Dict, Any, List


@dataclass
class ModelConfig:
    ml_platform: str 
    ml_algo: str 
    optimizer: str
    loss: str
    metrics: List[str]
    epochs: int
    batch_size: int
    model_params: Dict[str, Any] = field(default_factory=dict)

class BaseModel(ABC):
    """Base class for all machine learning models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    @abstractmethod
    def build_model(self):
        pass


    def compile_model(self):
        pass
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model

        Args:
            X_train (pd.DataFrame): Features of the training set
            y_train (pd.Series): Target variable of the training set

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Make predictions using the trained model

        Args:
            X_test (pd.DataFrame): Features of the test set

        Returns:
            pd.Series: Predictions of the target variable for the test set
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
        """Evaluate the model on the test set

        Args:
            X_test (pd.DataFrame): Features of the test set
            y_test (pd.Series): Target variable of the test set

        Returns:
            Tuple[float, float]: The evaluation metrics (e.g. accuracy, f1-score)
        """
        pass
