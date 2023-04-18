from typing import List, Tuple
from tensorflow import keras
from dataclasses import dataclass
from .base_model import BaseModel, ModelConfig
import numpy as np
from typing import Tuple, Any
from sklearn.neural_network import MLPRegressor

from base_model import BaseModel, ModelConfig

@dataclass
class NeuralNetConfig(ModelConfig):
    """Dataclass for neural network configuration"""
    hidden_layer_size: int
    activation: str = 'relu'
    
class NeuralNet(BaseModel):
    """Class for neural network model"""
    
    def __init__(self, config: NeuralNetConfig):
        super().__init__(config)
    
    def build_model(self, **kwargs) -> Any:
        """Build the neural network model"""
        ml_platform = self.config.ml_platform.lower()
        algorithm = self.config.model_type.lower()
        
        if ml_platform == 'sklearn':
            if algorithm == 'mlp':
                from sklearn.neural_network import MLPRegressor
                model = MLPRegressor(
                    hidden_layer_sizes=self.config.hidden_layer_size,
                    activation=self.config.activation,
                    solver=self.config.optimizer,
                    alpha=kwargs.get('alpha', 0.0001),
                    batch_size=self.config.batch_size,
                    learning_rate_init=kwargs.get('learning_rate', 0.001),
                    max_iter=self.config.epochs
                )
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the neural network model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the output of the neural network model"""
        return self.model.predict(X_test)
