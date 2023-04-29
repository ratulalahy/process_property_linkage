from typing import List, Tuple
from tensorflow import keras
from dataclasses import dataclass, field
from .base_model import BaseModel, ModelConfig
from ..evaluation.metrics import Metrics
import numpy as np
from typing import Tuple, Any, Optional,Dict
from sklearn.neural_network import MLPRegressor


@dataclass
class NeuralNetConfig(ModelConfig):
    """Dataclass for neural network configuration"""
    ml_platform: str = 'sklearn'
    ml_algo: str = 'mlp'
    optimizer: str = 'lbfgs'
    loss: str = 'mse'
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'precision', 'recall','f1_score', 'r2', 'rmse'])
    epochs: int = 500
    batch_size: int = 32
    hidden_layer_size: int = 100
    activation: str = 'relu'
    model_params: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = True
    random_state: int = 42
    
class NeuralNet(BaseModel):
    """Class for neural network model"""
    
    def __init__(self, config: NeuralNetConfig):
        super().__init__(config)
        self.model: Optional[Any] = None  # Add the type hint here
    
    def build_model(self) -> Any:
        """Build the neural network model"""
        ml_platform = self.config.ml_platform.lower()
        ml_algo = self.config.ml_algo.lower()
        model = None
        if ml_platform == 'sklearn':
            if ml_algo == 'mlp':
                from sklearn.neural_network import MLPRegressor
                model = MLPRegressor(
                    hidden_layer_sizes=self.config.hidden_layer_size,
                    activation=self.config.activation,
                    solver=self.config.optimizer,
                    alpha=self.config.model_params.get('alpha', 0.0001),
                    learning_rate_init=self.config.model_params.get('learning_rate', 0.001),
                    max_iter=self.config.epochs,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state
                )
            else:
                raise ValueError(f"Unsupported ML algorithm: {ml_algo}")
        else:
            raise ValueError(f"Unsupported ML platform: {ml_platform}")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the neural network model"""
        model = self.build_model()
        model.fit(X_train, y_train)
        self.model = model
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the output of the neural network model"""
        return self.model.predict(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the neural network model"""    
        y_pred = self.model.predict(X_test)
        matrix = Metrics(y_true = y_test, y_pred = y_pred)
        accuracy = matrix.accuracy()
        precision = matrix.precision()
        recall = matrix.recall()
        r2 = matrix.calculate_r2()
        f1_score = matrix.f1_score()
        rmse = matrix.calculate_rmse()
        return{
            'ml_algo': self.config.ml_algo,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'r2': r2,
            'f1_score': f1_score,
            'rmse': rmse
        }