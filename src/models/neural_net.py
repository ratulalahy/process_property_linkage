from typing import List, Tuple
from tensorflow import keras
from dataclasses import dataclass, field
from .base_model import BaseModel, ModelConfig
from ..evaluation.metrics import Metrics, MetricName
import numpy as np
from typing import Tuple, Any, Optional,Dict, Literal,Union
from sklearn.neural_network import MLPRegressor, MLPClassifier



@dataclass
class NeuralNetConfig(ModelConfig):
    """Dataclass for neural network configuration

    Args:
        ModelConfig (_type_): _description_
    """
    ml_platform: Literal['sklearn'] = 'sklearn'
    ml_algo: Literal['mlp'] = 'mlp'
    optimizer: Literal['lbfgs', 'sgd', 'adam'] = 'lbfgs'
    loss: Literal['mse'] = 'mse'
    #metrics: List[str] = field(default_factory=lambda: ['accuracy', 'precision', 'recall','f1_score', 'r2', 'rmse'])
    #metrics: List[MetricName] = field(default_factory=lambda: [MetricName.ACCURACY, MetricName.PRECISION, MetricName.RECALL, MetricName.F1_SCORE])
    epochs: int = 500
    batch_size: int = 32
    hidden_layer_size: int = 100
    activation: Literal['relu', 'identity', 'logistic', 'tanh'] = 'relu'
    model_params: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = True
    random_state: int = 42
    
class NeuralNet(BaseModel):
    """Class for neural network model

    Args:
        BaseModel (_type_): _description_
    """
    
    def __init__(self, config: NeuralNetConfig):
        # super().__init__(config)
        self.model: Any = None
        self.config = config
    
    def build_model(self) -> Optional[MLPRegressor]:
        """Build the neural network model

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            Optional[MLPRegressor]: _description_
        """
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
        """Train the neural network model

        Args:
            X_train (np.ndarray): _description_
            y_train (np.ndarray): _description_
        """
        model = self.build_model()
        model.fit(X_train, y_train)
        self.model = model
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the output of the neural network model

        Args:
            X_test (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.model.predict(X_test)

    def evaluate(self, X_inp: np.ndarray, y_true: np.ndarray, metric_names: List[MetricName]) -> Dict:
        """Evaluate the neural network model

        Args:
            X_test (np.ndarray): _description_
            y_test (np.ndarray): _description_
            metric_names (List[MetricName]): _description_

        Returns:
            Dict: _description_
        """
        y_pred = self.model.predict(X_inp)
        metrics_instance = Metrics(y_true, y_pred)
        result_dict = metrics_instance.calculate(metric_names)
        return result_dict
