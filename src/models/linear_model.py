from typing import List, Tuple
from tensorflow import keras
from dataclasses import dataclass
from .base_model import BaseModel, ModelConfig
import numpy as np
from typing import Tuple, Any, Dict
from sklearn.neural_network import MLPRegressor

from base_model import BaseModel, ModelConfig

@dataclass
class MLModelConfig(ModelConfig):
    """Dataclass for machine learning model configuration"""
    hyperparameters: Dict[str, Any] = {}


class MLModel(BaseModel):
    """Class for machine learning models"""
    
    def __init__(self, config: MLModelConfig):
        super().__init__(config)
    
    def build_model(self, **kwargs) -> Any:
        """Build the machine learning model"""
        ml_platform = self.config.ml_platform.lower()
        algorithm = self.config.model_type.lower()
        if algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        elif algorithm == 'xgboost':
            from xgboost import XGBRegressor
            model = XGBRegressor(objective='reg:squarederror')
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Set the hyperparameters
        model.set_params(**self.config.hyperparameters)
        self.model = model
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the machine learning model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the output of the machine learning model
