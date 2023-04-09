from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from .base_model import BaseModel
from typing import Dict, List
from omegaconf import DictConfig
from dataclasses import dataclass

@dataclass
class NeuralNetworkConfig:
    optimizer: str = 'adam'
    loss_fn : str = 'categorical_crossentropy'
    metrics: List[str] = ['accuracy']
    epochs: int = 10
    batch_size: int = 32
    input_shape: int = 11
        
    def __init__(self, config: DictConfig) -> None:
        self.optimizer = config.optimizer
        self.loss_fn = config.loss_fn
        self.metrics = config.metrics
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.input_shape = config.input_shape


class NeuralNetwork(BaseModel):
    properties: NeuralNetworkConfig
    model: tensorflow.keras.models
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.properties = NeuralNetworkConfig(config)
        self.model = None

    def build(self) -> None:
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=(self.properties.input_shape,)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
                
    def train(self, X_train, y_train) -> None:
        self.model.compile(optimizer=self.properties.optimizer, loss=self.properties.loss_fn, metrics=self.properties.metrics)
        self.model.fit(X_train, y_train, epochs=self.properties.epochs, batch_size=self.properties.batch_size)

    def predict(self, X_test) -> None:
        return self.model.predict(X_test)

