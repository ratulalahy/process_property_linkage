from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import List
import pandas as pd

class TrainModel:
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def train_linear_regression(self) -> LinearRegression:
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        return lr
    
    def train_random_forest(self, n_estimators: int, max_depth: int) -> RandomForestRegressor:
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(self.X_train, self.y_train)
        return rf
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)
    
    def compare_models(self, models: List) -> List:
        results = []
        for model in models:
            fitted_model = model.fit(self.X_train, self.y_train)
            mse = self.evaluate_model(fitted_model, self.X_test, self.y_test)
            results.append((model.__class__.__name__, mse))
        return results

