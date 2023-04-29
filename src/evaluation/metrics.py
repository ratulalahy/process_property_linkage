from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
# from typing import Float
class Metrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred
        
    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)
    
    def precision(self) -> float:
        return precision_score(self.y_true, self.y_pred)
    
    def recall(self) -> float:
        return recall_score(self.y_true, self.y_pred)
    
    def f1_score(self) -> float:
        return f1_score(self.y_true, self.y_pred)
    
    def calculate_r2(self) -> float:
        return r2_score(self.y_true, self.y_pred)

    def calculate_rmse(self) -> float:
        return mean_squared_error(self.y_true, self.y_pred, squared=False)
