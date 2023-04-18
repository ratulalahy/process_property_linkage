from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

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
    
    def calculate_r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    def calculate_rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
