from enum import Enum
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error, 
    mean_squared_log_error
)
import numpy as np


class MetricName(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    # Regression Metrics
    MEAN_SQUARED_ERROR = "mean_squared_error"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    R2_SCORE = "r2_score"
    ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
    MEAN_SQUARED_LOG_ERROR = "mean_squared_log_error"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error"

def _root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def _mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Metrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred
        

        
    def calculate(self, metrics: List[MetricName]) -> Dict:
        results = {}
        for metric in metrics:
            if metric == MetricName.ACCURACY:
                results[MetricName.ACCURACY.value] = accuracy_score(self.y_true, self.y_pred)
            elif metric == MetricName.PRECISION.value:
                results[MetricName.PRECISION.value] = precision_score(self.y_true, self.y_pred)
            elif metric == MetricName.RECALL.value:
                results[MetricName.RECALL.value] = recall_score(self.y_true, self.y_pred)
            elif metric == MetricName.F1_SCORE.value:
                results[MetricName.F1_SCORE.value] = f1_score(self.y_true, self.y_pred)
                
            # Regression Metrics
            elif metric == MetricName.MEAN_SQUARED_ERROR.value:
                results[MetricName.MEAN_SQUARED_ERROR.value] = mean_squared_error(self.y_true, self.y_pred)
            elif metric == MetricName.MEAN_ABSOLUTE_ERROR.value:
                results[MetricName.MEAN_ABSOLUTE_ERROR.value] = mean_absolute_error(self.y_true, self.y_pred)
            elif metric == MetricName.MEDIAN_ABSOLUTE_ERROR.value:
                results[MetricName.MEDIAN_ABSOLUTE_ERROR.value] = median_absolute_error(self.y_true, self.y_pred)
            elif metric == MetricName.R2_SCORE.value:
                results[MetricName.R2_SCORE.value] = r2_score(self.y_true, self.y_pred)
            elif metric == MetricName.ROOT_MEAN_SQUARED_ERROR.value:
                results[MetricName.ROOT_MEAN_SQUARED_ERROR.value] = _root_mean_squared_error(self.y_true, self.y_pred)
            elif metric == MetricName.MEAN_SQUARED_LOG_ERROR.value:
                results[MetricName.MEAN_SQUARED_LOG_ERROR.value] = mean_squared_log_error(self.y_true, self.y_pred)
            elif metric == MetricName.MEAN_ABSOLUTE_PERCENTAGE_ERROR.value:
                results[MetricName.MEAN_ABSOLUTE_PERCENTAGE_ERROR.value] = _mean_absolute_percentage_error(self.y_true, self.y_pred)
                
        return results
