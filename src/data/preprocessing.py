from enum import Enum
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.over_sampling import RandomOverSampler
from typing import Tuple, Dict, Union, List, Any, Optional
from dataclasses import dataclass
from scipy.sparse import spmatrix
import smogn

from src.data.data_loader import DataSet

class NormalizationMethod(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MANUAL = "manual"


class MissingValueMethod(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    KNN = "knn"


class OversamplingMethod(str, Enum):
    SMOTE = "smote"
    ADASYN = "adasyn"
    RANDOM = "random"

@dataclass
class PreprocessingConfig:
    missing_values: MissingValueMethod = MissingValueMethod.MEDIAN
    normalization: NormalizationMethod = NormalizationMethod.STANDARD
    oversampler: OversamplingMethod = OversamplingMethod.SMOTE
    

@dataclass
class Preprocessing:
    config: PreprocessingConfig
    
    def __init__(self, config: PreprocessingConfig) -> None:      
        self.config = config 

    def handle_missing_values(self, data : pd.DataFrame) -> None:
        if self.config.missing_values == 'median':
            data.data.fillna(data.data.median(), inplace=True)
        else:
            raise ValueError(f"Unsupported scaler: {self.config.missing_values}")
        
    def _manual_normalize(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, Union[np.ndarray, spmatrix, None]]:
        X_train_std = X_train.std()
        X_train_mean = X_train.mean()

        X_train_normalized = pd.DataFrame()
        for column in range(len(X_train.columns)):
            X_train_normalized[X_train.columns[column]] = (X_train.iloc[:, column] - X_train_mean[column]) / X_train_std[column]

        X_test_normalized = pd.DataFrame()
        for column in range(len(X_test.columns)):
            X_test_normalized[X_test.columns[column]] = (X_test.iloc[:, column] - X_train_mean[column]) / X_train_std[column]
        return X_train_normalized.values, X_test_normalized.values
    
    def handle_outliers(self) -> None:
        pass  # Implement outlier handling

    def normalize(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Union[np.ndarray, spmatrix, None]]:
        if self.config.normalization == NormalizationMethod.STANDARD:
            self.scaler = StandardScaler()
        elif self.config.normalization == NormalizationMethod.MINMAX:
            self.scaler = MinMaxScaler()
        elif self.config.normalization == NormalizationMethod.ROBUST:
            self.scaler = RobustScaler()
        elif self.config.normalization == NormalizationMethod.MANUAL:
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test) if X_test is not None else None
            return self._manual_normalize(X_train_df, X_test_df)            
        else:
            raise ValueError(f"Invalid scaler method: {self.config.normalization}")

        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled

        return X_train_scaled, None

    def oversample_data(self, data: pd.DataFrame, target_column: str, **params) ->pd.DataFrame:
        if self.config.oversampler == 'smote':
            self.oversampler = smogn.smoter
        else:
            raise ValueError(f"Unsupported scaler: {self.config.oversampler}")        
        data_copy = data.copy()    
        data_oversampled = self.oversampler(data = data_copy, y = target_column, **params)
        return data_oversampled
