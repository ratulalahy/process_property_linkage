import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from typing import Tuple, Dict, Union, List, Any
from dataclasses import dataclass
import smogn

from src.data.data_loader import DataSet

@dataclass
class PreprocessingConfig:
    missing_values: str = "median"
    scaler_method: str = "standard"
    oversampler: str = "smote"
    

@dataclass
class Preprocessing:
    scaler: Any
    config: PreprocessingConfig
    
        
    def handle_missing_values(self) -> None:
        if self.config.missing_values == 'median':
            self.dataset.data.fillna(self.dataset.data.median(), inplace=True)
        else:
            raise ValueError(f"Unsupported scaler: {self.config.missing_values}")
                
    def handle_outliers(self) -> None:
        pass  # Implement outlier handling

    def fit(self, X: pd.DataFrame) -> 'Preprocessing':
        """Fit the preprocessing pipeline.

        Args:
            X (pd.DataFrame): The training input samples.

        Returns:
            Preprocessing: The fitted preprocessing pipeline.
        """
        self.scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the preprocessing pipeline.

        Args:
            X (pd.DataFrame): The input samples.

        Returns:
            pd.DataFrame: The transformed samples.
        """
        X = self.scaler.transform(X)
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and apply the preprocessing pipeline.

        Args:
            X (pd.DataFrame): The input samples.

        Returns:
            pd.DataFrame: The transformed samples.
        """
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        return X
    
    def normalize(self, X_train: pd.DataFrame, X_test: Union[None, pd.DataFrame]=None) -> Union[pd.DataFrame, tuple]:
        """Normalize the input data.

        Args:
            X_train (pd.DataFrame): The training input samples.
            X_test (pd.DataFrame): The testing input samples.

        Returns:
            Union[pd.DataFrame, tuple]: The normalized training samples and the normalized testing samples, if `X_test` is not None. Otherwise, only the normalized training samples are returned.
        """
        if not X_test:
            return self.fit_transform(X_train)

        X_train_normalized = self.fit_transform(X_train)
        X_test_normalized = self.transform(X_test)
        return X_train_normalized, X_test_normalized
    
    def oversample_data(self, target_column: str, **params) ->pd.DataFrame:
        """_summary_

        Args:
            target_column (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
        
        Note:
            default smogn first remove columns containing missing values and then rows containing missing values
        """
        if self.config.oversampler == 'smote':
            self.oversampler = smogn.smoter
        else:
            raise ValueError(f"Unsupported scaler: {self.config.oversampler}")        
        data_copy = self.dataset.data.copy()    
        data_oversampled = self.oversampler(data = data_copy, y = target_column, **params)
        return data_oversampled

