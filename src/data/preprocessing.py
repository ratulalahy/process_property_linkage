import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from typing import Tuple

class Preprocessing:
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        self.X_train = X_train
        self.X_test = X_test
        
    def handle_missing_values(self) -> None:
        self.X_train.fillna(self.X_train.median(), inplace=True)
        self.X_test.fillna(self.X_test.median(), inplace=True)
        
    def handle_outliers(self) -> None:
        pass  # Implement outlier handling
    
    def encode_categorical_vars(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ohe = OneHotEncoder()
        X_train_encoded = ohe.fit_transform(self.X_train.select_dtypes(include=['object']))
        X_test_encoded = ohe.transform(self.X_test.select_dtypes(include=['object']))
        return X_train_encoded, X_test_encoded
    
    def normalize_numerical_vars(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train.select_dtypes(include=['float64']))
        X_test_scaled = scaler.transform(self.X_test.select_dtypes(include=['float64']))
        return X_train_scaled, X_test_scaled
    
    def oversample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        oversampler = RandomOverSampler(random_state=42)
        X_train_oversampled, y_train_oversampled = oversampler.fit_resample(self.X_train, y_train)
        return X_train_oversampled, y_train_oversampled
