from dataclasses import dataclass
import pandas as pd
from typing import List
from omegaconf import DictConfig

@dataclass
class DatasetProperties:
    file_path: str
    target_cols: Tuple[str, str, str]
    feature_columns: List[str] 
    target_columns: Tuple[str, str, str] 
    categorical_columns: List[str] 
    numerical_columns: List[str]

@dataclass
class DataSet:
    data: pd.DataFrame
    X_train = None
    X_test = None
    X_val = None
    y_train = None
    y_test = None
    y_val = None
    prop:  DatasetProperties
        
    def split_train_test(self, test_size: float, val_size: float, random_state: int = 42)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the data into training and test sets."""
        X = self.data.drop(list(self.prop.target_cols), axis=1)
        y = self.data[list(self.prop.target_cols)]
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_val_test_split(X, y, test_size=self.config.test_size, val_size=self.config.val_size)
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        return train_data, val_data, test_data
    
    def train_val_test_split(X, y, test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def impute_missing_values(self, impute_categorical: str = "mode", impute_numerical: str = "mean"):
        """Imputes missing values in the data."""
        from sklearn.impute import SimpleImputer
        
        # Impute categorical features
        if self.categorical_columns:
            cat_imputer = SimpleImputer(strategy=impute_categorical)
            self.X_train[self.categorical_columns] = cat_imputer.fit_transform(self.X_train[self.categorical_columns])
            self.X_test[self.categorical_columns] = cat_imputer.transform(self.X_test[self.categorical_columns])
        
        # Impute numerical features
        if self.numerical_columns:
            num_imputer = SimpleImputer(strategy=impute_numerical)
            self.X_train[self.numerical_columns] = num_imputer.fit_transform(self.X_train[self.numerical_columns])
            self.X_test[self.numerical_columns] = num_imputer.transform(self.X_test[self.numerical_columns])
    
    
    def scale_numerical_features(self, scaler: str = "standard"):
        """Scales numerical features."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        if scaler == "standard":
            standard_scaler = StandardScaler()
            self.X_train[self.numerical_columns] = standard_scaler.fit_transform(self.X_train[self.numerical_columns])
            self.X_test[self.numerical_columns] = standard_scaler.transform(self.X_test[self.numerical_columns])
        elif scaler == "minmax":
            minmax_scaler = MinMaxScaler()
            self.X_train[self.numerical_columns] = minmax_scaler.fit_transform(self.X
