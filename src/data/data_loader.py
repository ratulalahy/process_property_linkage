from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


@dataclass
class DatasetProperties:
    file_path: Optional[Path] = None
    target_cols: Optional[Tuple[str, str, str]] = None
    feature_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    numerical_columns: Optional[List[str]] = None
    test_size : float = 0.2
    valid_size : float = 0.2
    

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
    
    
    def __init__(self, props: DatasetProperties = None, delimiter: str = ',', index_col : List[int]= []) -> None:
        self.prop = props
        if (self.prop is not None and self.prop.file_path is not None):
            self.data = pd.read_csv(self.prop.file_path, delimiter=delimiter, index_col=index_col)
        
    def split_train_test(self, random_state: int = 42)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the data into training and test sets."""
        X = self.data.drop(list(self.prop.target_cols), axis=1)
        y = self.data[list(self.prop.target_cols)]
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_val_test_split(X, y, test_size= self.prop.test_size, val_size=self.prop.val_size)
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        return train_data, val_data, test_data
    
    def train_val_test_split(X, y, test_size: float = 0.2, val_size: float = 0.2, random_state = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
