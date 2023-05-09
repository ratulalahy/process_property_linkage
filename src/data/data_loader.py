from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


@dataclass
class DatasetProperties:
    file_path: Optional[Path] = None
    target_cols: Optional[List[str]]  = field(default_factory=list)
    feature_columns: Optional[List[str]] = field(default_factory=list)
    categorical_columns: Optional[List[str]] = field(default_factory=list)
    numerical_columns: Optional[List[str]] = field(default_factory=list)
    # test_size : float = 0.2
    # valid_size : float = 0.2
    

@dataclass
class DataSet:
    data: pd.DataFrame
    X_train = Optional[pd.DataFrame]
    X_test = Optional[pd.DataFrame]
    X_val = None
    y_train = None
    y_test = None
    y_val = None
    prop:  DatasetProperties
    
    
    def __init__(self, props: DatasetProperties = None, delimiter: str = ',', index_col : List[int]= []) -> None:
        self.prop = props
        if (self.prop is not None and self.prop.file_path is not None):
            self.data = pd.read_csv(self.prop.file_path, delimiter=delimiter, index_col=index_col)
        
    def split_train_test(self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the data into training and test sets."""
        X = self.data[self.prop.feature_columns]
        y = self.data[list(self.prop.target_cols)]
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.train_val_test_split(test_size= test_size, val_size=val_size)
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        val_data = pd.concat([self.X_val, self.y_val], axis=1)
        test_data = pd.concat([self.X_test, self.y_test], axis=1)
        return train_data, val_data, test_data
    
    def train_val_test_split(self, test_size: float = 0.2, val_size: float = 0.2, random_state = 42) -> None:
        X = self.data[self.prop.feature_columns]
        y = self.data[list(self.prop.target_cols)]
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if val_size <= 0:
            self.X_train = X_train_val
            self.y_train = y_train_val
            return
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)
        
