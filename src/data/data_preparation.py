import pandas as pd
from typing import Tuple

class DataPreparation:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        
    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filename)
        return df
    
    def split_data(self, df: pd.DataFrame, target_cols: Tuple[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = df.drop(columns=target_cols)
        y = df[target_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
