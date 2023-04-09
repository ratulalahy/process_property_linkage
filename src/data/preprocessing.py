import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from typing import Tuple
from dataclasses import dataclass
import smogn

@dataclass
class PreprocessingConfig:
    missing_values: str = "median"
    scaler: str = "standard"
    oversampler: str = "random"
    

@dataclass
class Preprocessing:
    data: pd.DataFrame 
    config: PreprocessingConfig
        
    def handle_missing_values(self) -> None:
        if self.config.missing_values == 'median':
            self.data.fillna(self.data.median(), inplace=True)
        else:
            raise ValueError(f"Unsupported scaler: {self.config.missing_values}")
                
    def handle_outliers(self) -> None:
        pass  # Implement outlier handling

    
    def normalize_numerical_vars(self):
        if self.config.scaler == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaler == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler: {self.config.scaler}")
        
        data_scaled = self.scaler.fit_transform(self.data.select_dtypes(include=['float64', 'int64']))
        return data_scaled
    
    def oversample_data(self, target_column: str) ->pd.DataFrame:
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
        if self.config.oversampler == 'random':
            self.oversampler = smogn.smoter
        else:
            raise ValueError(f"Unsupported scaler: {self.config.oversampler}")        
            
        data_oversampled = self.oversampler(data = self.data, y = target_column)
        return data_oversampled

