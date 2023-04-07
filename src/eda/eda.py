import pandas as pd
import seaborn as sns

class EDA:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        
    def univariate_analysis(self, col: str) -> None:
        sns.displot(self.df[col])
        
    def bivariate_analysis(self, col1: str, col2: str) -> None:
        sns.scatterplot(x=col1, y=col2, data=self.df)
