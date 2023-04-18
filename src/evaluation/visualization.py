from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

class Visualize:
    def __init__(self):
        pass
    
    def plot_feature_distribution(self, data: pd.DataFrame, features: List[str]):
        """Plots the distribution of the given features in the data.
        
        Args:
            data (pd.DataFrame): The input data.
            features (List[str]): A list of feature names to plot.
        """
        for feature in features:
            sns.histplot(data=data, x=feature)
            plt.show()
            
    def plot_correlation_heatmap(self, data: pd.DataFrame):
        """Plots the correlation matrix as a heatmap for the given data.
        
        Args:
            data (pd.DataFrame): The input data.
        """
        corr = data.corr()
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', annot_kws={'size': 10})
        plt.show()
        
    def plot_learning_curve(self, train_scores: List[float], test_scores: List[float], title: str):
        """Plots the learning curve for the given training and testing scores.
        
        Args:
            train_scores (List[float]): A list of training scores at each iteration.
            test_scores (List[float]): A list of testing scores at each iteration.
            title (str): The title of the plot.
        """
        plt.plot(train_scores, label='Training score')
        plt.plot(test_scores, label='Testing score')
        plt.xlabel('Iterations')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.show()
