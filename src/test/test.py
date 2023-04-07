import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt

from project.data_preparation import DataPreparation
from project.model_training import ModelTraining
from project.feature_importance import FeatureImportance

@pytest.fixture
def data():
    data = pd.read_csv("data.csv")
    return data

@pytest.fixture
def data_preparation(data):
    data_preparation = DataPreparation(data)
    return data_preparation

@pytest.fixture
def model_training(data_preparation):
    X_train, X_test, y_train, y_test = data_preparation.prepare_data()
    model_training = ModelTraining(X_train, X_test, y_train, y_test)
    return model_training

@pytest.fixture
def feature_importance(model_training, data_preparation):
    X_train, _, y_train, _ = data_preparation.prepare_data()
    lr = model_training.train_linear_regression()
    feature_importance = FeatureImportance(lr, X_train, y_train)
    return feature_importance

def test_data_preparation(data_preparation):
    assert isinstance(data_preparation.data, pd.DataFrame)
    assert data_preparation.data.shape == (1000, 25)
    assert data_preparation.data.isnull().sum().sum() == 0

def test_model_training(model_training):
    lr = model_training.train_linear_regression()
    assert isinstance(lr, LinearRegression)
    rf = model_training.train_random_forest(n_estimators=10, max_depth=5)
    assert isinstance(rf, RandomForestRegressor)
    results = model_training.compare_models([lr, rf])
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

def test_feature_importance(feature_importance):
    feature_importance.get_feature_importance()
    assert plt.gcf().axes[0].get_title() == "Permutation Importance"
    assert plt.gcf().axes[0].get_xlabel() == "Importance"
    assert isinstance(plt.gcf().axes[0].get_xticks()[0], np.float64)
