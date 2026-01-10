import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_train_data():
    """Create a sample DataFrame mimicking the Titanic training set."""
    data = {
        'PassengerId': [1, 2, 3, 4, 5],
        'Survived': [0, 1, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': [
            'Braund, Mr. Owen Harris',
            'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
            'Heikkinen, Miss. Laina',
            'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
            'Allen, Mr. William Henry'
        ],
        'Sex': ['male', 'female', 'female', 'female', 'male'],
        'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
        'SibSp': [1, 1, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
        'Fare': [7.2500, 71.2833, 7.9250, 53.1000, 8.0500],
        'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan],
        'Embarked': ['S', 'C', 'S', 'S', 'S']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_test_data():
    """Create a sample DataFrame mimicking the Titanic test set (with missing values)."""
    data = {
        'PassengerId': [6, 7],
        'Pclass': [3, 1],
        'Name': ['Moran, Mr. James', 'Chaffee, Mrs. Herbert Fuller (Carrie Constance Toogood)'],
        'Sex': ['male', 'female'],
        'Age': [np.nan, 47.0], # Missing Age
        'SibSp': [0, 1],
        'Parch': [0, 0],
        'Ticket': ['330877', 'W.E.P. 5734'],
        'Fare': [8.4583, 61.1750],
        'Cabin': [np.nan, 'E31'],
        'Embarked': ['Q', 'S']
    }
    return pd.DataFrame(data)

