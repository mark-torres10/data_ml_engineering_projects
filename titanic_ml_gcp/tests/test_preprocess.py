import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from src.features.preprocess import TitanicPreprocessor

class TestTitanicPreprocessor:
    
    def test_initialization(self):
        preprocessor = TitanicPreprocessor()
        assert not preprocessor.fitted
        assert isinstance(preprocessor.numerical_features, list)

    def test_fit(self, sample_train_data):
        preprocessor = TitanicPreprocessor()
        preprocessor.fit(sample_train_data)
        
        assert preprocessor.fitted
        assert preprocessor.embarked_mode == 'S'
        # Check age medians populated
        assert (3, 'male') in preprocessor.age_medians
        assert (1, 'female') in preprocessor.age_medians

    def test_transform_structure(self, sample_train_data):
        preprocessor = TitanicPreprocessor()
        preprocessor.fit(sample_train_data)
        transformed = preprocessor.transform(sample_train_data)
        
        # Check columns exist
        expected_cols = [
            'Pclass', 'Age', 'Fare', 'Family_Size', 'Is_Alone', 'Has_Cabin', 'Sex_male',
            'Embarked_Q', 'Embarked_S',
            'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty'
        ]
        for col in expected_cols:
            assert col in transformed.columns
            
        # Check shape
        assert transformed.shape[0] == sample_train_data.shape[0]
        # Count is dynamic based on one-hot, but our preprocessor enforces fixed set
        assert transformed.shape[1] == len(expected_cols)

    def test_imputation(self, sample_train_data, sample_test_data):
        preprocessor = TitanicPreprocessor()
        preprocessor.fit(sample_train_data)
        
        # Test data has missing Age
        assert sample_test_data['Age'].isnull().any()
        
        transformed = preprocessor.transform(sample_test_data)
        
        # Should be imputed
        assert not transformed['Age'].isnull().any()
        
    def test_feature_engineering(self, sample_train_data):
        preprocessor = TitanicPreprocessor()
        preprocessor.fit(sample_train_data)
        transformed = preprocessor.transform(sample_train_data)
        
        # Family Size = SibSp + Parch + 1
        # 1st row: SibSp=1, Parch=0 -> 2
        assert transformed.iloc[0]['Family_Size'] == 2
        
        # Is_Alone
        # 1st row: Family_Size=2 -> 0
        # 3rd row (index 2): SibSp=0, Parch=0 -> 1
        assert transformed.iloc[0]['Is_Alone'] == 0
        assert transformed.iloc[2]['Is_Alone'] == 1
        
        # Has_Cabin
        # 1st row: Cabin=NaN -> 0
        # 2nd row: Cabin=C85 -> 1
        assert transformed.iloc[0]['Has_Cabin'] == 0
        assert transformed.iloc[1]['Has_Cabin'] == 1

    def test_encoding_sex(self, sample_train_data):
        preprocessor = TitanicPreprocessor()
        preprocessor.fit(sample_train_data)
        transformed = preprocessor.transform(sample_train_data)
        
        # Male -> 1, Female -> 0
        assert transformed.iloc[0]['Sex_male'] == 1 # Mr. Owen Harris
        assert transformed.iloc[1]['Sex_male'] == 0 # Mrs. John Bradley

    def test_fit_transform_scaling(self, sample_train_data):
        preprocessor = TitanicPreprocessor()
        scaled = preprocessor.fit_transform(sample_train_data)
        
        # Scaled data should have mean approx 0 and std approx 1
        # Check Age (numerical)
        assert abs(scaled['Age'].mean()) < 1e-6
        # Sklearn StandardScaler uses ddof=0, Pandas std() uses ddof=1
        # We should check std(ddof=0)
        assert abs(scaled['Age'].std(ddof=0) - 1.0) < 1e-6

    def test_save_load(self, sample_train_data, tmp_path):
        preprocessor = TitanicPreprocessor()
        preprocessor.fit(sample_train_data)
        
        save_path = tmp_path / "preprocessor.joblib"
        preprocessor.save(str(save_path))
        
        assert os.path.exists(save_path)
        
        loaded_preprocessor = TitanicPreprocessor.load(str(save_path))
        assert loaded_preprocessor.fitted
        assert loaded_preprocessor.embarked_mode == preprocessor.embarked_mode

