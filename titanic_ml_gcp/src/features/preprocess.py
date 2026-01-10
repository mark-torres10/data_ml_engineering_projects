import logging
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from typing import Tuple, Optional, List, Dict

# Configure logging
logger = logging.getLogger(__name__)

class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """
    A comprehensive preprocessor for the Titanic dataset that handles:
    1. Missing value imputation
    2. Feature engineering (Title, Family Size, etc.)
    3. Categorical encoding
    4. Feature scaling
    """
    
    def __init__(self):
        self.age_medians: Dict[str, float] = {}
        self.fare_median_by_pclass: Dict[int, float] = {}
        self.embarked_mode: str = 'S'
        self.scaler = StandardScaler()
        self.fitted = False
        
        # Feature lists (will be populated during fit)
        self.numerical_features = ['Age', 'Fare', 'Family_Size', 'Pclass']
        self.binary_features = ['Sex_male', 'Is_Alone', 'Has_Cabin']
        self.one_hot_features = [] # Will depend on encoded columns
        
        # Artifact paths
        self.artifacts_dir = os.path.join("data", "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def _extract_title(self, name: str) -> str:
        """Extract title from passenger name."""
        title = name.split(',')[1].split('.')[0].strip()
        return title

    def _group_titles(self, title: str) -> str:
        """Group rare titles into broader categories."""
        if title in ['Mr', 'Mrs', 'Miss', 'Master']:
            return title
        elif title in ['Sir', 'Don', 'Dona', 'Jonkheer', 'Countess', 'Lady']:
            return 'Royalty'
        elif title in ['Dr', 'Rev', 'Col', 'Major', 'Capt']:
            return 'Officer'
        else:
            return 'Mrs' # Mlle, Ms, Mme usually map to Mrs/Miss

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn statistical parameters for imputation and scaling from training data.
        """
        logger.info("Fitting preprocessor...")
        df = X.copy()
        
        # 1. Learn imputations
        
        # Age: Median by Pclass and Sex
        self.age_medians = df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        
        # Fare: Median by Pclass
        self.fare_median_by_pclass = df.groupby('Pclass')['Fare'].median().to_dict()
        
        # Embarked: Mode
        self.embarked_mode = df['Embarked'].mode()[0]
        
        # Note: Scaler fitting happens in transform on the fully engineered data 
        # or we need to do a preliminary transform pass. 
        # To adhere to proper fit/transform pattern, we should ideally enable 
        # the scaler to be fit on the transformed data. 
        # For this custom class, we'll fit the scaler in a separate method or 
        # during fit_transform. Let's do it in fit_transform or a specific fit_scaler method 
        # to ensure we have the final columns. 
        # A common pattern for custom complex transformers is to chain them.
        # For simplicity in this specific project structure, we will fit scaler 
        # on the engineered features within fit_transform.
        
        self.fitted = True
        logger.info("Preprocessor fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to the data.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")
            
        logger.info("Transforming data...")
        df = X.copy()
        
        # --- 1. Missing Value Imputation ---
        
        # Embarked
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)
        
        # Fare
        df['Fare'] = df.apply(
            lambda row: self.fare_median_by_pclass.get(row['Pclass'], df['Fare'].median()) 
            if pd.isnull(row['Fare']) else row['Fare'], axis=1
        )
        
        # Age
        df['Age'] = df.apply(
            lambda row: self.age_medians.get((row['Pclass'], row['Sex']), df['Age'].median())
            if pd.isnull(row['Age']) else row['Age'], axis=1
        )
        
        # Create Age_Missing indicator (optional, but good practice)
        # df['Age_Missing'] = X['Age'].isnull().astype(int)
        
        # --- 2. Feature Engineering ---
        
        # Family Size
        df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
        
        # Is_Alone
        df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
        
        # Has_Cabin
        df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
        
        # Title extraction
        df['Title'] = df['Name'].apply(self._extract_title)
        df['Title'] = df['Title'].apply(self._group_titles)
        
        # --- 3. Encoding ---
        
        # Sex (Binary)
        df['Sex_male'] = df['Sex'].map({'male': 1, 'female': 0})
        
        # One-hot encoding for Categorical variables (Embarked, Title)
        # We use pd.get_dummies for readability in this custom pipeline, 
        # ensuring columns match is critical for prod. 
        # Ideally, we'd use OneHotEncoder from sklearn.
        # For this implementation, let's manually handle it or use pandas with specific columns enforced.
        
        # Dummy encoding
        df = pd.get_dummies(df, columns=['Embarked', 'Title'], prefix=['Embarked', 'Title'], drop_first=True)
        
        # Ensure specific columns exist (handling missing categories in test set)
        expected_cols = [
            'Embarked_Q', 'Embarked_S', 
            'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty'
        ]
        # Note: Drop first means 'Embarked_C' and 'Title_Master' are reference categories (implicit)
        
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
                
        # --- 4. Cleanup & Selection ---
        
        # Drop unused columns
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # Define final feature order
        self.final_features = [
            'Pclass', 'Age', 'Fare', 'Family_Size', 'Is_Alone', 'Has_Cabin', 'Sex_male',
            'Embarked_Q', 'Embarked_S',
            'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty'
        ]
        
        # Ensure all features exist and are in correct order
        for col in self.final_features:
            if col not in df.columns:
                df[col] = 0
                
        df = df[self.final_features]
        
        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit to data, then transform it. Also fits the internal scaler.
        """
        self.fit(X, y)
        df_transformed = self.transform(X)
        
        # Fit scaler on the transformed data
        logger.info("Fitting scaler on transformed features...")
        self.scaler.fit(df_transformed)
        
        # Apply scaling
        scaled_data = self.scaler.transform(df_transformed)
        
        # Return as DataFrame
        return pd.DataFrame(scaled_data, columns=df_transformed.columns, index=df_transformed.index)

    def transform_with_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation and then use the fitted scaler.
        """
        df_transformed = self.transform(X)
        scaled_data = self.scaler.transform(df_transformed)
        return pd.DataFrame(scaled_data, columns=df_transformed.columns, index=df_transformed.index)

    def save(self, filepath: str = None):
        """Save the preprocessor object."""
        if filepath is None:
            filepath = os.path.join(self.artifacts_dir, "preprocessor.joblib")
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
        
    @staticmethod
    def load(filepath: str) -> 'TitanicPreprocessor':
        """Load a saved preprocessor object."""
        logger.info(f"Loading preprocessor from {filepath}")
        return joblib.load(filepath)

if __name__ == "__main__":
    # Simple test
    from src.data.loader import TitanicDataLoader
    
    loader = TitanicDataLoader()
    train_df, test_df = loader.load_data()
    
    preprocessor = TitanicPreprocessor()
    X_train_processed = preprocessor.fit_transform(train_df)
    print("Processed Training Data Shape:", X_train_processed.shape)
    print("Features:", X_train_processed.columns.tolist())
    
    X_test_processed = preprocessor.transform_with_scaling(test_df)
    print("Processed Test Data Shape:", X_test_processed.shape)
    
    preprocessor.save()

