"""
Titanic Dataset Loader

This module provides a reusable data loader for the Titanic dataset from HuggingFace.
Supports loading, caching, and converting to pandas DataFrames for further processing.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TitanicDataLoader:
    """
    A reusable data loader for the Titanic dataset.
    
    This class handles:
    - Loading data from HuggingFace datasets
    - Converting to pandas DataFrames
    - Caching data locally
    - Providing dataset metadata and statistics
    
    Examples:
        >>> loader = TitanicDataLoader()
        >>> train_df, test_df = loader.load_data()
        >>> print(loader.get_dataset_info())
    """
    
    # Primary and fallback dataset identifiers
    PRIMARY_DATASET = "paulopontesm/titanic"
    FALLBACK_DATASETS = [
        "Tomate/Kaggle-Titanic",
        "vblagoje/titanic"
    ]
    
    # Direct CSV URLs from HuggingFace (fallback option)
    CSV_URLS = {
        "train": "https://huggingface.co/datasets/paulopontesm/titanic/raw/main/train.csv",
        "test": "https://huggingface.co/datasets/paulopontesm/titanic/raw/main/test.csv"
    }
    
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_reload: bool = False
    ):
        """
        Initialize the Titanic data loader.
        
        Args:
            dataset_name: HuggingFace dataset identifier (uses PRIMARY_DATASET if None)
            cache_dir: Directory to cache downloaded datasets (uses default HF cache if None)
            force_reload: If True, ignore cached data and reload from HuggingFace
        """
        self.dataset_name = dataset_name or self.PRIMARY_DATASET
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.force_reload = force_reload
        
        self._dataset: Optional[DatasetDict] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        
        logger.info(f"Initialized TitanicDataLoader with dataset: {self.dataset_name}")
    
    def load_data(
        self,
        return_huggingface: bool = False,
        use_csv_direct: bool = True
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], DatasetDict]:
        """
        Load the Titanic dataset from HuggingFace.
        
        Args:
            return_huggingface: If True, return HuggingFace Dataset objects.
                              If False, return pandas DataFrames (default)
            use_csv_direct: If True, load CSV files directly (faster, more reliable).
                           If False, use HuggingFace datasets library
        
        Returns:
            If return_huggingface=False: Tuple of (train_df, test_df) as pandas DataFrames
            If return_huggingface=True: HuggingFace DatasetDict with train/test splits
        
        Raises:
            RuntimeError: If dataset cannot be loaded from any source
        """
        logger.info(f"Loading Titanic dataset...")
        
        # Try direct CSV loading first (simpler and more reliable for this dataset)
        if use_csv_direct:
            try:
                self._train_df, self._test_df = self._load_from_csv_urls()
                logger.info(f"✓ Data loaded successfully from CSV")
                logger.info(f"  Train shape: {self._train_df.shape}")
                if self._test_df is not None:
                    logger.info(f"  Test shape: {self._test_df.shape}")
                return self._train_df, self._test_df
            except Exception as e:
                logger.warning(f"Failed to load from CSV URLs: {e}")
                logger.info("Falling back to HuggingFace datasets library...")
        
        # Try HuggingFace datasets library (if CSV direct failed or not requested)
        try:
            self._dataset = self._load_from_huggingface(self.dataset_name)
            logger.info(f"✓ Successfully loaded dataset from {self.dataset_name}")
        except Exception as e:
            logger.warning(f"Failed to load from {self.dataset_name}: {e}")
            
            # Try fallback datasets
            for fallback in self.FALLBACK_DATASETS:
                try:
                    logger.info(f"Trying fallback dataset: {fallback}")
                    self._dataset = self._load_from_huggingface(fallback)
                    self.dataset_name = fallback
                    logger.info(f"✓ Successfully loaded dataset from {fallback}")
                    break
                except Exception as fallback_error:
                    logger.warning(f"Failed to load from {fallback}: {fallback_error}")
                    continue
            else:
                raise RuntimeError(
                    "Failed to load Titanic dataset from all sources. "
                    "Please check your internet connection or try loading from local files."
                )
        
        if return_huggingface:
            return self._dataset
        
        # Convert to pandas DataFrames
        self._train_df = self._to_dataframe(self._dataset['train'])
        
        # Handle test split (might not have labels)
        if 'test' in self._dataset:
            self._test_df = self._to_dataframe(self._dataset['test'])
        else:
            # Some datasets don't have a separate test split
            logger.warning("No test split found. You may need to create one from train data.")
            self._test_df = None
        
        logger.info(f"✓ Data loaded successfully")
        logger.info(f"  Train shape: {self._train_df.shape}")
        if self._test_df is not None:
            logger.info(f"  Test shape: {self._test_df.shape}")
        
        return self._train_df, self._test_df
    
    def _load_from_csv_urls(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load data directly from CSV URLs.
        
        This is often more reliable than using the HuggingFace datasets library
        for simple CSV files, especially when train/test have different schemas.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Loading from CSV URLs...")
        
        # Load training data
        train_df = pd.read_csv(self.CSV_URLS["train"])
        logger.info(f"✓ Loaded train data: {train_df.shape}")
        
        # Load test data
        try:
            test_df = pd.read_csv(self.CSV_URLS["test"])
            logger.info(f"✓ Loaded test data: {test_df.shape}")
        except Exception as e:
            logger.warning(f"Could not load test data: {e}")
            test_df = None
        
        return train_df, test_df
    
    def _load_from_huggingface(self, dataset_name: str) -> DatasetDict:
        """
        Load dataset from HuggingFace.
        
        Args:
            dataset_name: HuggingFace dataset identifier
        
        Returns:
            DatasetDict containing train/test splits
        """
        # Load train and test separately to handle missing target column in test
        try:
            # Try loading with default behavior first
            dataset = load_dataset(
                dataset_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                download_mode="force_redownload" if self.force_reload else None
            )
            return dataset
        except Exception as e:
            # If that fails, try loading train and test separately
            logger.info("Attempting to load train/test splits separately...")
            train_dataset = load_dataset(
                dataset_name,
                split="train",
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                download_mode="force_redownload" if self.force_reload else None
            )
            
            try:
                test_dataset = load_dataset(
                    dataset_name,
                    split="test",
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    download_mode="force_redownload" if self.force_reload else None
                )
                # Create a DatasetDict manually
                from datasets import DatasetDict as DD
                return DD({"train": train_dataset, "test": test_dataset})
            except:
                # If test split doesn't exist, just return train
                from datasets import DatasetDict as DD
                return DD({"train": train_dataset})
    
    def _to_dataframe(self, dataset: Dataset) -> pd.DataFrame:
        """
        Convert HuggingFace Dataset to pandas DataFrame.
        
        Args:
            dataset: HuggingFace Dataset object
        
        Returns:
            pandas DataFrame
        """
        return dataset.to_pandas()
    
    def save_to_csv(
        self,
        train_path: Union[str, Path],
        test_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Save loaded DataFrames to CSV files.
        
        Args:
            train_path: Path to save training data
            test_path: Path to save test data (optional if no test split)
        
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self._train_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        train_path = Path(train_path)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._train_df.to_csv(train_path, index=False)
        logger.info(f"✓ Saved training data to {train_path}")
        
        if self._test_df is not None and test_path:
            test_path = Path(test_path)
            test_path.parent.mkdir(parents=True, exist_ok=True)
            self._test_df.to_csv(test_path, index=False)
            logger.info(f"✓ Saved test data to {test_path}")
    
    def load_from_csv(
        self,
        train_path: Union[str, Path],
        test_path: Optional[Union[str, Path]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load data from local CSV files instead of HuggingFace.
        
        Useful for loading preprocessed data or working offline.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file (optional)
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Loading data from local CSV files...")
        
        train_path = Path(train_path)
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        
        self._train_df = pd.read_csv(train_path)
        logger.info(f"✓ Loaded training data: {self._train_df.shape}")
        
        if test_path:
            test_path = Path(test_path)
            if test_path.exists():
                self._test_df = pd.read_csv(test_path)
                logger.info(f"✓ Loaded test data: {self._test_df.shape}")
            else:
                logger.warning(f"Test file not found: {test_path}")
                self._test_df = None
        else:
            self._test_df = None
        
        return self._train_df, self._test_df
    
    def get_dataset_info(self) -> Dict[str, any]:
        """
        Get metadata and statistics about the loaded dataset.
        
        Returns:
            Dictionary containing dataset information
        
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self._train_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            'dataset_name': self.dataset_name,
            'train_samples': len(self._train_df),
            'test_samples': len(self._test_df) if self._test_df is not None else 0,
            'features': list(self._train_df.columns),
            'num_features': len(self._train_df.columns),
            'train_shape': self._train_df.shape,
            'test_shape': self._test_df.shape if self._test_df is not None else None,
            'missing_values': self._train_df.isnull().sum().to_dict(),
            'dtypes': self._train_df.dtypes.astype(str).to_dict()
        }
        
        # Add target variable statistics if present
        if 'Survived' in self._train_df.columns:
            info['survival_rate'] = self._train_df['Survived'].mean()
            info['class_distribution'] = self._train_df['Survived'].value_counts().to_dict()
        
        return info
    
    def display_info(self) -> None:
        """
        Display formatted dataset information.
        """
        info = self.get_dataset_info()
        
        print("\n" + "="*70)
        print("Titanic Dataset Information")
        print("="*70)
        print(f"Dataset Source:     {info['dataset_name']}")
        print(f"Training Samples:   {info['train_samples']}")
        print(f"Test Samples:       {info['test_samples']}")
        print(f"Number of Features: {info['num_features']}")
        print(f"\nFeatures:")
        for feature in info['features']:
            print(f"  - {feature}")
        
        if 'survival_rate' in info:
            print(f"\nTarget Variable (Survived):")
            print(f"  Survival Rate:    {info['survival_rate']:.2%}")
            print(f"  Class Distribution:")
            for cls, count in info['class_distribution'].items():
                print(f"    {cls}: {count} samples ({count/info['train_samples']:.2%})")
        
        print(f"\nMissing Values:")
        missing = info['missing_values']
        has_missing = False
        for feature, count in missing.items():
            if count > 0:
                pct = count / info['train_samples'] * 100
                print(f"  {feature}: {count} ({pct:.1f}%)")
                has_missing = True
        if not has_missing:
            print("  No missing values")
        
        print("="*70 + "\n")
    
    @property
    def train_df(self) -> Optional[pd.DataFrame]:
        """Get the loaded training DataFrame."""
        return self._train_df
    
    @property
    def test_df(self) -> Optional[pd.DataFrame]:
        """Get the loaded test DataFrame."""
        return self._test_df
    
    def get_feature_names(self) -> list:
        """
        Get list of feature names (excluding target if present).
        
        Returns:
            List of feature column names
        """
        if self._train_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Exclude common ID and target columns
        exclude_cols = {'PassengerId', 'Survived'}
        return [col for col in self._train_df.columns if col not in exclude_cols]
    
    def get_target_name(self) -> str:
        """
        Get the name of the target variable.
        
        Returns:
            Name of target column (default: 'Survived')
        """
        return 'Survived'


def quick_load_titanic(
    cache_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to quickly load Titanic data.
    
    Args:
        cache_dir: Optional cache directory for datasets
    
    Returns:
        Tuple of (train_df, test_df)
    
    Example:
        >>> train_df, test_df = quick_load_titanic()
    """
    loader = TitanicDataLoader(cache_dir=cache_dir)
    return loader.load_data()


if __name__ == "__main__":
    # Demo usage
    print("Loading Titanic dataset...")
    
    loader = TitanicDataLoader()
    train_df, test_df = loader.load_data()
    
    # Display information
    loader.display_info()
    
    # Show first few rows
    print("First 5 rows of training data:")
    print(train_df.head())

