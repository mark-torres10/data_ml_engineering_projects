"""
Google Cloud Storage utilities for the Titanic ML project.

This module provides reusable functions for interacting with GCS buckets,
including uploading, downloading, listing, and managing files.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union
from google.cloud import storage
from src.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GCSManager:
    """
    Manager class for Google Cloud Storage operations.
    
    Provides convenient methods for common GCS operations like uploading,
    downloading, and listing files in the project bucket.
    
    Examples:
        >>> gcs = GCSManager()
        >>> gcs.upload_file("local_file.csv", "data/raw/file.csv")
        >>> files = gcs.list_files("data/raw/")
        >>> gcs.download_file("data/raw/file.csv", "local_download.csv")
    """
    
    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize GCS Manager.
        
        Args:
            bucket_name: GCS bucket name (uses config default if None)
        """
        self.bucket_name = bucket_name or config.GCS_BUCKET_NAME
        self.client = storage.Client(project=config.GCP_PROJECT_ID)
        
        try:
            self.bucket = self.client.bucket(self.bucket_name)
            # Verify bucket exists
            self.bucket.exists()
            logger.info(f"Connected to GCS bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to connect to bucket {self.bucket_name}: {e}")
            raise
    
    def upload_file(
        self,
        local_path: Union[str, Path],
        gcs_path: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to GCS.
        
        Args:
            local_path: Path to local file
            gcs_path: Destination path in GCS (without gs:// prefix)
            content_type: Optional content type (e.g., 'text/csv')
        
        Returns:
            GCS URI of uploaded file (gs://bucket/path)
        
        Example:
            >>> gcs.upload_file("data.csv", "data/raw/data.csv")
            'gs://bucket-name/data/raw/data.csv'
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        blob = self.bucket.blob(gcs_path)
        
        if content_type:
            blob.content_type = content_type
        
        blob.upload_from_filename(str(local_path))
        
        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info(f"✓ Uploaded {local_path} to {gcs_uri}")
        
        return gcs_uri
    
    def download_file(
        self,
        gcs_path: str,
        local_path: Union[str, Path]
    ) -> Path:
        """
        Download a file from GCS.
        
        Args:
            gcs_path: Source path in GCS (without gs:// prefix)
            local_path: Destination path for downloaded file
        
        Returns:
            Path to downloaded file
        
        Example:
            >>> gcs.download_file("data/raw/data.csv", "local_data.csv")
            Path('local_data.csv')
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        blob = self.bucket.blob(gcs_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"GCS file not found: gs://{self.bucket_name}/{gcs_path}")
        
        blob.download_to_filename(str(local_path))
        
        logger.info(f"✓ Downloaded gs://{self.bucket_name}/{gcs_path} to {local_path}")
        
        return local_path
    
    def list_files(
        self,
        prefix: str = "",
        delimiter: Optional[str] = None
    ) -> List[str]:
        """
        List files in GCS bucket with optional prefix.
        
        Args:
            prefix: Filter files by prefix (like a folder path)
            delimiter: If set, returns folder-like structure
        
        Returns:
            List of file paths (blob names)
        
        Example:
            >>> gcs.list_files("data/raw/")
            ['data/raw/train.csv', 'data/raw/test.csv']
        """
        blobs = self.client.list_blobs(
            self.bucket_name,
            prefix=prefix,
            delimiter=delimiter
        )
        
        file_list = [blob.name for blob in blobs]
        
        logger.info(f"Found {len(file_list)} files with prefix '{prefix}'")
        
        return file_list
    
    def file_exists(self, gcs_path: str) -> bool:
        """
        Check if a file exists in GCS.
        
        Args:
            gcs_path: Path in GCS (without gs:// prefix)
        
        Returns:
            True if file exists, False otherwise
        """
        blob = self.bucket.blob(gcs_path)
        return blob.exists()
    
    def delete_file(self, gcs_path: str) -> None:
        """
        Delete a file from GCS.
        
        Args:
            gcs_path: Path in GCS (without gs:// prefix)
        """
        blob = self.bucket.blob(gcs_path)
        
        if blob.exists():
            blob.delete()
            logger.info(f"✓ Deleted gs://{self.bucket_name}/{gcs_path}")
        else:
            logger.warning(f"File not found: gs://{self.bucket_name}/{gcs_path}")
    
    def get_file_info(self, gcs_path: str) -> dict:
        """
        Get metadata about a file in GCS.
        
        Args:
            gcs_path: Path in GCS (without gs:// prefix)
        
        Returns:
            Dictionary with file metadata
        """
        blob = self.bucket.blob(gcs_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"GCS file not found: gs://{self.bucket_name}/{gcs_path}")
        
        # Reload to get current metadata
        blob.reload()
        
        return {
            'name': blob.name,
            'size_bytes': blob.size,
            'size_mb': blob.size / (1024 * 1024) if blob.size else 0,
            'content_type': blob.content_type,
            'created': blob.time_created,
            'updated': blob.updated,
            'md5_hash': blob.md5_hash,
            'uri': f"gs://{self.bucket_name}/{blob.name}"
        }
    
    def upload_directory(
        self,
        local_dir: Union[str, Path],
        gcs_prefix: str,
        pattern: str = "**/*"
    ) -> List[str]:
        """
        Upload all files from a local directory to GCS.
        
        Args:
            local_dir: Local directory path
            gcs_prefix: Destination prefix in GCS
            pattern: Glob pattern for file matching (default: all files)
        
        Returns:
            List of uploaded GCS URIs
        """
        local_dir = Path(local_dir)
        
        if not local_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {local_dir}")
        
        uploaded_files = []
        
        for local_file in local_dir.glob(pattern):
            if local_file.is_file():
                # Create relative path for GCS
                relative_path = local_file.relative_to(local_dir)
                gcs_path = f"{gcs_prefix}/{relative_path}".replace("\\", "/")
                
                gcs_uri = self.upload_file(local_file, gcs_path)
                uploaded_files.append(gcs_uri)
        
        logger.info(f"✓ Uploaded {len(uploaded_files)} files from {local_dir}")
        
        return uploaded_files
    
    def get_signed_url(
        self,
        gcs_path: str,
        expiration_minutes: int = 60
    ) -> str:
        """
        Generate a signed URL for temporary file access.
        
        Args:
            gcs_path: Path in GCS (without gs:// prefix)
            expiration_minutes: URL expiration time in minutes
        
        Returns:
            Signed URL for file access
        """
        blob = self.bucket.blob(gcs_path)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_minutes * 60,
            method="GET"
        )
        
        logger.info(f"Generated signed URL for {gcs_path} (expires in {expiration_minutes}min)")
        
        return url


# Convenience functions for common operations
def upload_to_gcs(local_path: Union[str, Path], gcs_path: str) -> str:
    """
    Quick upload function.
    
    Args:
        local_path: Local file path
        gcs_path: GCS destination path
    
    Returns:
        GCS URI
    """
    gcs = GCSManager()
    return gcs.upload_file(local_path, gcs_path)


def download_from_gcs(gcs_path: str, local_path: Union[str, Path]) -> Path:
    """
    Quick download function.
    
    Args:
        gcs_path: GCS source path
        local_path: Local destination path
    
    Returns:
        Path to downloaded file
    """
    gcs = GCSManager()
    return gcs.download_file(gcs_path, local_path)


def list_gcs_files(prefix: str = "") -> List[str]:
    """
    Quick list function.
    
    Args:
        prefix: Filter by prefix
    
    Returns:
        List of file paths
    """
    gcs = GCSManager()
    return gcs.list_files(prefix)


if __name__ == "__main__":
    # Test GCS operations
    print("\nTesting GCS Manager...")
    print("="*70)
    
    gcs = GCSManager()
    
    # List files in data/raw/
    print("\nFiles in data/raw/:")
    files = gcs.list_files("data/raw/")
    for file in files:
        print(f"  - {file}")
    
    # Get info about train file
    if "data/raw/titanic_train.csv" in files:
        print("\nTrain file info:")
        info = gcs.get_file_info("data/raw/titanic_train.csv")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("GCS Manager test complete!")

