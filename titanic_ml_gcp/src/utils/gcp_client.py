"""
GCP Client utilities for authenticating and connecting to GCP services.

This module provides helper functions to create authenticated clients
for various GCP services using application default credentials.
"""

from google.cloud import storage, aiplatform
from google.oauth2 import service_account
import os
from pathlib import Path


def get_storage_client():
    """
    Create and return an authenticated Google Cloud Storage client.
    
    Uses application default credentials (from gcloud auth application-default login)
    or service account key if GOOGLE_APPLICATION_CREDENTIALS is set.
    
    Returns:
        storage.Client: Authenticated GCS client
    """
    return storage.Client()


def get_vertex_ai_client(project_id: str, region: str):
    """
    Initialize Vertex AI client.
    
    Args:
        project_id: GCP project ID
        region: GCP region (e.g., 'us-central1')
    
    Returns:
        None (initializes aiplatform globally)
    """
    aiplatform.init(
        project=project_id,
        location=region
    )


def verify_credentials():
    """
    Verify that GCP credentials are properly configured.
    
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    try:
        # Try to create a storage client and list buckets
        client = storage.Client()
        # This will fail if credentials are invalid
        list(client.list_buckets(max_results=1))
        print("✓ GCP credentials are valid")
        return True
    except Exception as e:
        print(f"✗ GCP credentials verification failed: {e}")
        print("\nTo fix this, run:")
        print("  gcloud auth application-default login")
        return False


if __name__ == "__main__":
    # Test credential verification
    print("Testing GCP credentials...")
    verify_credentials()

