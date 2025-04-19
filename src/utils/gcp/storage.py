"""
Google Cloud Storage (GCS) utilities for the fine-tuning pipeline.

This module provides functions for interacting with Google Cloud Storage,
including uploading and downloading files, and managing datasets.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from google.cloud import storage

logger = logging.getLogger(__name__)

def upload_directory_to_gcs(
    local_dir: Union[str, Path],
    bucket_name: str,
    gcs_dir: str,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a directory to GCS.
    
    Args:
        local_dir: Local directory to upload
        bucket_name: GCS bucket name
        gcs_dir: GCS directory path
        project_id: GCP project ID (optional)
        
    Returns:
        Dictionary with upload statistics
    """
    # Initialize storage client
    client = storage.Client(project=project_id)
    
    # Get bucket
    bucket = client.bucket(bucket_name)
    
    # Upload files
    uploaded_files = []
    local_dir_path = Path(local_dir)
    
    logger.info(f"Uploading directory {local_dir_path} to gs://{bucket_name}/{gcs_dir}")
    
    for local_file in local_dir_path.glob("**/*"):
        if local_file.is_file():
            # Create GCS path
            rel_path = local_file.relative_to(local_dir_path)
            gcs_path = f"{gcs_dir}/{rel_path}"
            
            # Upload file
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_file))
            
            uploaded_files.append(gcs_path)
    
    logger.info(f"Uploaded {len(uploaded_files)} files to gs://{bucket_name}/{gcs_dir}")
    
    return {
        "bucket": bucket_name,
        "gcs_dir": gcs_dir,
        "uploaded_files": uploaded_files,
        "file_count": len(uploaded_files),
    }

def download_directory_from_gcs(
    bucket_name: str,
    gcs_dir: str,
    local_dir: Union[str, Path],
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Download a directory from GCS.
    
    Args:
        bucket_name: GCS bucket name
        gcs_dir: GCS directory path
        local_dir: Local directory to download to
        project_id: GCP project ID (optional)
        
    Returns:
        Dictionary with download statistics
    """
    # Initialize storage client
    client = storage.Client(project=project_id)
    
    # Get bucket
    bucket = client.bucket(bucket_name)
    
    # Create local directory if it doesn't exist
    local_dir_path = Path(local_dir)
    os.makedirs(local_dir_path, exist_ok=True)
    
    logger.info(f"Downloading gs://{bucket_name}/{gcs_dir} to {local_dir_path}")
    
    # List blobs
    blobs = list(bucket.list_blobs(prefix=gcs_dir))
    
    # Download files
    downloaded_files = []
    
    for blob in blobs:
        # Skip directories
        if blob.name.endswith("/"):
            continue
        
        # Create local path
        rel_path = blob.name[len(gcs_dir):].lstrip("/")
        local_path = local_dir_path / rel_path
        
        # Create directories if needed
        os.makedirs(local_path.parent, exist_ok=True)
        
        # Download file
        blob.download_to_filename(str(local_path))
        
        downloaded_files.append(str(local_path))
    
    logger.info(f"Downloaded {len(downloaded_files)} files from gs://{bucket_name}/{gcs_dir}")
    
    return {
        "bucket": bucket_name,
        "gcs_dir": gcs_dir,
        "downloaded_files": downloaded_files,
        "file_count": len(downloaded_files),
    }

def upload_file_to_gcs(
    local_file: Union[str, Path],
    bucket_name: str,
    gcs_path: str,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a single file to GCS.
    
    Args:
        local_file: Local file to upload
        bucket_name: GCS bucket name
        gcs_path: GCS path including filename
        project_id: GCP project ID (optional)
        
    Returns:
        Dictionary with upload information
    """
    # Initialize storage client
    client = storage.Client(project=project_id)
    
    # Get bucket
    bucket = client.bucket(bucket_name)
    
    # Upload file
    local_file_path = Path(local_file)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_file_path))
    
    logger.info(f"Uploaded {local_file_path} to gs://{bucket_name}/{gcs_path}")
    
    return {
        "bucket": bucket_name,
        "gcs_path": gcs_path,
        "public_url": blob.public_url,
    }

def check_gcs_path_exists(
    bucket_name: str,
    gcs_path: str,
    project_id: Optional[str] = None,
) -> bool:
    """Check if a path exists in GCS.
    
    Args:
        bucket_name: GCS bucket name
        gcs_path: GCS path to check
        project_id: GCP project ID (optional)
        
    Returns:
        True if path exists, False otherwise
    """
    # Initialize storage client
    client = storage.Client(project=project_id)
    
    # Get bucket
    bucket = client.bucket(bucket_name)
    
    # Check if blob exists
    blob = bucket.blob(gcs_path)
    return blob.exists()

def list_gcs_directory(
    bucket_name: str,
    gcs_dir: str,
    project_id: Optional[str] = None,
) -> List[str]:
    """List files in a GCS directory.
    
    Args:
        bucket_name: GCS bucket name
        gcs_dir: GCS directory path
        project_id: GCP project ID (optional)
        
    Returns:
        List of file paths in the directory
    """
    # Initialize storage client
    client = storage.Client(project=project_id)
    
    # Get bucket
    bucket = client.bucket(bucket_name)
    
    # List blobs
    blobs = list(bucket.list_blobs(prefix=gcs_dir))
    
    # Get file paths
    file_paths = [blob.name for blob in blobs if not blob.name.endswith("/")]
    
    return file_paths 