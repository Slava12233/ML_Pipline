"""
Data export module.

This module provides functions for exporting data to Google Cloud Storage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def export_to_gcs(
    local_path: Union[str, Path],
    gcs_path: str,
    project_id: str,
    bucket_name: str,
) -> str:
    """
    Export a local file to Google Cloud Storage.

    Args:
        local_path: Path to the local file.
        gcs_path: Path within the GCS bucket.
        project_id: Google Cloud project ID.
        bucket_name: GCS bucket name.

    Returns:
        str: GCS URI of the uploaded file.
    """
    logger.info(f"Exporting {local_path} to gs://{bucket_name}/{gcs_path}")
    
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    
    try:
        from google.cloud import storage
        
        # Initialize GCS client
        client = storage.Client(project=project_id)
        
        # Get bucket
        bucket = client.bucket(bucket_name)
        
        # Create blob
        blob = bucket.blob(gcs_path)
        
        # Upload file
        blob.upload_from_filename(str(local_path))
        
        # Return GCS URI
        gcs_uri = f"gs://{bucket_name}/{gcs_path}"
        logger.info(f"Exported {local_path} to {gcs_uri}")
        
        return gcs_uri
    
    except ImportError:
        logger.error("google-cloud-storage package not installed.")
        raise
    except Exception as e:
        logger.error(f"Error exporting {local_path} to GCS: {str(e)}")
        raise


def export_directory_to_gcs(
    local_dir: Union[str, Path],
    gcs_dir: str,
    project_id: str,
    bucket_name: str,
    file_pattern: str = "*",
    recursive: bool = True,
) -> List[str]:
    """
    Export all files in a directory to Google Cloud Storage.

    Args:
        local_dir: Path to the local directory.
        gcs_dir: Directory path within the GCS bucket.
        project_id: Google Cloud project ID.
        bucket_name: GCS bucket name.
        file_pattern: Pattern to match files (e.g., "*.jsonl").
        recursive: Whether to recursively export subdirectories.

    Returns:
        List[str]: List of GCS URIs of the uploaded files.
    """
    logger.info(f"Exporting directory {local_dir} to gs://{bucket_name}/{gcs_dir}")
    
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")
    
    # Normalize GCS directory path
    gcs_dir = gcs_dir.rstrip("/")
    if gcs_dir:
        gcs_dir += "/"
    
    # Find files to export
    if recursive:
        files = list(local_dir.glob(f"**/{file_pattern}"))
    else:
        files = list(local_dir.glob(file_pattern))
    
    logger.info(f"Found {len(files)} files to export")
    
    # Export files
    gcs_uris = []
    for file_path in files:
        # Calculate relative path
        rel_path = file_path.relative_to(local_dir)
        
        # Calculate GCS path
        file_gcs_path = f"{gcs_dir}{rel_path}"
        
        # Export file
        gcs_uri = export_to_gcs(file_path, file_gcs_path, project_id, bucket_name)
        gcs_uris.append(gcs_uri)
    
    return gcs_uris


def export_training_data_to_gcs(
    data_dir: Union[str, Path],
    project_id: str,
    bucket_name: str,
    gcs_dir: str = "training_data",
) -> Dict[str, List[str]]:
    """
    Export training data to Google Cloud Storage.

    Args:
        data_dir: Path to the local training data directory.
        project_id: Google Cloud project ID.
        bucket_name: GCS bucket name.
        gcs_dir: Directory path within the GCS bucket.

    Returns:
        Dict[str, List[str]]: Dictionary mapping split names to lists of GCS URIs.
    """
    logger.info(f"Exporting training data from {data_dir} to gs://{bucket_name}/{gcs_dir}")
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")
    
    # Check for expected files
    expected_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    missing_files = [f for f in expected_files if not (data_dir / f).exists()]
    
    if missing_files:
        logger.warning(f"Missing expected files: {missing_files}")
    
    # Export files
    result = {}
    for split in ["train", "val", "test"]:
        file_path = data_dir / f"{split}.jsonl"
        if file_path.exists():
            gcs_path = f"{gcs_dir}/{split}.jsonl"
            gcs_uri = export_to_gcs(file_path, gcs_path, project_id, bucket_name)
            result[split] = [gcs_uri]
    
    return result


def export_metadata_to_gcs(
    metadata_dir: Union[str, Path],
    project_id: str,
    bucket_name: str,
    gcs_dir: str = "metadata",
) -> List[str]:
    """
    Export metadata to Google Cloud Storage.

    Args:
        metadata_dir: Path to the local metadata directory.
        project_id: Google Cloud project ID.
        bucket_name: GCS bucket name.
        gcs_dir: Directory path within the GCS bucket.

    Returns:
        List[str]: List of GCS URIs of the uploaded files.
    """
    logger.info(f"Exporting metadata from {metadata_dir} to gs://{bucket_name}/{gcs_dir}")
    
    return export_directory_to_gcs(
        local_dir=metadata_dir,
        gcs_dir=gcs_dir,
        project_id=project_id,
        bucket_name=bucket_name,
        file_pattern="*.json",
        recursive=False,
    )


def create_manifest_file(
    gcs_uris: Dict[str, List[str]],
    output_path: Union[str, Path],
) -> str:
    """
    Create a manifest file with GCS URIs.

    Args:
        gcs_uris: Dictionary mapping split names to lists of GCS URIs.
        output_path: Path to save the manifest file.

    Returns:
        str: Path to the manifest file.
    """
    logger.info(f"Creating manifest file at {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(gcs_uris, f, indent=2)
    
    logger.info(f"Manifest file created at {output_path}")
    
    return str(output_path)


def export_all_to_gcs(
    base_dir: Union[str, Path],
    project_id: str,
    bucket_name: str,
    gcs_base_dir: str = "",
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    """
    Export all data to Google Cloud Storage.

    Args:
        base_dir: Base directory containing all data.
        project_id: Google Cloud project ID.
        bucket_name: GCS bucket name.
        gcs_base_dir: Base directory path within the GCS bucket.

    Returns:
        Dict[str, Union[List[str], Dict[str, List[str]]]]: Dictionary with GCS URIs.
    """
    logger.info(f"Exporting all data from {base_dir} to gs://{bucket_name}/{gcs_base_dir}")
    
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Normalize GCS base directory path
    gcs_base_dir = gcs_base_dir.rstrip("/")
    if gcs_base_dir:
        gcs_base_dir += "/"
    
    result = {}
    
    # Export extracted text
    text_dir = base_dir / "extracted_text"
    if text_dir.exists():
        text_gcs_dir = f"{gcs_base_dir}extracted_text"
        result["text_files"] = export_directory_to_gcs(
            local_dir=text_dir,
            gcs_dir=text_gcs_dir,
            project_id=project_id,
            bucket_name=bucket_name,
            file_pattern="*.txt",
            recursive=True,
        )
    
    # Export metadata
    metadata_dir = base_dir / "extracted_text" / "metadata"
    if metadata_dir.exists():
        metadata_gcs_dir = f"{gcs_base_dir}metadata"
        result["metadata_files"] = export_metadata_to_gcs(
            metadata_dir=metadata_dir,
            project_id=project_id,
            bucket_name=bucket_name,
            gcs_dir=metadata_gcs_dir,
        )
    
    # Export training data
    training_data_dir = base_dir / "training_data"
    if training_data_dir.exists():
        training_data_gcs_dir = f"{gcs_base_dir}training_data"
        result["training_data"] = export_training_data_to_gcs(
            data_dir=training_data_dir,
            project_id=project_id,
            bucket_name=bucket_name,
            gcs_dir=training_data_gcs_dir,
        )
    
    # Create manifest file
    manifest_path = base_dir / "gcs_manifest.json"
    create_manifest_file(result, manifest_path)
    
    # Export manifest file
    manifest_gcs_path = f"{gcs_base_dir}gcs_manifest.json"
    result["manifest"] = export_to_gcs(
        local_path=manifest_path,
        gcs_path=manifest_gcs_path,
        project_id=project_id,
        bucket_name=bucket_name,
    )
    
    return result
