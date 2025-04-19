#!/usr/bin/env python3
"""
Test GCS Integration.

This script tests the Google Cloud Storage integration by uploading
and downloading files from a GCS bucket.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utilities
from src.utils.config import get_config
from src.utils.gcp.storage import (
    upload_directory_to_gcs, 
    download_directory_from_gcs,
    upload_file_to_gcs,
    check_gcs_path_exists,
    list_gcs_directory
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main():
    """Run the GCS integration test."""
    parser = argparse.ArgumentParser(description="Test GCS integration")
    parser.add_argument("--config", type=str, default="config/vertex_config.yaml", help="Path to configuration file")
    parser.add_argument("--env", type=str, default="vertex", help="Environment to use")
    parser.add_argument("--pdf-dir", type=str, default="data/pdfs", help="Directory containing PDF files")
    parser.add_argument("--download-dir", type=str, default="data/test_download", help="Directory to download files to")
    parser.add_argument("--test-dir", type=str, default="test", help="GCS directory for testing")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config, env=args.env)
    
    # Get GCP settings
    project_id = config.gcp.project_id
    bucket_name = config.gcp.storage.bucket_name
    
    logger.info(f"Using GCP project: {project_id}")
    logger.info(f"Using GCS bucket: {bucket_name}")
    
    # Test directory paths
    gcs_test_dir = f"{args.test_dir}/pdfs"
    local_pdf_dir = args.pdf_dir
    local_download_dir = args.download_dir
    
    # Test uploading directory
    logger.info(f"Testing directory upload from {local_pdf_dir} to gs://{bucket_name}/{gcs_test_dir}")
    
    upload_result = upload_directory_to_gcs(
        local_dir=local_pdf_dir,
        bucket_name=bucket_name,
        gcs_dir=gcs_test_dir,
        project_id=project_id,
    )
    
    logger.info(f"Successfully uploaded {upload_result['file_count']} files")
    
    # Test listing directory
    logger.info(f"Testing directory listing for gs://{bucket_name}/{gcs_test_dir}")
    
    file_list = list_gcs_directory(
        bucket_name=bucket_name,
        gcs_dir=gcs_test_dir,
        project_id=project_id,
    )
    
    logger.info(f"Found {len(file_list)} files in directory")
    for file_path in file_list:
        logger.info(f"  - {file_path}")
    
    # Test downloading directory
    logger.info(f"Testing directory download from gs://{bucket_name}/{gcs_test_dir} to {local_download_dir}")
    
    download_result = download_directory_from_gcs(
        bucket_name=bucket_name,
        gcs_dir=gcs_test_dir,
        local_dir=local_download_dir,
        project_id=project_id,
    )
    
    logger.info(f"Successfully downloaded {download_result['file_count']} files")
    
    # Test file exists
    if file_list:
        test_file = file_list[0]
        logger.info(f"Testing file existence for gs://{bucket_name}/{test_file}")
        
        exists = check_gcs_path_exists(
            bucket_name=bucket_name,
            gcs_path=test_file,
            project_id=project_id,
        )
        
        logger.info(f"File exists: {exists}")
    
    logger.info("GCS integration test completed successfully")

if __name__ == "__main__":
    main() 