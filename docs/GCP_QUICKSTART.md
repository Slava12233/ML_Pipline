# GCP Training Quick Start Guide

This guide provides step-by-step instructions for setting up and executing the Phase 1 tasks from our GCP Training Plan.

## Prerequisites

- Python 3.8+
- Git
- Google Cloud SDK
- A GCP project with billing enabled
- Appropriate GCP permissions

## Setup Steps

### 1. Clone and Configure Repository

```bash
# Clone the repository (if you haven't already)
git clone <your-repo-url>
cd <repo-directory>

# Install dependencies
pip install -r requirements.txt

# Set up your Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 2. Configure GCP Authentication

```bash
# Install Google Cloud SDK (if needed)
# https://cloud.google.com/sdk/docs/install

# Login to GCP
gcloud auth login

# Set your project
gcloud config set project <your-project-id>

# Create application default credentials
gcloud auth application-default login
```

### 3. Update Configuration Files

Create a new Vertex AI configuration file:

```bash
# Copy the example configuration
cp config/config.yaml config/vertex_config.yaml

# Edit the file with your preferred editor
nano config/vertex_config.yaml
```

Update the following sections in `vertex_config.yaml`:

```yaml
# GCP Project Settings
gcp:
  project_id: <your-project-id>  # Replace with your project ID
  region: us-central1
  zone: us-central1-a
  storage:
    bucket_name: <your-bucket-name>  # Replace with your bucket name
    pdf_folder: pdfs
    extracted_text_folder: extracted_text
    training_data_folder: training_data
    model_folder: models

# Vertex AI Settings
vertex_ai:
  machine_type: n1-standard-8
  accelerator_type: NVIDIA_TESLA_T4
  accelerator_count: 1
  container_uri: gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest
  
# Environment
environment: vertex
```

### 4. Test Local Pipeline Components

```bash
# Process PDFs
python scripts/run.py pdf process data/pdfs data/extracted_text --config config/vertex_config.yaml --env vertex

# Prepare training data
python scripts/run.py data prepare data/extracted_text data/training_data --config config/vertex_config.yaml --env vertex

# Run a small fine-tuning job locally (to test the code)
python scripts/run.py train finetune data/training_data data/model --config config/vertex_config.yaml --env vertex
```

### 5. Set Up GCS Utilities

Create a utility script to upload data to GCS:

```bash
# Create directories if needed
mkdir -p src/utils/gcp

# Create GCS utility file
touch src/utils/gcp/storage.py
```

Edit `src/utils/gcp/storage.py` with:

```python
"""GCS storage utilities."""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any

from google.cloud import storage

def upload_directory_to_gcs(
    local_dir: str,
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
    
    for local_file in local_dir_path.glob("**/*"):
        if local_file.is_file():
            # Create GCS path
            rel_path = local_file.relative_to(local_dir_path)
            gcs_path = f"{gcs_dir}/{rel_path}"
            
            # Upload file
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_file))
            
            uploaded_files.append(gcs_path)
    
    return {
        "bucket": bucket_name,
        "gcs_dir": gcs_dir,
        "uploaded_files": uploaded_files,
        "file_count": len(uploaded_files),
    }

def download_directory_from_gcs(
    bucket_name: str,
    gcs_dir: str,
    local_dir: str,
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
    os.makedirs(local_dir, exist_ok=True)
    
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
        local_path = os.path.join(local_dir, rel_path)
        
        # Create directories if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        blob.download_to_filename(local_path)
        
        downloaded_files.append(local_path)
    
    return {
        "bucket": bucket_name,
        "gcs_dir": gcs_dir,
        "downloaded_files": downloaded_files,
        "file_count": len(downloaded_files),
    }
```

### 6. Test GCS Integration

Upload test data to GCS:

```python
# Create a test script
cat > scripts/test_gcs.py << 'EOF'
#!/usr/bin/env python3
"""Test GCS integration."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.gcp.storage import upload_directory_to_gcs, download_directory_from_gcs
from src.utils.config import get_config

def main():
    """Run GCS test."""
    # Get configuration
    config = get_config("config/vertex_config.yaml", env="vertex")
    
    # Get GCP settings
    project_id = config.gcp.project_id
    bucket_name = config.gcp.storage.bucket_name
    
    # Upload test data
    print(f"Uploading data to GCS bucket: {bucket_name}")
    upload_result = upload_directory_to_gcs(
        local_dir="data/pdfs",
        bucket_name=bucket_name,
        gcs_dir="test/pdfs",
        project_id=project_id,
    )
    
    print(f"Uploaded {upload_result['file_count']} files")
    
    # Download test data
    print(f"Downloading data from GCS bucket: {bucket_name}")
    download_result = download_directory_from_gcs(
        bucket_name=bucket_name,
        gcs_dir="test/pdfs",
        local_dir="data/test_download",
        project_id=project_id,
    )
    
    print(f"Downloaded {download_result['file_count']} files")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x scripts/test_gcs.py

# Run the test
python scripts/test_gcs.py
```

### 7. Check GCS Permissions

Ensure your GCS bucket has the correct permissions:

```bash
# Create a bucket if needed
gsutil mb -p <your-project-id> -l us-central1 gs://<your-bucket-name>

# Set appropriate access control
gsutil iam ch serviceAccount:<your-service-account>:roles/storage.objectAdmin gs://<your-bucket-name>
```

## Next Steps

After completing these steps, you'll have:

1. Set up the local environment for GCP integration
2. Created the necessary configuration files
3. Implemented basic GCS utilities
4. Tested the local components of the pipeline

You're now ready to proceed to Phase 2: GCP Environment Setup.

## Troubleshooting

### Permission Issues

If you encounter permission errors:

```bash
# Verify your gcloud authentication
gcloud auth list

# Check your application default credentials
gcloud auth application-default print-access-token

# Ensure the service account has the necessary permissions
gcloud projects add-iam-policy-binding <your-project-id> \
    --member="serviceAccount:<your-service-account>" \
    --role="roles/storage.admin"
```

### GCS Bucket Issues

If you have issues with the GCS bucket:

```bash
# Check if the bucket exists
gsutil ls -p <your-project-id>

# Check bucket permissions
gsutil iam get gs://<your-bucket-name>
```

### Local Environment Issues

If you encounter issues with your local environment:

```bash
# Check your PYTHONPATH
echo $PYTHONPATH

# Ensure the config file exists
ls -la config/vertex_config.yaml

# Verify the config file contents
cat config/vertex_config.yaml
``` 