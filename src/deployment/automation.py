"""
Deployment automation module.

This module provides functions for automating model deployment.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import yaml
from google.cloud import storage, aiplatform

from src.deployment.vertex import (
    initialize_vertex_ai,
    upload_model,
    create_endpoint,
    deploy_model,
    get_model_info,
    get_endpoint_info,
)
from src.deployment.monitoring import setup_monitoring
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def upload_model_artifacts(
    model_dir: Union[str, Path],
    bucket_name: str,
    destination_blob_name: Optional[str] = None,
    credentials: Optional[Any] = None,
) -> str:
    """
    Upload model artifacts to GCS.

    Args:
        model_dir: Directory containing model artifacts.
        bucket_name: GCS bucket name.
        destination_blob_name: Destination blob name.
        credentials: Google Cloud credentials.

    Returns:
        str: GCS URI of uploaded model artifacts.
    """
    model_dir = Path(model_dir)
    
    # Set destination blob name if not provided
    if destination_blob_name is None:
        destination_blob_name = model_dir.name
    
    # Initialize storage client
    storage_client = storage.Client(credentials=credentials)
    
    # Get bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Upload model artifacts
    for file_path in model_dir.glob("**/*"):
        if file_path.is_file():
            # Get relative path
            relative_path = file_path.relative_to(model_dir)
            
            # Create blob name
            blob_name = f"{destination_blob_name}/{relative_path}"
            
            # Upload file
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            
            logger.info(f"Uploaded {file_path} to {blob_name}")
    
    # Return GCS URI
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    
    logger.info(f"Uploaded model artifacts to {gcs_uri}")
    return gcs_uri


def create_deployment_config(
    model_dir: Union[str, Path],
    config_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    serving_container_image_uri: Optional[str] = None,
    machine_type: Optional[str] = None,
    min_replica_count: Optional[int] = None,
    max_replica_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create deployment configuration.

    Args:
        model_dir: Directory containing model artifacts.
        config_path: Path to configuration file.
        output_path: Path to output configuration file.
        model_name: Model name.
        endpoint_name: Endpoint name.
        serving_container_image_uri: Serving container image URI.
        machine_type: Machine type.
        min_replica_count: Minimum replica count.
        max_replica_count: Maximum replica count.

    Returns:
        Dict[str, Any]: Deployment configuration.
    """
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration
    project_id = config.get("gcp", {}).get("project_id")
    region = config.get("gcp", {}).get("region")
    staging_bucket = config.get("gcp", {}).get("storage", {}).get("bucket_name")
    
    # Create deployment configuration
    deployment_config = {
        "gcp": {
            "project_id": project_id,
            "region": region,
            "storage": {
                "bucket_name": staging_bucket,
            },
        },
        "deployment": {
            "model_name": model_name or f"gemini-finetuned-{int(time.time())}",
            "endpoint_name": endpoint_name or f"gemini-finetuned-endpoint-{int(time.time())}",
            "serving_container_image_uri": serving_container_image_uri or "us-docker.pkg.dev/vertex-ai/prediction/text-generation:latest",
            "machine_type": machine_type or "n1-standard-4",
            "min_replicas": min_replica_count or 1,
            "max_replicas": max_replica_count or 1,
        },
        "monitoring": {
            "enable_monitoring": True,
            "enable_logging": True,
        },
    }
    
    # Save deployment configuration if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(output_path, "w") as f:
            yaml.dump(deployment_config, f, default_flow_style=False)
        
        logger.info(f"Saved deployment configuration to {output_path}")
    
    return deployment_config


def deploy_model_with_config(
    model_dir: Union[str, Path],
    config_path: Union[str, Path],
    artifact_uri: Optional[str] = None,
    credentials: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Deploy model with configuration.

    Args:
        model_dir: Directory containing model artifacts.
        config_path: Path to configuration file.
        artifact_uri: URI of model artifacts in GCS.
        credentials: Google Cloud credentials.

    Returns:
        Dict[str, Any]: Deployment information.
    """
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration
    project_id = config.get("gcp", {}).get("project_id")
    region = config.get("gcp", {}).get("region")
    staging_bucket = config.get("gcp", {}).get("storage", {}).get("bucket_name")
    
    model_config = config.get("deployment", {})
    model_name = model_config.get("model_name")
    endpoint_name = model_config.get("endpoint_name")
    serving_container_image_uri = model_config.get("serving_container_image_uri")
    machine_type = model_config.get("machine_type")
    min_replica_count = model_config.get("min_replicas")
    max_replica_count = model_config.get("max_replicas")
    
    monitoring_config = config.get("monitoring", {})
    enable_monitoring = monitoring_config.get("enable_monitoring", True)
    enable_logging = monitoring_config.get("enable_logging", True)
    
    # Initialize Vertex AI
    initialize_vertex_ai(
        project_id=project_id,
        region=region,
        staging_bucket=staging_bucket,
    )
    
    # Upload model artifacts if not provided
    if artifact_uri is None:
        artifact_uri = upload_model_artifacts(
            model_dir=model_dir,
            bucket_name=staging_bucket,
            destination_blob_name=model_name,
            credentials=credentials,
        )
    
    # Upload model
    model = upload_model(
        model_dir=model_dir,
        display_name=model_name,
        serving_container_image_uri=serving_container_image_uri,
        artifact_uri=artifact_uri,
    )
    
    # Create endpoint
    endpoint = create_endpoint(
        display_name=endpoint_name,
    )
    
    # Deploy model
    endpoint = deploy_model(
        model=model,
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
    )
    
    # Set up monitoring if enabled
    if enable_monitoring:
        setup_monitoring(config_path)
    
    # Get deployment information
    model_info = get_model_info(model)
    endpoint_info = get_endpoint_info(endpoint)
    
    # Create deployment information
    deployment_info = {
        "model": model_info,
        "endpoint": endpoint_info,
        "artifact_uri": artifact_uri,
        "config": config,
    }
    
    logger.info(f"Deployed model {model_name} to endpoint {endpoint_name}")
    return deployment_info


def create_deployment_script(
    model_dir: Union[str, Path],
    config_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Create deployment script.

    Args:
        model_dir: Directory containing model artifacts.
        config_path: Path to configuration file.
        output_path: Path to output script file.

    Returns:
        str: Path to deployment script.
    """
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    
    # Set output path if not provided
    if output_path is None:
        output_path = model_dir / "deploy.sh"
    
    output_path = Path(output_path)
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration
    project_id = config.get("gcp", {}).get("project_id")
    region = config.get("gcp", {}).get("region")
    staging_bucket = config.get("gcp", {}).get("storage", {}).get("bucket_name")
    
    model_config = config.get("deployment", {})
    model_name = model_config.get("model_name")
    endpoint_name = model_config.get("endpoint_name")
    serving_container_image_uri = model_config.get("serving_container_image_uri")
    machine_type = model_config.get("machine_type")
    min_replica_count = model_config.get("min_replicas")
    max_replica_count = model_config.get("max_replicas")
    
    # Create deployment script
    script = f"""#!/bin/bash

# Deployment script for {model_name}

# Set environment variables
export PROJECT_ID="{project_id}"
export REGION="{region}"
export BUCKET_NAME="{staging_bucket}"
export MODEL_NAME="{model_name}"
export ENDPOINT_NAME="{endpoint_name}"
export SERVING_CONTAINER_IMAGE_URI="{serving_container_image_uri}"
export MACHINE_TYPE="{machine_type}"
export MIN_REPLICA_COUNT="{min_replica_count}"
export MAX_REPLICA_COUNT="{max_replica_count}"
export MODEL_DIR="{model_dir}"

# Upload model artifacts to GCS
echo "Uploading model artifacts to GCS..."
gsutil -m cp -r $MODEL_DIR/* gs://$BUCKET_NAME/$MODEL_NAME/

# Create model
echo "Creating model..."
MODEL_ID=$(gcloud ai models upload \\
  --region=$REGION \\
  --display-name=$MODEL_NAME \\
  --artifact-uri=gs://$BUCKET_NAME/$MODEL_NAME/ \\
  --container-image-uri=$SERVING_CONTAINER_IMAGE_URI \\
  --format="value(name)")

echo "Model ID: $MODEL_ID"

# Create endpoint
echo "Creating endpoint..."
ENDPOINT_ID=$(gcloud ai endpoints create \\
  --region=$REGION \\
  --display-name=$ENDPOINT_NAME \\
  --format="value(name)")

echo "Endpoint ID: $ENDPOINT_ID"

# Deploy model to endpoint
echo "Deploying model to endpoint..."
gcloud ai endpoints deploy-model $ENDPOINT_ID \\
  --region=$REGION \\
  --model=$MODEL_ID \\
  --display-name=$MODEL_NAME-deployment \\
  --machine-type=$MACHINE_TYPE \\
  --min-replica-count=$MIN_REPLICA_COUNT \\
  --max-replica-count=$MAX_REPLICA_COUNT \\
  --traffic-split=0=100

echo "Deployment complete!"
"""
    
    # Save deployment script
    with open(output_path, "w") as f:
        f.write(script)
    
    # Make script executable
    os.chmod(output_path, 0o755)
    
    logger.info(f"Created deployment script at {output_path}")
    return str(output_path)


def run_deployment_script(
    script_path: Union[str, Path],
) -> subprocess.CompletedProcess:
    """
    Run deployment script.

    Args:
        script_path: Path to deployment script.

    Returns:
        subprocess.CompletedProcess: Completed process.
    """
    script_path = Path(script_path)
    
    # Run deployment script
    process = subprocess.run(
        [str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    
    logger.info(f"Ran deployment script {script_path}")
    return process


def create_deployment_pipeline(
    model_dir: Union[str, Path],
    config_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Create deployment pipeline.

    Args:
        model_dir: Directory containing model artifacts.
        config_path: Path to configuration file.
        output_dir: Path to output directory.

    Returns:
        Dict[str, Any]: Deployment pipeline information.
    """
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    
    # Set output directory if not provided
    if output_dir is None:
        output_dir = model_dir / "deployment"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create deployment configuration
    deployment_config = create_deployment_config(
        model_dir=model_dir,
        config_path=config_path,
        output_path=output_dir / "deployment_config.yaml",
    )
    
    # Create deployment script
    script_path = create_deployment_script(
        model_dir=model_dir,
        config_path=output_dir / "deployment_config.yaml",
        output_path=output_dir / "deploy.sh",
    )
    
    # Create deployment pipeline information
    pipeline_info = {
        "model_dir": str(model_dir),
        "config_path": str(output_dir / "deployment_config.yaml"),
        "script_path": script_path,
        "config": deployment_config,
    }
    
    # Save pipeline information
    with open(output_dir / "pipeline_info.json", "w") as f:
        json.dump(pipeline_info, f, indent=2)
    
    logger.info(f"Created deployment pipeline in {output_dir}")
    return pipeline_info


def create_deployment_job(
    model_dir: Union[str, Path],
    config_path: Union[str, Path],
    schedule: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create deployment job.

    Args:
        model_dir: Directory containing model artifacts.
        config_path: Path to configuration file.
        schedule: Cron schedule.

    Returns:
        Dict[str, Any]: Deployment job information.
    """
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration
    project_id = config.get("gcp", {}).get("project_id")
    region = config.get("gcp", {}).get("region")
    
    model_config = config.get("deployment", {})
    model_name = model_config.get("model_name")
    
    # Create deployment pipeline
    pipeline_info = create_deployment_pipeline(
        model_dir=model_dir,
        config_path=config_path,
        output_dir=model_dir / "deployment",
    )
    
    # Create deployment job
    job_id = f"deploy-{model_name}-{int(time.time())}"
    
    # Create job command
    job_command = f"bash {pipeline_info['script_path']}"
    
    # Create job
    if schedule is None:
        # Create one-time job
        job = subprocess.run(
            [
                "gcloud",
                "scheduler",
                "jobs",
                "create",
                "http",
                job_id,
                f"--location={region}",
                f"--schedule=once",
                f"--uri=https://cloudbuild.googleapis.com/v1/projects/{project_id}/builds",
                f"--message-body={{'steps': [{'name': 'gcr.io/cloud-builders/bash', 'args': ['-c', '{job_command}']}]}}",
                "--oauth-service-account-email=default",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        # Create scheduled job
        job = subprocess.run(
            [
                "gcloud",
                "scheduler",
                "jobs",
                "create",
                "http",
                job_id,
                f"--location={region}",
                f"--schedule={schedule}",
                f"--uri=https://cloudbuild.googleapis.com/v1/projects/{project_id}/builds",
                f"--message-body={{'steps': [{'name': 'gcr.io/cloud-builders/bash', 'args': ['-c', '{job_command}']}]}}",
                "--oauth-service-account-email=default",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    
    # Create job information
    job_info = {
        "job_id": job_id,
        "project_id": project_id,
        "region": region,
        "schedule": schedule,
        "command": job_command,
        "pipeline_info": pipeline_info,
    }
    
    logger.info(f"Created deployment job {job_id}")
    return job_info


def main():
    """Main function."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deploy model")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing model artifacts")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, help="Path to output directory")
    parser.add_argument("--script", action="store_true", help="Create deployment script")
    parser.add_argument("--pipeline", action="store_true", help="Create deployment pipeline")
    parser.add_argument("--job", action="store_true", help="Create deployment job")
    parser.add_argument("--schedule", type=str, help="Cron schedule for deployment job")
    parser.add_argument("--deploy", action="store_true", help="Deploy model")
    args = parser.parse_args()
    
    # Create deployment script
    if args.script:
        create_deployment_script(
            model_dir=args.model_dir,
            config_path=args.config,
            output_path=args.output_dir,
        )
    
    # Create deployment pipeline
    if args.pipeline:
        create_deployment_pipeline(
            model_dir=args.model_dir,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    
    # Create deployment job
    if args.job:
        create_deployment_job(
            model_dir=args.model_dir,
            config_path=args.config,
            schedule=args.schedule,
        )
    
    # Deploy model
    if args.deploy:
        deploy_model_with_config(
            model_dir=args.model_dir,
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
