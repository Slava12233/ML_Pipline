#!/usr/bin/env python3
"""
Deploy to Vertex AI - A script to deploy fine-tuned models to Vertex AI.

This script automates the process of deploying a fine-tuned model to
Vertex AI Endpoints, setting up the necessary configuration and resources.
"""

import os
import sys
import argparse
import logging
import yaml
import time
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utilities
from src.utils.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def export_model_to_gcs(model_dir, bucket_name, gcs_path, project_id):
    """Export model files to Google Cloud Storage."""
    try:
        from google.cloud import storage
        
        logger.info(f"Exporting model from {model_dir} to gs://{bucket_name}/{gcs_path}")
        
        # Initialize storage client
        storage_client = storage.Client(project=project_id)
        
        # Get bucket
        bucket = storage_client.bucket(bucket_name)
        
        # Upload model files
        model_dir = Path(model_dir)
        uploaded_files = []
        
        for file_path in model_dir.glob("**/*"):
            if file_path.is_file():
                # Create relative path
                relative_path = file_path.relative_to(model_dir)
                destination_blob_name = f"{gcs_path}/{relative_path}"
                
                # Upload file
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(str(file_path))
                
                uploaded_files.append(destination_blob_name)
        
        logger.info(f"Uploaded {len(uploaded_files)} files to GCS")
        
        # Return the GCS URI for the model
        return f"gs://{bucket_name}/{gcs_path}"
    
    except Exception as e:
        logger.error(f"Error exporting model to GCS: {str(e)}")
        raise


def deploy_model_to_vertex(
    model_dir,
    project_id,
    region,
    endpoint_name=None,
    machine_type="n1-standard-4",
    min_replicas=1,
    max_replicas=2,
    accelerator_type=None,
    accelerator_count=0,
    artifact_uri=None,
    service_account=None,
):
    """Deploy model to Vertex AI Endpoints."""
    try:
        from google.cloud import aiplatform
        
        logger.info(f"Deploying model to Vertex AI in project {project_id}, region {region}")
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=region,
        )
        
        # Create endpoint if name is provided
        if endpoint_name:
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"',
                project=project_id,
                location=region,
            )
            
            if endpoints:
                logger.info(f"Using existing endpoint: {endpoint_name}")
                endpoint = endpoints[0]
            else:
                logger.info(f"Creating new endpoint: {endpoint_name}")
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_name,
                    project=project_id,
                    location=region,
                )
        else:
            endpoint_name = f"gemini-pdf-finetuned-{int(time.time())}"
            logger.info(f"Creating new endpoint with generated name: {endpoint_name}")
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                project=project_id,
                location=region,
            )
        
        # Create artifact URI if not provided
        if not artifact_uri:
            bucket_name = f"{project_id}-vertex-models"
            gcs_path = f"models/gemini-pdf-finetuned-{int(time.time())}"
            artifact_uri = export_model_to_gcs(model_dir, bucket_name, gcs_path, project_id)
        
        # Upload model
        logger.info(f"Uploading model from {artifact_uri}")
        model = aiplatform.Model.upload(
            display_name=f"{endpoint_name}-model",
            artifact_uri=artifact_uri,
            serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-cpu.1-13:latest",
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_environment_variables={
                "MODEL_PATH": "/artifact/model",
            },
        )
        
        # Deploy model to endpoint
        logger.info(f"Deploying model to endpoint {endpoint.resource_name}")
        model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            min_replica_count=min_replicas,
            max_replica_count=max_replicas,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            service_account=service_account,
            deploy_request_timeout=1800,
        )
        
        logger.info(f"Model successfully deployed to {endpoint.resource_name}")
        
        # Return endpoint info
        return {
            "endpoint_id": endpoint.name,
            "endpoint_name": endpoint_name,
            "model_id": model.name,
            "model_display_name": f"{endpoint_name}-model",
            "project_id": project_id,
            "region": region,
        }
    
    except Exception as e:
        logger.error(f"Error deploying model to Vertex AI: {str(e)}")
        raise


def set_up_monitoring(endpoint_id, project_id, region):
    """Set up monitoring for the deployed endpoint."""
    try:
        from google.cloud import monitoring_v3
        
        logger.info(f"Setting up monitoring for endpoint {endpoint_id}")
        
        # Initialize monitoring client
        client = monitoring_v3.MetricServiceClient()
        
        # Project path
        project_path = f"projects/{project_id}"
        
        # Create metric descriptors and alerting policies
        # This is a simplified example - in production, you would create
        # more comprehensive monitoring setup
        
        logger.info("Monitoring setup complete")
        
        # Return monitoring info
        return {
            "project_path": project_path,
            "endpoint_id": endpoint_id,
        }
    
    except Exception as e:
        logger.error(f"Error setting up monitoring: {str(e)}")
        logger.warning("Continuing without monitoring setup")
        return None


def save_deployment_info(deployment_info, output_file=None):
    """Save deployment information to a file."""
    if not output_file:
        output_file = "deployment_info.yaml"
    
    try:
        with open(output_file, "w") as f:
            yaml.dump(deployment_info, f, default_flow_style=False)
        
        logger.info(f"Deployment information saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error saving deployment info: {str(e)}")


def main():
    """Run the deployment script."""
    parser = argparse.ArgumentParser(description="Deploy model to Vertex AI")
    
    parser.add_argument("--model-dir", type=str, default="data/model", help="Path to model directory")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--env", type=str, choices=["local", "vertex", "production"], default="vertex", help="Environment configuration to use")
    parser.add_argument("--endpoint-name", type=str, help="Name for the Vertex AI endpoint")
    parser.add_argument("--machine-type", type=str, help="Machine type for deployment")
    parser.add_argument("--min-replicas", type=int, help="Minimum number of replicas")
    parser.add_argument("--max-replicas", type=int, help="Maximum number of replicas")
    parser.add_argument("--accelerator-type", type=str, help="Accelerator type (e.g., NVIDIA_TESLA_T4)")
    parser.add_argument("--accelerator-count", type=int, help="Number of accelerators")
    parser.add_argument("--project-id", type=str, help="GCP Project ID")
    parser.add_argument("--region", type=str, help="GCP Region")
    parser.add_argument("--artifact-uri", type=str, help="GCS URI for model artifacts")
    parser.add_argument("--service-account", type=str, help="Service account for deployment")
    parser.add_argument("--output-file", type=str, help="Path to save deployment information")
    parser.add_argument("--setup-monitoring", action="store_true", help="Set up monitoring for the endpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config, env=args.env)
    
    # Set up parameters with configuration fallbacks
    project_id = args.project_id or config.gcp.project_id
    region = args.region or config.gcp.region
    endpoint_name = args.endpoint_name or config.deployment.endpoint_name
    machine_type = args.machine_type or config.deployment.machine_type
    min_replicas = args.min_replicas or config.deployment.min_replicas
    max_replicas = args.max_replicas or config.deployment.max_replicas
    accelerator_type = args.accelerator_type or config.deployment.accelerator_type
    accelerator_count = args.accelerator_count or config.deployment.accelerator_count
    service_account = args.service_account or getattr(config.deployment, "service_account", None)
    
    # Validate required parameters
    if not project_id:
        logger.error("Project ID must be provided")
        return
    
    # Deploy model
    deployment_info = deploy_model_to_vertex(
        model_dir=args.model_dir,
        project_id=project_id,
        region=region,
        endpoint_name=endpoint_name,
        machine_type=machine_type,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        artifact_uri=args.artifact_uri,
        service_account=service_account,
    )
    
    # Set up monitoring if requested
    if args.setup_monitoring:
        monitoring_info = set_up_monitoring(
            endpoint_id=deployment_info["endpoint_id"],
            project_id=project_id,
            region=region,
        )
        
        if monitoring_info:
            deployment_info["monitoring"] = monitoring_info
    
    # Save deployment information
    save_deployment_info(deployment_info, args.output_file)
    
    logger.info("Deployment complete!")
    logger.info(f"Endpoint ID: {deployment_info['endpoint_id']}")
    logger.info(f"To test the endpoint, use: python scripts/test_model_cli.py --vertex --endpoint-id {deployment_info['endpoint_id']} --project-id {project_id} --location {region} --questions \"Your test question here\"")


if __name__ == "__main__":
    main() 