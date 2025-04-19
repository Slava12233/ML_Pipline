"""
Vertex AI deployment module.

This module provides functions for deploying models to Vertex AI.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform import Model
from google.cloud.aiplatform.models import Endpoint

logger = logging.getLogger(__name__)


def initialize_vertex_ai(
    project_id: str,
    region: str,
    staging_bucket: Optional[str] = None,
) -> None:
    """
    Initialize Vertex AI.

    Args:
        project_id: Google Cloud project ID.
        region: Google Cloud region.
        staging_bucket: Google Cloud Storage bucket for staging.
    """
    # Initialize Vertex AI SDK
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
    )
    
    logger.info(f"Initialized Vertex AI with project_id={project_id}, region={region}")


def upload_model(
    model_dir: Union[str, Path],
    display_name: str,
    serving_container_image_uri: str,
    artifact_uri: Optional[str] = None,
    serving_container_predict_route: str = "/predict",
    serving_container_health_route: str = "/health",
    description: Optional[str] = None,
    serving_container_command: Optional[List[str]] = None,
    serving_container_args: Optional[List[str]] = None,
    serving_container_environment_variables: Optional[Dict[str, str]] = None,
    serving_container_ports: Optional[List[int]] = None,
    explanation_metadata: Optional[Dict[str, Any]] = None,
    explanation_parameters: Optional[Dict[str, Any]] = None,
    sync: bool = True,
) -> Model:
    """
    Upload model to Vertex AI Model Registry.

    Args:
        model_dir: Directory containing the model.
        display_name: Display name for the model.
        serving_container_image_uri: URI of the serving container image.
        artifact_uri: URI of the model artifacts in GCS.
        serving_container_predict_route: HTTP path to send prediction requests.
        serving_container_health_route: HTTP path to send health check requests.
        description: Description of the model.
        serving_container_command: Command to run when the container starts.
        serving_container_args: Arguments to the command.
        serving_container_environment_variables: Environment variables.
        serving_container_ports: Ports to expose.
        explanation_metadata: Metadata for model explanations.
        explanation_parameters: Parameters for model explanations.
        sync: Whether to wait for the operation to complete.

    Returns:
        Model: Vertex AI Model.
    """
    model_dir = Path(model_dir)
    
    # If artifact_uri is not provided, upload model artifacts to GCS
    if artifact_uri is None:
        # Get default staging bucket
        staging_bucket = aiplatform.initializer.global_config.staging_bucket
        
        if staging_bucket is None:
            raise ValueError("No staging bucket provided. Either provide artifact_uri or initialize Vertex AI with a staging bucket.")
        
        # Create artifact URI
        artifact_uri = f"gs://{staging_bucket}/{display_name}"
        
        # Upload model artifacts
        logger.info(f"Uploading model artifacts to {artifact_uri}")
        # TODO: Implement model artifact upload
    
    # Create model
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route=serving_container_predict_route,
        serving_container_health_route=serving_container_health_route,
        description=description,
        serving_container_command=serving_container_command,
        serving_container_args=serving_container_args,
        serving_container_environment_variables=serving_container_environment_variables,
        serving_container_ports=serving_container_ports,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        sync=sync,
    )
    
    logger.info(f"Uploaded model {display_name} to Vertex AI Model Registry")
    return model


def create_endpoint(
    display_name: str,
    description: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Endpoint:
    """
    Create Vertex AI Endpoint.

    Args:
        display_name: Display name for the endpoint.
        description: Description of the endpoint.
        project: Google Cloud project ID.
        location: Google Cloud region.
        labels: Labels to associate with the endpoint.

    Returns:
        Endpoint: Vertex AI Endpoint.
    """
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=display_name,
        description=description,
        project=project,
        location=location,
        labels=labels,
    )
    
    logger.info(f"Created endpoint {display_name}")
    return endpoint


def deploy_model(
    model: Union[Model, str],
    endpoint: Union[Endpoint, str],
    deployed_model_display_name: Optional[str] = None,
    traffic_percentage: int = 100,
    machine_type: str = "n1-standard-4",
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    accelerator_type: Optional[str] = None,
    accelerator_count: Optional[int] = None,
    service_account: Optional[str] = None,
    explanation_metadata: Optional[Dict[str, Any]] = None,
    explanation_parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    sync: bool = True,
) -> Endpoint:
    """
    Deploy model to Vertex AI Endpoint.

    Args:
        model: Vertex AI Model or model ID.
        endpoint: Vertex AI Endpoint or endpoint ID.
        deployed_model_display_name: Display name for the deployed model.
        traffic_percentage: Percentage of traffic to route to this model.
        machine_type: Machine type for the deployment.
        min_replica_count: Minimum number of replicas.
        max_replica_count: Maximum number of replicas.
        accelerator_type: Type of accelerator to attach to the machine.
        accelerator_count: Number of accelerators to attach to the machine.
        service_account: Service account to use for deployment.
        explanation_metadata: Metadata for model explanations.
        explanation_parameters: Parameters for model explanations.
        metadata: Additional metadata.
        sync: Whether to wait for the operation to complete.

    Returns:
        Endpoint: Vertex AI Endpoint.
    """
    # Get model if model ID is provided
    if isinstance(model, str):
        model = aiplatform.Model(model)
    
    # Get endpoint if endpoint ID is provided
    if isinstance(endpoint, str):
        endpoint = aiplatform.Endpoint(endpoint)
    
    # Set deployed model display name if not provided
    if deployed_model_display_name is None:
        deployed_model_display_name = f"{model.display_name}-deployment"
    
    # Deploy model
    endpoint = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        traffic_percentage=traffic_percentage,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        service_account=service_account,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        metadata=metadata,
        sync=sync,
    )
    
    logger.info(f"Deployed model {model.display_name} to endpoint {endpoint.display_name}")
    return endpoint


def predict(
    endpoint: Union[Endpoint, str],
    instances: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Make predictions with deployed model.

    Args:
        endpoint: Vertex AI Endpoint or endpoint ID.
        instances: Instances to predict.
        parameters: Additional parameters.

    Returns:
        Dict[str, Any]: Prediction results.
    """
    # Get endpoint if endpoint ID is provided
    if isinstance(endpoint, str):
        endpoint = aiplatform.Endpoint(endpoint)
    
    # Make prediction
    response = endpoint.predict(
        instances=instances,
        parameters=parameters,
    )
    
    return response


def undeploy_model(
    endpoint: Union[Endpoint, str],
    deployed_model_id: str,
    sync: bool = True,
) -> None:
    """
    Undeploy model from Vertex AI Endpoint.

    Args:
        endpoint: Vertex AI Endpoint or endpoint ID.
        deployed_model_id: ID of the deployed model.
        sync: Whether to wait for the operation to complete.
    """
    # Get endpoint if endpoint ID is provided
    if isinstance(endpoint, str):
        endpoint = aiplatform.Endpoint(endpoint)
    
    # Undeploy model
    endpoint.undeploy(
        deployed_model_id=deployed_model_id,
        sync=sync,
    )
    
    logger.info(f"Undeployed model {deployed_model_id} from endpoint {endpoint.display_name}")


def delete_endpoint(
    endpoint: Union[Endpoint, str],
    sync: bool = True,
) -> None:
    """
    Delete Vertex AI Endpoint.

    Args:
        endpoint: Vertex AI Endpoint or endpoint ID.
        sync: Whether to wait for the operation to complete.
    """
    # Get endpoint if endpoint ID is provided
    if isinstance(endpoint, str):
        endpoint = aiplatform.Endpoint(endpoint)
    
    # Delete endpoint
    endpoint.delete(
        sync=sync,
    )
    
    logger.info(f"Deleted endpoint {endpoint.display_name}")


def delete_model(
    model: Union[Model, str],
    sync: bool = True,
) -> None:
    """
    Delete Vertex AI Model.

    Args:
        model: Vertex AI Model or model ID.
        sync: Whether to wait for the operation to complete.
    """
    # Get model if model ID is provided
    if isinstance(model, str):
        model = aiplatform.Model(model)
    
    # Delete model
    model.delete(
        sync=sync,
    )
    
    logger.info(f"Deleted model {model.display_name}")


def get_model_info(
    model: Union[Model, str],
) -> Dict[str, Any]:
    """
    Get model information.

    Args:
        model: Vertex AI Model or model ID.

    Returns:
        Dict[str, Any]: Model information.
    """
    # Get model if model ID is provided
    if isinstance(model, str):
        model = aiplatform.Model(model)
    
    # Get model information
    model_info = {
        "resource_name": model.resource_name,
        "display_name": model.display_name,
        "description": model.description,
        "version_id": model.version_id,
        "version_aliases": model.version_aliases,
        "create_time": model.create_time,
        "update_time": model.update_time,
        "deployed_models": model.deployed_models,
    }
    
    return model_info


def get_endpoint_info(
    endpoint: Union[Endpoint, str],
) -> Dict[str, Any]:
    """
    Get endpoint information.

    Args:
        endpoint: Vertex AI Endpoint or endpoint ID.

    Returns:
        Dict[str, Any]: Endpoint information.
    """
    # Get endpoint if endpoint ID is provided
    if isinstance(endpoint, str):
        endpoint = aiplatform.Endpoint(endpoint)
    
    # Get endpoint information
    endpoint_info = {
        "resource_name": endpoint.resource_name,
        "display_name": endpoint.display_name,
        "description": endpoint.description,
        "deployed_models": endpoint.deployed_models,
        "traffic_split": endpoint.traffic_split,
        "create_time": endpoint.create_time,
        "update_time": endpoint.update_time,
    }
    
    return endpoint_info


def list_models(
    filter: Optional[str] = None,
    order_by: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    credentials: Optional[Any] = None,
) -> List[Model]:
    """
    List Vertex AI Models.

    Args:
        filter: Filter expression.
        order_by: Order by expression.
        project: Google Cloud project ID.
        location: Google Cloud region.
        credentials: Google Cloud credentials.

    Returns:
        List[Model]: List of Vertex AI Models.
    """
    # List models
    models = aiplatform.Model.list(
        filter=filter,
        order_by=order_by,
        project=project,
        location=location,
        credentials=credentials,
    )
    
    return models


def list_endpoints(
    filter: Optional[str] = None,
    order_by: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    credentials: Optional[Any] = None,
) -> List[Endpoint]:
    """
    List Vertex AI Endpoints.

    Args:
        filter: Filter expression.
        order_by: Order by expression.
        project: Google Cloud project ID.
        location: Google Cloud region.
        credentials: Google Cloud credentials.

    Returns:
        List[Endpoint]: List of Vertex AI Endpoints.
    """
    # List endpoints
    endpoints = aiplatform.Endpoint.list(
        filter=filter,
        order_by=order_by,
        project=project,
        location=location,
        credentials=credentials,
    )
    
    return endpoints


def deploy_model_from_config(
    model_dir: Union[str, Path],
    config_path: Union[str, Path],
) -> Endpoint:
    """
    Deploy model from configuration file.

    Args:
        model_dir: Directory containing the model.
        config_path: Path to configuration file.

    Returns:
        Endpoint: Vertex AI Endpoint.
    """
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Extract configuration
    project_id = config.get("gcp", {}).get("project_id")
    region = config.get("gcp", {}).get("region")
    staging_bucket = config.get("gcp", {}).get("storage", {}).get("bucket_name")
    
    model_config = config.get("deployment", {})
    model_name = model_config.get("model_name", "gemini-finetuned")
    endpoint_name = model_config.get("endpoint_name", "gemini-finetuned-endpoint")
    serving_container_image_uri = model_config.get("serving_container_image_uri")
    machine_type = model_config.get("machine_type", "n1-standard-4")
    min_replica_count = model_config.get("min_replicas", 1)
    max_replica_count = model_config.get("max_replicas", 1)
    
    # Initialize Vertex AI
    initialize_vertex_ai(
        project_id=project_id,
        region=region,
        staging_bucket=staging_bucket,
    )
    
    # Upload model
    model = upload_model(
        model_dir=model_dir,
        display_name=model_name,
        serving_container_image_uri=serving_container_image_uri,
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
    
    return endpoint
