"""
Tests for deployment modules.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import Model
from google.cloud.aiplatform.models import Endpoint

from src.deployment.vertex import (
    initialize_vertex_ai,
    upload_model,
    create_endpoint,
    deploy_model,
    predict,
    get_model_info,
    get_endpoint_info,
)
from src.deployment.api import (
    APIConfig,
    PredictionRequest,
    PredictionResponse,
)
from src.deployment.monitoring import (
    MonitoringClient,
    LoggingClient,
    ModelMonitor,
)
from src.deployment.automation import (
    upload_model_artifacts,
    create_deployment_config,
    deploy_model_with_config,
    create_deployment_script,
    create_deployment_pipeline,
)


@pytest.fixture
def mock_vertex_client():
    """Mock Vertex AI client."""
    with mock.patch("google.cloud.aiplatform.Model") as mock_model:
        with mock.patch("google.cloud.aiplatform.Endpoint") as mock_endpoint:
            yield mock_model, mock_endpoint


@pytest.fixture
def mock_storage_client():
    """Mock Storage client."""
    with mock.patch("google.cloud.storage.Client") as mock_client:
        yield mock_client


@pytest.fixture
def mock_monitoring_client():
    """Mock Monitoring client."""
    with mock.patch("google.cloud.monitoring_v3.MetricServiceClient") as mock_client:
        yield mock_client


@pytest.fixture
def mock_logging_client():
    """Mock Logging client."""
    with mock.patch("google.cloud.logging_v2.LoggingServiceV2Client") as mock_client:
        yield mock_client


@pytest.fixture
def model_dir():
    """Create model directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "model"
        model_dir.mkdir()
        
        # Create model files
        (model_dir / "model.pt").touch()
        (model_dir / "tokenizer.json").touch()
        (model_dir / "config.json").touch()
        
        yield model_dir


@pytest.fixture
def config_file():
    """Create configuration file."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
        config = {
            "gcp": {
                "project_id": "test-project",
                "region": "us-central1",
                "storage": {
                    "bucket_name": "test-bucket",
                },
            },
            "deployment": {
                "model_name": "test-model",
                "endpoint_name": "test-endpoint",
                "serving_container_image_uri": "us-docker.pkg.dev/vertex-ai/prediction/text-generation:latest",
                "machine_type": "n1-standard-4",
                "min_replicas": 1,
                "max_replicas": 1,
            },
            "monitoring": {
                "enable_monitoring": True,
                "enable_logging": True,
            },
        }
        
        yaml.dump(config, temp_file)
        
        yield temp_file.name
        
        # Clean up
        os.unlink(temp_file.name)


def test_initialize_vertex_ai():
    """Test initializing Vertex AI."""
    with mock.patch("google.cloud.aiplatform.init") as mock_init:
        initialize_vertex_ai(
            project_id="test-project",
            region="us-central1",
            staging_bucket="test-bucket",
        )
        
        mock_init.assert_called_once_with(
            project="test-project",
            location="us-central1",
            staging_bucket="test-bucket",
        )


def test_upload_model(mock_vertex_client):
    """Test uploading model."""
    mock_model, _ = mock_vertex_client
    
    # Mock upload method
    mock_model.upload.return_value = mock_model
    
    # Upload model
    model = upload_model(
        model_dir="test-model",
        display_name="test-model",
        serving_container_image_uri="test-image",
        artifact_uri="gs://test-bucket/test-model",
    )
    
    # Check model
    assert model == mock_model
    
    # Check upload call
    mock_model.upload.assert_called_once_with(
        display_name="test-model",
        artifact_uri="gs://test-bucket/test-model",
        serving_container_image_uri="test-image",
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        description=None,
        serving_container_command=None,
        serving_container_args=None,
        serving_container_environment_variables=None,
        serving_container_ports=None,
        explanation_metadata=None,
        explanation_parameters=None,
        sync=True,
    )


def test_create_endpoint(mock_vertex_client):
    """Test creating endpoint."""
    _, mock_endpoint = mock_vertex_client
    
    # Mock create method
    mock_endpoint.create.return_value = mock_endpoint
    
    # Create endpoint
    endpoint = create_endpoint(
        display_name="test-endpoint",
    )
    
    # Check endpoint
    assert endpoint == mock_endpoint
    
    # Check create call
    mock_endpoint.create.assert_called_once_with(
        display_name="test-endpoint",
        description=None,
        project=None,
        location=None,
        labels=None,
    )


def test_deploy_model(mock_vertex_client):
    """Test deploying model."""
    mock_model, mock_endpoint = mock_vertex_client
    
    # Mock deploy method
    mock_model.deploy.return_value = mock_endpoint
    
    # Deploy model
    endpoint = deploy_model(
        model=mock_model,
        endpoint=mock_endpoint,
        machine_type="n1-standard-4",
    )
    
    # Check endpoint
    assert endpoint == mock_endpoint
    
    # Check deploy call
    mock_model.deploy.assert_called_once()


def test_predict(mock_vertex_client):
    """Test making predictions."""
    _, mock_endpoint = mock_vertex_client
    
    # Mock predict method
    mock_endpoint.predict.return_value = {"predictions": [{"text": "test"}]}
    
    # Make prediction
    response = predict(
        endpoint=mock_endpoint,
        instances=[{"text": "test"}],
    )
    
    # Check response
    assert response == {"predictions": [{"text": "test"}]}
    
    # Check predict call
    mock_endpoint.predict.assert_called_once_with(
        instances=[{"text": "test"}],
        parameters=None,
    )


def test_get_model_info(mock_vertex_client):
    """Test getting model information."""
    mock_model, _ = mock_vertex_client
    
    # Set model attributes
    mock_model.resource_name = "projects/test-project/models/test-model"
    mock_model.display_name = "test-model"
    mock_model.description = "Test model"
    mock_model.version_id = "1"
    mock_model.version_aliases = ["latest"]
    mock_model.create_time = "2023-01-01T00:00:00Z"
    mock_model.update_time = "2023-01-01T00:00:00Z"
    mock_model.deployed_models = []
    
    # Get model information
    model_info = get_model_info(mock_model)
    
    # Check model information
    assert model_info["resource_name"] == "projects/test-project/models/test-model"
    assert model_info["display_name"] == "test-model"
    assert model_info["description"] == "Test model"
    assert model_info["version_id"] == "1"
    assert model_info["version_aliases"] == ["latest"]
    assert model_info["create_time"] == "2023-01-01T00:00:00Z"
    assert model_info["update_time"] == "2023-01-01T00:00:00Z"
    assert model_info["deployed_models"] == []


def test_get_endpoint_info(mock_vertex_client):
    """Test getting endpoint information."""
    _, mock_endpoint = mock_vertex_client
    
    # Set endpoint attributes
    mock_endpoint.resource_name = "projects/test-project/endpoints/test-endpoint"
    mock_endpoint.display_name = "test-endpoint"
    mock_endpoint.description = "Test endpoint"
    mock_endpoint.deployed_models = []
    mock_endpoint.traffic_split = {}
    mock_endpoint.create_time = "2023-01-01T00:00:00Z"
    mock_endpoint.update_time = "2023-01-01T00:00:00Z"
    
    # Get endpoint information
    endpoint_info = get_endpoint_info(mock_endpoint)
    
    # Check endpoint information
    assert endpoint_info["resource_name"] == "projects/test-project/endpoints/test-endpoint"
    assert endpoint_info["display_name"] == "test-endpoint"
    assert endpoint_info["description"] == "Test endpoint"
    assert endpoint_info["deployed_models"] == []
    assert endpoint_info["traffic_split"] == {}
    assert endpoint_info["create_time"] == "2023-01-01T00:00:00Z"
    assert endpoint_info["update_time"] == "2023-01-01T00:00:00Z"


def test_api_config():
    """Test API configuration."""
    # Create API configuration
    config = APIConfig()
    
    # Check default values
    assert config.endpoint_id == ""
    assert config.project_id == ""
    assert config.location == "us-central1"
    assert config.api_key == ""
    assert config.log_predictions is True
    assert config.enable_monitoring is True
    
    # Create API configuration with values
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
        config_data = {
            "endpoint_id": "test-endpoint",
            "project_id": "test-project",
            "location": "us-central1",
            "api_key": "test-api-key",
            "log_predictions": False,
            "enable_monitoring": False,
        }
        
        yaml.dump(config_data, temp_file)
        
        config = APIConfig(temp_file.name)
        
        # Check values
        assert config.endpoint_id == "test-endpoint"
        assert config.project_id == "test-project"
        assert config.location == "us-central1"
        assert config.api_key == "test-api-key"
        assert config.log_predictions is False
        assert config.enable_monitoring is False
        
        # Clean up
        os.unlink(temp_file.name)


def test_prediction_request():
    """Test prediction request model."""
    # Create prediction request
    request = PredictionRequest(
        inputs=[{"text": "test"}],
        parameters={"temperature": 0.5},
    )
    
    # Check values
    assert request.inputs == [{"text": "test"}]
    assert request.parameters == {"temperature": 0.5}
    
    # Convert to dictionary
    request_dict = request.dict()
    
    # Check dictionary
    assert request_dict["inputs"] == [{"text": "test"}]
    assert request_dict["parameters"] == {"temperature": 0.5}


def test_prediction_response():
    """Test prediction response model."""
    # Create prediction response
    response = PredictionResponse(
        predictions=[{"text": "test"}],
        model_info={"model_id": "test-model"},
    )
    
    # Check values
    assert response.predictions == [{"text": "test"}]
    assert response.model_info == {"model_id": "test-model"}
    
    # Convert to dictionary
    response_dict = response.dict()
    
    # Check dictionary
    assert response_dict["predictions"] == [{"text": "test"}]
    assert response_dict["model_info"] == {"model_id": "test-model"}


def test_monitoring_client(mock_monitoring_client):
    """Test monitoring client."""
    # Create monitoring client
    client = MonitoringClient(
        project_id="test-project",
        location="global",
    )
    
    # Check client
    assert client.project_id == "test-project"
    assert client.location == "global"
    assert client.project_path == "projects/test-project"


def test_logging_client(mock_logging_client):
    """Test logging client."""
    # Create logging client
    client = LoggingClient(
        project_id="test-project",
        log_name="test-log",
        location="global",
    )
    
    # Check client
    assert client.project_id == "test-project"
    assert client.log_name == "test-log"
    assert client.location == "global"
    assert client.log_path == "projects/test-project/logs/test-log"


def test_model_monitor(mock_monitoring_client, mock_logging_client):
    """Test model monitor."""
    # Create model monitor
    monitor = ModelMonitor(
        project_id="test-project",
        model_id="test-model",
        endpoint_id="test-endpoint",
        location="global",
    )
    
    # Check monitor
    assert monitor.project_id == "test-project"
    assert monitor.model_id == "test-model"
    assert monitor.endpoint_id == "test-endpoint"
    assert monitor.location == "global"


def test_upload_model_artifacts(mock_storage_client, model_dir):
    """Test uploading model artifacts."""
    # Mock bucket and blob
    mock_bucket = mock.MagicMock()
    mock_blob = mock.MagicMock()
    
    # Set up mock client
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    # Upload model artifacts
    uri = upload_model_artifacts(
        model_dir=model_dir,
        bucket_name="test-bucket",
        destination_blob_name="test-model",
    )
    
    # Check URI
    assert uri == "gs://test-bucket/test-model"
    
    # Check bucket call
    mock_storage_client.return_value.bucket.assert_called_once_with("test-bucket")
    
    # Check blob calls
    assert mock_bucket.blob.call_count == 3
    
    # Check upload calls
    assert mock_blob.upload_from_filename.call_count == 3


def test_create_deployment_config(model_dir, config_file):
    """Test creating deployment configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create deployment configuration
        config = create_deployment_config(
            model_dir=model_dir,
            config_path=config_file,
            output_path=Path(temp_dir) / "deployment_config.yaml",
            model_name="test-model",
            endpoint_name="test-endpoint",
            serving_container_image_uri="test-image",
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=1,
        )
        
        # Check configuration
        assert config["gcp"]["project_id"] == "test-project"
        assert config["gcp"]["region"] == "us-central1"
        assert config["gcp"]["storage"]["bucket_name"] == "test-bucket"
        assert config["deployment"]["model_name"] == "test-model"
        assert config["deployment"]["endpoint_name"] == "test-endpoint"
        assert config["deployment"]["serving_container_image_uri"] == "test-image"
        assert config["deployment"]["machine_type"] == "n1-standard-4"
        assert config["deployment"]["min_replicas"] == 1
        assert config["deployment"]["max_replicas"] == 1
        assert config["monitoring"]["enable_monitoring"] is True
        assert config["monitoring"]["enable_logging"] is True
        
        # Check output file
        output_path = Path(temp_dir) / "deployment_config.yaml"
        assert output_path.exists()
        
        # Load output file
        with open(output_path, "r") as f:
            output_config = yaml.safe_load(f)
        
        # Check output configuration
        assert output_config["gcp"]["project_id"] == "test-project"
        assert output_config["gcp"]["region"] == "us-central1"
        assert output_config["gcp"]["storage"]["bucket_name"] == "test-bucket"
        assert output_config["deployment"]["model_name"] == "test-model"
        assert output_config["deployment"]["endpoint_name"] == "test-endpoint"
        assert output_config["deployment"]["serving_container_image_uri"] == "test-image"
        assert output_config["deployment"]["machine_type"] == "n1-standard-4"
        assert output_config["deployment"]["min_replicas"] == 1
        assert output_config["deployment"]["max_replicas"] == 1
        assert output_config["monitoring"]["enable_monitoring"] is True
        assert output_config["monitoring"]["enable_logging"] is True


def test_create_deployment_script(model_dir, config_file):
    """Test creating deployment script."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create deployment script
        script_path = create_deployment_script(
            model_dir=model_dir,
            config_path=config_file,
            output_path=Path(temp_dir) / "deploy.sh",
        )
        
        # Check script path
        assert script_path == str(Path(temp_dir) / "deploy.sh")
        
        # Check script file
        script_path = Path(temp_dir) / "deploy.sh"
        assert script_path.exists()
        
        # Check script permissions
        assert os.access(script_path, os.X_OK)
        
        # Check script content
        with open(script_path, "r") as f:
            script_content = f.read()
        
        assert "#!/bin/bash" in script_content
        assert "export PROJECT_ID=\"test-project\"" in script_content
        assert "export REGION=\"us-central1\"" in script_content
        assert "export BUCKET_NAME=\"test-bucket\"" in script_content
        assert "export MODEL_NAME=\"test-model\"" in script_content
        assert "export ENDPOINT_NAME=\"test-endpoint\"" in script_content
        assert "gsutil -m cp -r $MODEL_DIR/* gs://$BUCKET_NAME/$MODEL_NAME/" in script_content
        assert "gcloud ai models upload" in script_content
        assert "gcloud ai endpoints create" in script_content
        assert "gcloud ai endpoints deploy-model" in script_content


def test_create_deployment_pipeline(model_dir, config_file):
    """Test creating deployment pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create deployment pipeline
        pipeline_info = create_deployment_pipeline(
            model_dir=model_dir,
            config_path=config_file,
            output_dir=Path(temp_dir) / "deployment",
        )
        
        # Check pipeline information
        assert pipeline_info["model_dir"] == str(model_dir)
        assert pipeline_info["config_path"] == str(Path(temp_dir) / "deployment" / "deployment_config.yaml")
        assert pipeline_info["script_path"] == str(Path(temp_dir) / "deployment" / "deploy.sh")
        
        # Check output directory
        output_dir = Path(temp_dir) / "deployment"
        assert output_dir.exists()
        
        # Check output files
        assert (output_dir / "deployment_config.yaml").exists()
        assert (output_dir / "deploy.sh").exists()
        assert (output_dir / "pipeline_info.json").exists()
        
        # Check pipeline information file
        with open(output_dir / "pipeline_info.json", "r") as f:
            pipeline_info_file = json.load(f)
        
        assert pipeline_info_file["model_dir"] == str(model_dir)
        assert pipeline_info_file["config_path"] == str(Path(temp_dir) / "deployment" / "deployment_config.yaml")
        assert pipeline_info_file["script_path"] == str(Path(temp_dir) / "deployment" / "deploy.sh")
