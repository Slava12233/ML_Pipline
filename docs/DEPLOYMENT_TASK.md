# DEPLOYMENT_TASK.md - Gemini PDF Fine-tuning Deployment Plan

## Overview

This document outlines the specific steps for deploying our fine-tuned model on Vertex AI. The primary focus is on creating a scalable, reliable API endpoint that our applications can use to query the model with new documents.

## Target Timeline
- **Deployment Setup**: 1 day
- **API Integration**: 1 day
- **Testing**: 1 day
- **Monitoring Setup**: 1 day

## Prerequisites

- Fine-tuned model in `data/model` directory
- GCP project with Vertex AI API enabled
- Service account with appropriate permissions
- Updated `config/modified_config.yaml` with deployment settings

## Deployment Tasks

### 1. Deploy Model to Vertex AI Endpoint

#### Task 1.1: Prepare Model for Deployment
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/deployment/vertex.py`
- **Function:** `deploy_model()`
- **Input:** Model directory (`data/model`)
- **Output:** Deployed model endpoint

```python
# Example command to run
python -m src.main deploy data/model --config-path config/modified_config.yaml
```

#### Task 1.2: Update Configuration
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `config/deployment_config.yaml`
- **Task:** Create a deployment-specific configuration file based on `modified_config.yaml`

Ensure the following settings are correctly specified:
```yaml
deployment:
  endpoint_name: gemini-cv-finetuned
  machine_type: n1-standard-4
  min_replicas: 1
  max_replicas: 5
  accelerator_type: null  # Change to NVIDIA_TESLA_T4 if GPU is needed
  accelerator_count: 0
```

### 2. Create API Wrapper

#### Task 2.1: Implement API Client
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/deployment/api_client.py`
- **Function:** `create_client()`
- **Task:** Implement a wrapper around the Vertex AI endpoint for easy querying

```python
# File structure example
def create_client(endpoint_id, project_id, location):
    """Create a client to interact with the deployed model."""
    # Implementation details
    pass

def query_model(client, text, max_tokens=100):
    """Send a query to the deployed model."""
    # Implementation details
    pass
```

#### Task 2.2: Add Authentication
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/deployment/auth.py`
- **Function:** `setup_authentication()`
- **Task:** Implement secure authentication for API access

### 3. Testing

#### Task 3.1: Create Test Suite
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `tests/test_deployment_api.py`
- **Task:** Implement tests for the API wrapper

Test the following scenarios:
- Basic query functionality
- Error handling
- Authentication
- Performance under load

#### Task 3.2: Create Test Script
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `scripts/test_deployment.py`
- **Task:** Create a script to test the deployment end-to-end

### 4. Monitoring & Logging

#### Task 4.1: Implement Monitoring
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/deployment/monitoring.py`
- **Function:** `setup_monitoring()`
- **Task:** Set up monitoring for the deployed endpoint

```python
# File structure example
def setup_monitoring(endpoint_id, project_id, location):
    """Set up monitoring for the deployed endpoint."""
    # Implementation details
    pass

def log_prediction(prediction_input, prediction_output, latency):
    """Log predictions for monitoring."""
    # Implementation details
    pass
```

#### Task 4.2: Implement Logging
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/deployment/logging_client.py`
- **Function:** `setup_logging()`
- **Task:** Implement comprehensive logging for debugging and auditing

### 5. Documentation

#### Task 5.1: Create API Documentation
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `docs/api.md`
- **Task:** Create documentation for the API

#### Task 5.2: Update Deployment Documentation
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `docs/deployment.md`
- **Task:** Document the deployment process and configuration

## Technical Implementation Details

### Model Deployment

The model should be deployed using the Vertex AI Endpoints API. The following code in `src/deployment/vertex.py` should be used as a reference:

```python
def deploy_model(model_dir, config):
    """
    Deploy model to Vertex AI Endpoints.
    
    Args:
        model_dir (str): Directory containing the model
        config (dict): Deployment configuration
        
    Returns:
        str: Endpoint ID
    """
    from google.cloud import aiplatform
    
    # Initialize Vertex AI
    aiplatform.init(
        project=config.gcp.project_id,
        location=config.gcp.region,
    )
    
    # Deploy model
    endpoint = aiplatform.Endpoint.create(
        display_name=config.deployment.endpoint_name,
    )
    
    model = aiplatform.Model.upload(
        display_name=f"{config.deployment.endpoint_name}-model",
        artifact_uri=model_dir,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-cpu.1-13:latest",
    )
    
    # Deploy model to endpoint
    model.deploy(
        endpoint=endpoint,
        machine_type=config.deployment.machine_type,
        min_replica_count=config.deployment.min_replicas,
        max_replica_count=config.deployment.max_replicas,
        accelerator_type=config.deployment.accelerator_type,
        accelerator_count=config.deployment.accelerator_count,
    )
    
    return endpoint.name
```

### API Integration

The API wrapper should support the following functionality:
1. Authentication with service account
2. Sending queries to the model
3. Processing responses
4. Error handling
5. Logging and monitoring

Example usage pattern:

```python
from src.deployment.api_client import create_client, query_model

# Create client
client = create_client(
    endpoint_id="your-endpoint-id",
    project_id="your-project-id",
    location="us-central1",
)

# Query model
response = query_model(
    client=client,
    text="What can you tell me about Slava's experience with AI?",
    max_tokens=200,
)

print(response)
```

## Monitoring Requirements

- **Latency**: Track response time for queries
- **Usage**: Monitor number of requests and tokens generated
- **Error Rate**: Track and alert on errors
- **Resource Utilization**: Monitor CPU, memory, and (if applicable) GPU usage

## Success Criteria

1. Model successfully deployed to Vertex AI
2. API wrapper implemented and tested
3. End-to-end tests passing
4. Monitoring and logging in place
5. Documentation updated

## Notes for Development Team

- All code should follow the project's existing patterns and style guidelines
- Write comprehensive tests for new functionality
- Document any changes to the configuration format
- The deployment should be repeatable and automated where possible
- Ensure proper error handling and resilience in the API wrapper
- Consider cost optimization strategies for the deployment 