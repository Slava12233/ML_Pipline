"""
Pytest fixtures for GCP mocking and test setup.
"""

import os
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Configuration model for the pipeline."""
    project: dict
    gcp: dict
    pdf_processing: dict
    training_data: dict
    fine_tuning: dict
    evaluation: dict
    deployment: dict


@pytest.fixture
def mock_gcp_auth():
    """Mock GCP authentication."""
    with mock.patch("google.auth.default") as mock_auth:
        mock_auth.return_value = (mock.MagicMock(), "test-project-id")
        yield mock_auth


@pytest.fixture
def mock_aiplatform():
    """Mock Google Cloud Vertex AI."""
    # First, patch google.auth.default to avoid authentication errors
    with mock.patch("google.auth.default") as mock_auth:
        # Set up the mock credentials
        mock_creds = mock.MagicMock()
        mock_auth.return_value = (mock_creds, "test-project-id")
        
        # Now patch the aiplatform module
        with mock.patch("google.cloud.aiplatform") as mock_aiplatform:
            # Configure the mock
            mock_aiplatform.init.return_value = None
            
            # Mock PipelineJob
            mock_pipeline_job = mock.MagicMock()
            mock_pipeline_job.display_name = "test-job"
            mock_aiplatform.PipelineJob.return_value = mock_pipeline_job
            
            # Mock submit method
            mock_pipeline_job.submit.return_value = None
            
            # Set up the create_schedule method on the instance, not on the class
            mock_schedule = mock.MagicMock()
            mock_schedule.name = "test-schedule"
            mock_pipeline_job.create_schedule.return_value = mock_schedule

            # Mock create_pipeline_job method to avoid actual API calls
            mock_aiplatform.PipelineServiceClient = mock.MagicMock()
            
            yield mock_aiplatform


@pytest.fixture
def mock_storage():
    """Mock Google Cloud Storage."""
    with mock.patch("google.cloud.storage.Client") as mock_storage:
        # Configure the mock
        mock_client = mock.MagicMock()
        mock_storage.return_value = mock_client
        
        # Mock bucket
        mock_bucket = mock.MagicMock()
        mock_client.bucket.return_value = mock_bucket
        
        # Mock blob
        mock_blob = mock.MagicMock()
        mock_bucket.blob.return_value = mock_blob
        
        yield mock_storage


@pytest.fixture
def mock_document_ai():
    """Mock Google Cloud Document AI."""
    with mock.patch("google.cloud.documentai_v1") as mock_documentai:
        # Configure the mock
        mock_client = mock.MagicMock()
        mock_documentai.DocumentProcessorServiceClient.return_value = mock_client
        
        # Mock process_document
        mock_response = mock.MagicMock()
        mock_response.document.text = "Sample extracted text from Document AI"
        mock_client.process_document.return_value = mock_response
        
        yield mock_documentai


@pytest.fixture
def mock_logging():
    """Mock Google Cloud Logging."""
    with mock.patch("google.cloud.logging_v2.Client") as mock_logging:
        # Configure the mock
        mock_client = mock.MagicMock()
        mock_logging.return_value = mock_client
        
        yield mock_logging


@pytest.fixture
def mock_monitoring():
    """Mock Google Cloud Monitoring."""
    with mock.patch("google.cloud.monitoring_v3.MetricServiceClient") as mock_monitoring:
        # Configure the mock
        mock_client = mock.MagicMock()
        mock_monitoring.return_value = mock_client
        
        yield mock_monitoring


@pytest.fixture
def config_file(tmp_path):
    """Create a sample configuration file."""
    config_data = {
        "project": {
            "name": "test-project",
            "version": "0.1.0",
            "description": "Test project"
        },
        "gcp": {
            "project_id": "test-project-id",
            "region": "us-central1",
            "zone": "us-central1-a",
            "storage": {
                "bucket_name": "test-bucket",
                "pdf_folder": "pdfs",
                "extracted_text_folder": "extracted_text",
                "training_data_folder": "training_data",
                "model_folder": "models"
            }
        },
        "pdf_processing": {
            "extraction_method": "pypdf2",
            "document_ai": {
                "processor_id": "test-processor-id"
            },
            "chunk_size": 1000,
            "overlap": 200,
            "min_chunk_length": 100
        },
        "training_data": {
            "format": "jsonl",
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "examples_per_document": 10,
            "min_input_length": 10,
            "max_input_length": 1024,
            "min_output_length": 20,
            "max_output_length": 2048
        },
        "fine_tuning": {
            "model": "gemini-pro",
            "method": "peft",
            "peft": {
                "method": "lora",
                "r": 16,
                "alpha": 32,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 1.0e-5,
                "epochs": 3,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1.0,
                "fp16": True,
                "warmup_ratio": 0.1  # Added for test_hyperparameter.py
            }
        },
        "evaluation": {
            "metrics": [
                "rouge",
                "bleu",
                "bertscore"
            ],
            "human_evaluation": {
                "enabled": True,
                "num_examples": 50,
                "criteria": [
                    "relevance",
                    "factuality",
                    "coherence",
                    "helpfulness"
                ]
            }
        },
        "deployment": {
            "endpoint_name": "test-endpoint",
            "machine_type": "n1-standard-4",
            "min_replicas": 1,
            "max_replicas": 5,
            "accelerator_type": None,
            "accelerator_count": 0
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    
    return config_path


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a sample data directory with training data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create train data
    train_dir = data_dir / "train"
    train_dir.mkdir()
    
    train_data = [
        {
            "input": "What is the main feature of the product?",
            "output": "The main feature is its ability to process PDFs efficiently."
        },
        {
            "input": "How do I install the software?",
            "output": "You can install the software by running 'pip install package'."
        }
    ]
    
    with open(train_dir / "train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    # Create validation data
    val_dir = data_dir / "val"
    val_dir.mkdir()
    
    val_data = [
        {
            "input": "What are the system requirements?",
            "output": "The system requires Python 3.9+ and 4GB RAM."
        }
    ]
    
    with open(val_dir / "val.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    
    # Create test data
    test_dir = data_dir / "test"
    test_dir.mkdir()
    
    test_data = [
        {
            "input": "How do I upgrade to the latest version?",
            "output": "You can upgrade by running 'pip install --upgrade package'."
        }
    ]
    
    with open(test_dir / "test.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    # Create stats file
    stats = {
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "test_examples": len(test_data),
        "total_examples": len(train_data) + len(val_data) + len(test_data),
        "field_stats": {
            "input_length": {
                "mean": 30.5,
                "min": 20,
                "max": 45,
                "median": 30,
                "std": 8.5
            },
            "output_length": {
                "mean": 45.0,
                "min": 30,
                "max": 60,
                "median": 45,
                "std": 10.5
            }
        }
    }
    
    with open(data_dir / "stats.json", "w") as f:
        json.dump(stats, f)
    
    return data_dir 