"""
Tests for the Vertex AI Pipeline module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml
from kfp.v2 import compiler

from src.fine_tuning.config import FineTuningConfig
from src.fine_tuning.pipeline import (
    process_pdfs_component,
    prepare_data_component,
    optimize_hyperparameters_component,
    finetune_component,
    evaluate_component,
    deploy_component,
    create_pipeline,
    run_pipeline,
    create_pipeline_trigger,
)


@pytest.fixture
def sample_config():
    """Sample fine-tuning configuration."""
    return {
        "project": {
            "name": "gemini-pdf-finetuning",
            "version": "0.1.0",
        },
        "gcp": {
            "project_id": "test-project",
            "region": "us-central1",
            "storage": {
                "bucket_name": "test-bucket",
                "training_data_folder": "training_data",
            },
        },
        "pdf_processing": {
            "extraction_method": "pypdf2",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "training_data": {
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "min_quality_score": 0.7,
        },
        "fine_tuning": {
            "model_name": "gemini-pro",
            "method": "peft",
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-5,
                "epochs": 3,
            },
            "vertex_ai": {
                "project_id": "test-project",
                "region": "us-central1",
            },
        },
        "evaluation": {
            "metrics": ["accuracy", "f1", "rouge"],
        },
        "deployment": {
            "endpoint_name": "gemini-pdf-finetuned",
            "min_replicas": 1,
            "max_replicas": 5,
        },
    }


@pytest.fixture
def config_file(sample_config, tmp_path):
    """Create a sample configuration file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def mock_aiplatform():
    """Mock Google Cloud Vertex AI."""
    with mock.patch("google.cloud.aiplatform") as mock_ai:
        # Configure mock
        mock_job = mock.MagicMock()
        mock_job.display_name = "test-job"
        mock_ai.PipelineJob.return_value = mock_job
        
        yield mock_ai


def test_process_pdfs_component():
    """Test process PDFs component."""
    # Create component
    component = process_pdfs_component
    
    # Check component
    assert component.name == "process_pdfs_component"
    assert "pdf_dir" in component.inputs
    assert "output_dir" in component.inputs
    assert "config_path" in component.inputs
    assert "method" in component.inputs
    assert "extract_metadata" in component.inputs
    assert component.output.type == "String"


def test_prepare_data_component():
    """Test prepare data component."""
    # Create component
    component = prepare_data_component
    
    # Check component
    assert component.name == "prepare_data_component"
    assert "text_dir" in component.inputs
    assert "output_dir" in component.inputs
    assert "config_path" in component.inputs
    assert "export_to_gcs" in component.inputs
    assert "generate_quality_report" in component.inputs
    assert component.output.type == "String"


def test_optimize_hyperparameters_component():
    """Test optimize hyperparameters component."""
    # Create component
    component = optimize_hyperparameters_component
    
    # Check component
    assert component.name == "optimize_hyperparameters_component"
    assert "data_dir" in component.inputs
    assert "output_dir" in component.inputs
    assert "config_path" in component.inputs
    assert "n_trials" in component.inputs
    assert "timeout" in component.inputs
    assert component.output.type == "String"


def test_finetune_component():
    """Test finetune component."""
    # Create component
    component = finetune_component
    
    # Check component
    assert component.name == "finetune_component"
    assert "data_dir" in component.inputs
    assert "output_dir" in component.inputs
    assert "config_path" in component.inputs
    assert "method" in component.inputs
    assert "model_name" in component.inputs
    assert "use_best_params" in component.inputs
    assert "hpo_dir" in component.inputs
    assert component.output.type == "String"


def test_evaluate_component():
    """Test evaluate component."""
    # Create component
    component = evaluate_component
    
    # Check component
    assert component.name == "evaluate_component"
    assert "model_dir" in component.inputs
    assert "test_data" in component.inputs
    assert "output_dir" in component.inputs
    assert "config_path" in component.inputs
    assert component.output.type == "String"


def test_deploy_component():
    """Test deploy component."""
    # Create component
    component = deploy_component
    
    # Check component
    assert component.name == "deploy_component"
    assert "model_dir" in component.inputs
    assert "config_path" in component.inputs
    assert component.output.type == "String"


def test_create_pipeline(config_file, tmp_path):
    """Test creating a pipeline."""
    # Mock compiler
    with mock.patch("kfp.v2.compiler.Compiler.compile") as mock_compile:
        # Create pipeline
        pipeline_path = create_pipeline(
            pipeline_name="test-pipeline",
            config_path=config_file,
            output_dir=tmp_path,
        )
        
        # Check that compiler was called
        mock_compile.assert_called_once()
        
        # Check that pipeline path was returned
        assert pipeline_path == str(tmp_path / "test-pipeline.json")


def test_run_pipeline(config_file, tmp_path, mock_aiplatform):
    """Test running a pipeline."""
    # Create pipeline file
    pipeline_path = tmp_path / "test-pipeline.json"
    with open(pipeline_path, "w") as f:
        f.write("{}")
    
    # Run pipeline
    job_name = run_pipeline(
        pipeline_path=pipeline_path,
        pdf_dir="pdf_dir",
        output_dir="output_dir",
        config_path=config_file,
        project_id="test-project",
        region="us-central1",
    )
    
    # Check that Vertex AI was initialized
    mock_aiplatform.init.assert_called_once_with(
        project="test-project",
        location="us-central1",
    )
    
    # Check that pipeline job was created
    mock_aiplatform.PipelineJob.assert_called_once()
    
    # Check that pipeline job was submitted
    mock_aiplatform.PipelineJob.return_value.submit.assert_called_once()
    
    # Check that job name was returned
    assert job_name == "test-job"


def test_create_pipeline_trigger(config_file, tmp_path, mock_aiplatform):
    """Test creating a pipeline trigger."""
    # Create pipeline file
    pipeline_path = tmp_path / "test-pipeline.json"
    with open(pipeline_path, "w") as f:
        f.write("{}")
    
    # Create trigger
    trigger_name = create_pipeline_trigger(
        pipeline_path=pipeline_path,
        trigger_name="test-trigger",
        schedule="0 0 * * *",
        pdf_dir="pdf_dir",
        output_dir="output_dir",
        config_path=config_file,
        project_id="test-project",
        region="us-central1",
    )
    
    # Check that Vertex AI was initialized
    mock_aiplatform.init.assert_called_once_with(
        project="test-project",
        location="us-central1",
    )
    
    # Check that pipeline trigger was created
    mock_aiplatform.PipelineJob.create_schedule.assert_called_once()
    
    # Check that trigger name was returned
    assert trigger_name == "test-trigger"


def test_create_pipeline_with_all_options(config_file, tmp_path):
    """Test creating a pipeline with all options."""
    # Mock compiler
    with mock.patch("kfp.v2.compiler.Compiler.compile") as mock_compile:
        # Create pipeline
        pipeline_path = create_pipeline(
            pipeline_name="test-pipeline",
            config_path=config_file,
            output_dir=tmp_path,
            steps=["process", "prepare", "finetune", "evaluate", "deploy"],
            extract_metadata=True,
            generate_quality_report=True,
            export_to_gcs=True,
            optimize_hyperparams=True,
            n_trials=10,
        )
        
        # Check that compiler was called
        mock_compile.assert_called_once()
        
        # Check that pipeline path was returned
        assert pipeline_path == str(tmp_path / "test-pipeline.json")


def test_run_pipeline_with_service_account(config_file, tmp_path, mock_aiplatform):
    """Test running a pipeline with a service account."""
    # Create pipeline file
    pipeline_path = tmp_path / "test-pipeline.json"
    with open(pipeline_path, "w") as f:
        f.write("{}")
    
    # Run pipeline
    job_name = run_pipeline(
        pipeline_path=pipeline_path,
        pdf_dir="pdf_dir",
        output_dir="output_dir",
        config_path=config_file,
        project_id="test-project",
        region="us-central1",
        service_account="test-service-account",
    )
    
    # Check that Vertex AI was initialized
    mock_aiplatform.init.assert_called_once_with(
        project="test-project",
        location="us-central1",
    )
    
    # Check that pipeline job was created
    mock_aiplatform.PipelineJob.assert_called_once()
    
    # Check that pipeline job was submitted with service account
    mock_aiplatform.PipelineJob.return_value.submit.assert_called_once_with(
        service_account="test-service-account"
    )
    
    # Check that job name was returned
    assert job_name == "test-job"


def test_create_pipeline_trigger_with_service_account(config_file, tmp_path, mock_aiplatform):
    """Test creating a pipeline trigger with a service account."""
    # Create pipeline file
    pipeline_path = tmp_path / "test-pipeline.json"
    with open(pipeline_path, "w") as f:
        f.write("{}")
    
    # Create trigger
    trigger_name = create_pipeline_trigger(
        pipeline_path=pipeline_path,
        trigger_name="test-trigger",
        schedule="0 0 * * *",
        pdf_dir="pdf_dir",
        output_dir="output_dir",
        config_path=config_file,
        project_id="test-project",
        region="us-central1",
        service_account="test-service-account",
    )
    
    # Check that Vertex AI was initialized
    mock_aiplatform.init.assert_called_once_with(
        project="test-project",
        location="us-central1",
    )
    
    # Check that pipeline trigger was created with service account
    mock_aiplatform.PipelineJob.create_schedule.assert_called_once_with(
        display_name="test-trigger",
        template_path=pipeline_path,
        pipeline_root="output_dir",
        parameter_values={
            "pdf_dir": "pdf_dir",
            "output_dir": "output_dir",
            "config_path": config_file,
        },
        schedule="0 0 * * *",
        service_account="test-service-account",
    )
    
    # Check that trigger name was returned
    assert trigger_name == "test-trigger"
