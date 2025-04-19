"""
Tests for the fine-tuning configuration module.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.fine_tuning.config import (
    FineTuningConfig,
    LoRAConfig,
    TrainingConfig,
    VertexAIConfig,
    load_config,
    create_config_template,
    validate_config,
)


def test_lora_config():
    """Test LoRA configuration."""
    # Create default configuration
    config = LoRAConfig()
    
    # Check default values
    assert config.r == 16
    assert config.alpha == 32
    assert config.dropout == 0.1
    assert config.target_modules == ["q_proj", "v_proj"]
    assert config.bias == "none"
    assert config.task_type == "CAUSAL_LM"
    
    # Create custom configuration
    config = LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.2,
        target_modules=["q_proj", "k_proj", "v_proj"],
        bias="all",
        task_type="SEQ_2_SEQ_LM",
    )
    
    # Check custom values
    assert config.r == 8
    assert config.alpha == 16
    assert config.dropout == 0.2
    assert config.target_modules == ["q_proj", "k_proj", "v_proj"]
    assert config.bias == "all"
    assert config.task_type == "SEQ_2_SEQ_LM"


def test_training_config():
    """Test training configuration."""
    # Create default configuration
    config = TrainingConfig()
    
    # Check default values
    assert config.batch_size == 8
    assert config.learning_rate == 1e-5
    assert config.epochs == 3
    assert config.warmup_steps == 100
    assert config.weight_decay == 0.01
    assert config.gradient_accumulation_steps == 4
    assert config.max_grad_norm == 1.0
    assert config.fp16 is True
    assert config.logging_steps == 10
    assert config.save_steps == 100
    assert config.eval_steps == 100
    
    # Create custom configuration
    config = TrainingConfig(
        batch_size=16,
        learning_rate=2e-5,
        epochs=5,
        warmup_steps=200,
        weight_decay=0.1,
        gradient_accumulation_steps=8,
        max_grad_norm=2.0,
        fp16=False,
        logging_steps=20,
        save_steps=200,
        eval_steps=200,
    )
    
    # Check custom values
    assert config.batch_size == 16
    assert config.learning_rate == 2e-5
    assert config.epochs == 5
    assert config.warmup_steps == 200
    assert config.weight_decay == 0.1
    assert config.gradient_accumulation_steps == 8
    assert config.max_grad_norm == 2.0
    assert config.fp16 is False
    assert config.logging_steps == 20
    assert config.save_steps == 200
    assert config.eval_steps == 200


def test_vertex_ai_config():
    """Test Vertex AI configuration."""
    # Create configuration
    config = VertexAIConfig(
        project_id="test-project",
    )
    
    # Check default values
    assert config.project_id == "test-project"
    assert config.region == "us-central1"
    assert config.machine_type == "n1-standard-8"
    assert config.accelerator_type == "NVIDIA_TESLA_V100"
    assert config.accelerator_count == 1
    assert config.service_account is None
    assert config.container_uri == "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest"
    
    # Create custom configuration
    config = VertexAIConfig(
        project_id="test-project",
        region="us-west1",
        machine_type="n1-standard-16",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=2,
        service_account="test-service-account",
        container_uri="test-container-uri",
    )
    
    # Check custom values
    assert config.project_id == "test-project"
    assert config.region == "us-west1"
    assert config.machine_type == "n1-standard-16"
    assert config.accelerator_type == "NVIDIA_TESLA_T4"
    assert config.accelerator_count == 2
    assert config.service_account == "test-service-account"
    assert config.container_uri == "test-container-uri"


def test_fine_tuning_config():
    """Test fine-tuning configuration."""
    # Create configuration
    config = FineTuningConfig(
        model_name="gemini-pro",
        vertex_ai={"project_id": "test-project"},
    )
    
    # Check default values
    assert config.model_name == "gemini-pro"
    assert config.method == "peft"
    assert isinstance(config.lora, LoRAConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.vertex_ai, VertexAIConfig)
    assert config.vertex_ai.project_id == "test-project"
    assert config.output_dir == "./output"
    assert config.cache_dir is None
    assert config.seed == 42
    
    # Create custom configuration
    config = FineTuningConfig(
        model_name="gemini-1.5-pro",
        method="full",
        lora=LoRAConfig(r=8, alpha=16),
        training=TrainingConfig(batch_size=16, epochs=5),
        vertex_ai=VertexAIConfig(
            project_id="test-project",
            region="us-west1",
        ),
        output_dir="./custom-output",
        cache_dir="./cache",
        seed=123,
    )
    
    # Check custom values
    assert config.model_name == "gemini-1.5-pro"
    assert config.method == "full"
    assert config.lora.r == 8
    assert config.lora.alpha == 16
    assert config.training.batch_size == 16
    assert config.training.epochs == 5
    assert config.vertex_ai.project_id == "test-project"
    assert config.vertex_ai.region == "us-west1"
    assert config.output_dir == "./custom-output"
    assert config.cache_dir == "./cache"
    assert config.seed == 123


def test_create_config_template():
    """Test creating a configuration template."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create template
        output_path = os.path.join(tmp_dir, "config.yaml")
        template_path = create_config_template(output_path)
        
        # Check that template was created
        assert os.path.exists(template_path)
        
        # Load template
        with open(template_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Check template contents
        assert "model_name" in config_dict
        assert "method" in config_dict
        assert "lora" in config_dict
        assert "training" in config_dict
        assert "vertex_ai" in config_dict
        assert "output_dir" in config_dict
        assert "seed" in config_dict
        
        # Check that template can be loaded
        config = FineTuningConfig(**config_dict)
        assert config.model_name == "gemini-pro"
        assert config.vertex_ai.project_id == "your-project-id"


def test_load_config():
    """Test loading configuration from a file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create configuration file
        config_path = os.path.join(tmp_dir, "config.yaml")
        config_dict = {
            "model_name": "gemini-1.5-pro",
            "method": "full",
            "lora": {
                "r": 8,
                "alpha": 16,
            },
            "training": {
                "batch_size": 16,
                "epochs": 5,
            },
            "vertex_ai": {
                "project_id": "test-project",
                "region": "us-west1",
            },
            "output_dir": "./custom-output",
            "seed": 123,
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
        
        # Load configuration
        config = load_config(config_path)
        
        # Check configuration
        assert config.model_name == "gemini-1.5-pro"
        assert config.method == "full"
        assert config.lora.r == 8
        assert config.lora.alpha == 16
        assert config.training.batch_size == 16
        assert config.training.epochs == 5
        assert config.vertex_ai.project_id == "test-project"
        assert config.vertex_ai.region == "us-west1"
        assert config.output_dir == "./custom-output"
        assert config.seed == 123


def test_load_config_with_gcp():
    """Test loading configuration with GCP section."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create configuration file
        config_path = os.path.join(tmp_dir, "config.yaml")
        config_dict = {
            "fine_tuning": {
                "model_name": "gemini-1.5-pro",
                "method": "full",
            },
            "gcp": {
                "project_id": "test-project",
                "region": "us-west1",
            },
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
        
        # Load configuration
        config = load_config(config_path)
        
        # Check configuration
        assert config.model_name == "gemini-1.5-pro"
        assert config.method == "full"
        assert config.vertex_ai.project_id == "test-project"
        assert config.vertex_ai.region == "us-west1"


def test_validate_config():
    """Test validating configuration."""
    # Create valid configuration
    config = FineTuningConfig(
        model_name="gemini-pro",
        vertex_ai=VertexAIConfig(
            project_id="test-project",
        ),
    )
    
    # Validate configuration
    assert validate_config(config) is True
    
    # Create invalid configuration
    config = FineTuningConfig(
        model_name="invalid-model",
        vertex_ai=VertexAIConfig(
            project_id="your-project-id",
        ),
    )
    
    # Validate configuration
    assert validate_config(config) is False


def test_load_config_with_nonexistent_file():
    """Test loading configuration from a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")
