#!/usr/bin/env python3
"""
Configuration management utilities for fine-tuning pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Define configuration models for type safety
class PdfProcessingConfig(BaseModel):
    extraction_method: str = "pypdf2"
    document_ai: Dict[str, str] = Field(default_factory=lambda: {"processor_id": "your-processor-id"})
    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_length: int = 100

class TrainingDataConfig(BaseModel):
    format: str = "jsonl"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    examples_per_document: int = 10
    min_input_length: int = 10
    max_input_length: int = 1024
    min_output_length: int = 20
    max_output_length: int = 2048

class PeftConfig(BaseModel):
    method: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = Field(default_factory=lambda: ["c_attn", "c_proj"])

class TrainingConfig(BaseModel):
    batch_size: int = 8
    learning_rate: float = 1.0e-5
    epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True

class VertexAIConfig(BaseModel):
    project_id: str = "your-project-id"
    region: str = "us-central1"
    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_V100"
    accelerator_count: int = 1
    service_account: Optional[str] = None
    container_uri: str = "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest"

class FineTuningConfig(BaseModel):
    model: str = "gpt2"
    method: str = "peft"
    peft: PeftConfig = Field(default_factory=PeftConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    vertex_ai: VertexAIConfig = Field(default_factory=VertexAIConfig)

class StorageConfig(BaseModel):
    bucket_name: str = "your-bucket-name"
    pdf_folder: str = "pdfs"
    extracted_text_folder: str = "extracted_text"
    training_data_folder: str = "training_data"
    model_folder: str = "models"

class GcpConfig(BaseModel):
    project_id: str = "your-project-id"
    region: str = "us-central1"
    zone: str = "us-central1-a"
    storage: StorageConfig = Field(default_factory=StorageConfig)

class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(default_factory=lambda: ["rouge", "bleu", "bertscore"])
    human_evaluation: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "num_examples": 50,
            "criteria": ["relevance", "factuality", "coherence", "helpfulness"]
        }
    )

class DeploymentConfig(BaseModel):
    endpoint_name: str = "model-endpoint"
    machine_type: str = "n1-standard-4"
    min_replicas: int = 1
    max_replicas: int = 5
    accelerator_type: Optional[str] = None
    accelerator_count: int = 0

class ProjectConfig(BaseModel):
    name: str = "fine-tuning-pipeline"
    version: str = "0.1.0"
    description: str = "Pipeline for fine-tuning models on PDF documentation"

class PipelineConfig(BaseModel):
    """Full configuration model for the pipeline."""
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    gcp: GcpConfig = Field(default_factory=GcpConfig)
    pdf_processing: PdfProcessingConfig = Field(default_factory=PdfProcessingConfig)
    training_data: TrainingDataConfig = Field(default_factory=TrainingDataConfig)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    environment: str = "local"  # Can be 'local', 'vertex', 'production', etc.

def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries recursively.
    
    Args:
        base_dict: Base dictionary
        override_dict: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def load_config(
    base_config_path: Union[str, Path], 
    env_config_path: Optional[Union[str, Path]] = None,
    env: Optional[str] = None,
    override_values: Optional[Dict[str, Any]] = None
) -> PipelineConfig:
    """
    Load configuration from YAML files with environment-specific overrides.
    
    Args:
        base_config_path: Path to base configuration file
        env_config_path: Path to environment-specific configuration file (optional)
        env: Environment name (local, vertex, production, etc.)
        override_values: Additional values to override configuration (optional)
        
    Returns:
        Pipeline configuration object
    """
    # Convert paths to Path objects
    base_config_path = Path(base_config_path)
    
    # Load base configuration
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    with open(base_config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Default environment from config or parameter
    config_env = config_dict.get("environment", "local")
    env = env or config_env
    
    # Determine environment config path if not provided
    if env_config_path is None and env != "local":
        # Check if {env}_config.yaml exists in the same directory
        potential_env_path = base_config_path.parent / f"{env}_config.yaml"
        if potential_env_path.exists():
            env_config_path = potential_env_path
    
    # Load environment-specific configuration if available
    if env_config_path:
        env_config_path = Path(env_config_path)
        if env_config_path.exists():
            with open(env_config_path, "r") as f:
                env_config = yaml.safe_load(f)
            
            # Merge configurations
            config_dict = deep_merge(config_dict, env_config)
            logger.info(f"Loaded environment configuration for '{env}' from {env_config_path}")
    
    # Apply any additional override values
    if override_values:
        config_dict = deep_merge(config_dict, override_values)
    
    # Store the environment in the config
    config_dict["environment"] = env
    
    # Create configuration object
    try:
        config = PipelineConfig(**config_dict)
        return config
    except Exception as e:
        logger.error(f"Error parsing configuration: {str(e)}")
        raise

def save_config(config: PipelineConfig, output_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Pipeline configuration object
        output_path: Path to save the configuration
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Convert to dictionary
    config_dict = config.model_dump()
    
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {output_path}")

def create_default_configs(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Create default configuration files for different environments.
    
    Args:
        output_dir: Directory to save configuration files
        
    Returns:
        Dictionary mapping environment names to configuration file paths
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base config
    base_config = PipelineConfig()
    base_path = output_dir / "config.yaml"
    save_config(base_config, base_path)
    
    # Create local config (same as base)
    local_config = PipelineConfig(environment="local")
    local_path = output_dir / "local_config.yaml"
    save_config(local_config, local_path)
    
    # Create vertex config with specific overrides
    vertex_config_dict = {
        "environment": "vertex",
        "fine_tuning": {
            "vertex_ai": {
                "project_id": base_config.gcp.project_id,
                "region": base_config.gcp.region,
            }
        }
    }
    vertex_config = PipelineConfig(**deep_merge(base_config.model_dump(), vertex_config_dict))
    vertex_path = output_dir / "vertex_config.yaml"
    save_config(vertex_config, vertex_path)
    
    # Create production config
    prod_config_dict = {
        "environment": "production",
        "fine_tuning": {
            "training": {
                "epochs": 5,
                "batch_size": 16
            }
        },
        "deployment": {
            "min_replicas": 2,
            "max_replicas": 10
        }
    }
    prod_config = PipelineConfig(**deep_merge(base_config.model_dump(), prod_config_dict))
    prod_path = output_dir / "production_config.yaml"
    save_config(prod_config, prod_path)
    
    return {
        "base": base_path,
        "local": local_path,
        "vertex": vertex_path,
        "production": prod_path
    }

def load_config_from_env_var() -> Optional[PipelineConfig]:
    """
    Load configuration from environment variable.
    
    Returns:
        Pipeline configuration object or None if environment variable not set
    """
    config_path = os.environ.get("PIPELINE_CONFIG_PATH")
    if not config_path:
        return None
    
    try:
        return load_config(config_path)
    except Exception as e:
        logger.error(f"Error loading config from environment variable: {str(e)}")
        return None

def get_config(
    config_path: Optional[Union[str, Path]] = None,
    env: Optional[str] = None,
    override_values: Optional[Dict[str, Any]] = None
) -> PipelineConfig:
    """
    Get configuration from multiple possible sources.
    
    Priority order:
    1. Explicitly provided config_path
    2. PIPELINE_CONFIG_PATH environment variable
    3. Default "config/config.yaml" in the project root
    
    Args:
        config_path: Path to configuration file (optional)
        env: Environment name (optional)
        override_values: Additional values to override configuration (optional)
        
    Returns:
        Pipeline configuration object
    """
    # Try loading from explicit path
    if config_path:
        try:
            return load_config(config_path, env=env, override_values=override_values)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
    
    # Try loading from environment variable
    env_config = load_config_from_env_var()
    if env_config:
        # Apply environment and overrides if provided
        if env or override_values:
            config_dict = env_config.model_dump()
            if env:
                config_dict["environment"] = env
            if override_values:
                config_dict = deep_merge(config_dict, override_values)
            return PipelineConfig(**config_dict)
        return env_config
    
    # Try loading from default location
    default_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if default_path.exists():
        try:
            return load_config(default_path, env=env, override_values=override_values)
        except Exception as e:
            logger.error(f"Error loading config from default path {default_path}: {str(e)}")
    
    # If all else fails, create a default config
    logger.warning("No configuration found. Using default configuration.")
    config = PipelineConfig()
    if env:
        config.environment = env
    if override_values:
        config_dict = deep_merge(config.model_dump(), override_values)
        config = PipelineConfig(**config_dict)
    
    return config
