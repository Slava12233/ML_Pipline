"""
Fine-tuning configuration module.

This module provides utilities for configuring the fine-tuning environment.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LoRAConfig(BaseModel):
    """Configuration for LoRA fine-tuning."""
    
    r: int = Field(16, description="LoRA rank")
    alpha: int = Field(32, description="LoRA alpha")
    dropout: float = Field(0.1, description="LoRA dropout")
    target_modules: List[str] = Field(
        ["q_proj", "v_proj"],
        description="Target modules to apply LoRA"
    )
    bias: str = Field("none", description="LoRA bias type")
    task_type: str = Field("CAUSAL_LM", description="Task type")


class TrainingConfig(BaseModel):
    """Configuration for training."""
    
    batch_size: int = Field(8, description="Batch size")
    learning_rate: float = Field(1e-5, description="Learning rate")
    epochs: int = Field(3, description="Number of epochs")
    warmup_steps: int = Field(100, description="Warmup steps")
    weight_decay: float = Field(0.01, description="Weight decay")
    gradient_accumulation_steps: int = Field(4, description="Gradient accumulation steps")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm")
    fp16: bool = Field(True, description="Use mixed precision training")
    logging_steps: int = Field(10, description="Logging steps")
    save_steps: int = Field(100, description="Save steps")
    eval_steps: int = Field(100, description="Evaluation steps")


class VertexAIConfig(BaseModel):
    """Configuration for Vertex AI."""
    
    project_id: str = Field(..., description="GCP project ID")
    region: str = Field("us-central1", description="GCP region")
    machine_type: str = Field("n1-standard-8", description="Machine type")
    accelerator_type: str = Field("NVIDIA_TESLA_V100", description="Accelerator type")
    accelerator_count: int = Field(1, description="Number of accelerators")
    service_account: Optional[str] = Field(None, description="Service account")
    container_uri: str = Field(
        "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest",
        description="Container URI"
    )


class FineTuningConfig(BaseModel):
    """Configuration for fine-tuning."""
    
    model_name: str = Field("gemini-pro", description="Model name")
    method: str = Field("peft", description="Fine-tuning method")
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    vertex_ai: VertexAIConfig = Field(..., description="Vertex AI configuration")
    output_dir: str = Field("./output", description="Output directory")
    cache_dir: Optional[str] = Field(None, description="Cache directory")
    seed: int = Field(42, description="Random seed")


def load_config(config_path: Union[str, Path]) -> FineTuningConfig:
    """
    Load fine-tuning configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        FineTuningConfig: Fine-tuning configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Extract fine-tuning configuration
    if "fine_tuning" in config_dict:
        fine_tuning_dict = config_dict["fine_tuning"]
    else:
        fine_tuning_dict = config_dict
    
    # Extract GCP configuration for Vertex AI
    if "gcp" in config_dict:
        gcp_dict = config_dict["gcp"]
        
        # Create Vertex AI configuration
        if "vertex_ai" not in fine_tuning_dict:
            fine_tuning_dict["vertex_ai"] = {}
        
        # Set project ID and region
        fine_tuning_dict["vertex_ai"]["project_id"] = gcp_dict.get("project_id")
        fine_tuning_dict["vertex_ai"]["region"] = gcp_dict.get("region", "us-central1")
    
    return FineTuningConfig(**fine_tuning_dict)


def create_config_template(output_path: Union[str, Path]) -> str:
    """
    Create a configuration template file.

    Args:
        output_path: Path to save the template.

    Returns:
        str: Path to the created template.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create default configuration
    config = FineTuningConfig(
        model_name="gemini-pro",
        vertex_ai=VertexAIConfig(
            project_id="your-project-id",
        ),
    )
    
    # Convert to dictionary
    config_dict = config.dict()
    
    # Save to YAML file
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created configuration template at {output_path}")
    
    return str(output_path)


def validate_config(config: FineTuningConfig) -> bool:
    """
    Validate fine-tuning configuration.

    Args:
        config: Fine-tuning configuration.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    valid = True
    
    # Check if project ID is set
    if config.vertex_ai.project_id == "your-project-id":
        logger.warning("Project ID is not set. Please update the configuration.")
        valid = False
    
    # Check if model name is valid
    valid_models = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
    if config.model_name not in valid_models:
        logger.warning(f"Model name {config.model_name} is not in the list of valid models: {valid_models}")
        valid = False
    
    # Check if fine-tuning method is valid
    valid_methods = ["peft", "full"]
    if config.method not in valid_methods:
        logger.warning(f"Fine-tuning method {config.method} is not in the list of valid methods: {valid_methods}")
        valid = False
    
    # Check if LoRA configuration is valid
    if config.method == "peft":
        if config.lora.r <= 0:
            logger.warning(f"LoRA rank must be positive, got {config.lora.r}")
            valid = False
        
        if config.lora.alpha <= 0:
            logger.warning(f"LoRA alpha must be positive, got {config.lora.alpha}")
            valid = False
        
        if not 0 <= config.lora.dropout < 1:
            logger.warning(f"LoRA dropout must be between 0 and 1, got {config.lora.dropout}")
            valid = False
    
    # Check if training configuration is valid
    if config.training.batch_size <= 0:
        logger.warning(f"Batch size must be positive, got {config.training.batch_size}")
        valid = False
    
    if config.training.learning_rate <= 0:
        logger.warning(f"Learning rate must be positive, got {config.training.learning_rate}")
        valid = False
    
    if config.training.epochs <= 0:
        logger.warning(f"Number of epochs must be positive, got {config.training.epochs}")
        valid = False
    
    # Check if Vertex AI configuration is valid
    valid_accelerators = ["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100", "NVIDIA_TESLA_P100", "NVIDIA_TESLA_P4", "NVIDIA_TESLA_K80"]
    if config.vertex_ai.accelerator_type not in valid_accelerators:
        logger.warning(f"Accelerator type {config.vertex_ai.accelerator_type} is not in the list of valid accelerators: {valid_accelerators}")
        valid = False
    
    if config.vertex_ai.accelerator_count <= 0:
        logger.warning(f"Number of accelerators must be positive, got {config.vertex_ai.accelerator_count}")
        valid = False
    
    return valid


def setup_vertex_ai_environment(config: FineTuningConfig) -> Dict:
    """
    Set up Vertex AI environment.

    Args:
        config: Fine-tuning configuration.

    Returns:
        Dict: Vertex AI environment configuration.
    """
    try:
        from google.cloud import aiplatform
        
        # Initialize Vertex AI
        aiplatform.init(
            project=config.vertex_ai.project_id,
            location=config.vertex_ai.region,
        )
        
        logger.info(f"Initialized Vertex AI for project {config.vertex_ai.project_id} in {config.vertex_ai.region}")
        
        # Return environment configuration
        return {
            "project_id": config.vertex_ai.project_id,
            "region": config.vertex_ai.region,
            "machine_type": config.vertex_ai.machine_type,
            "accelerator_type": config.vertex_ai.accelerator_type,
            "accelerator_count": config.vertex_ai.accelerator_count,
            "service_account": config.vertex_ai.service_account,
            "container_uri": config.vertex_ai.container_uri,
        }
    
    except ImportError:
        logger.error("google-cloud-aiplatform package not installed.")
        raise
    except Exception as e:
        logger.error(f"Error setting up Vertex AI environment: {str(e)}")
        raise
