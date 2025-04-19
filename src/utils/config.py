"""
Configuration utilities.

This module provides utilities for loading and validating configuration.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ProjectConfig(BaseModel):
    """Project configuration model."""
    
    name: str
    version: str
    description: str


class GCPConfig(BaseModel):
    """Google Cloud Platform configuration model."""
    
    project_id: str
    region: str
    zone: str
    storage: Dict[str, str]


class PDFProcessingConfig(BaseModel):
    """PDF processing configuration model."""
    
    extraction_method: str
    document_ai: Optional[Dict[str, str]] = None
    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_length: int = 100


class TrainingDataConfig(BaseModel):
    """Training data configuration model."""
    
    format: str = "jsonl"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    examples_per_document: int = 10
    min_input_length: int = 10
    max_input_length: int = 1024
    min_output_length: int = 20
    max_output_length: int = 2048


class PEFTConfig(BaseModel):
    """Parameter-Efficient Fine-Tuning configuration model."""
    
    method: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1


class TrainingConfig(BaseModel):
    """Training configuration model."""
    
    batch_size: int = 8
    learning_rate: float = 1.0e-5
    epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True


class FineTuningConfig(BaseModel):
    """Fine-tuning configuration model."""
    
    model: str
    method: str = "peft"
    peft: PEFTConfig = PEFTConfig()
    training: TrainingConfig = TrainingConfig()


class EvaluationConfig(BaseModel):
    """Evaluation configuration model."""
    
    metrics: list
    human_evaluation: Dict[str, Any]


class DeploymentConfig(BaseModel):
    """Deployment configuration model."""
    
    endpoint_name: str
    machine_type: str
    min_replicas: int = 1
    max_replicas: int = 5
    accelerator_type: Optional[str] = None
    accelerator_count: int = 0


class Config(BaseModel):
    """Complete configuration model."""
    
    project: ProjectConfig
    gcp: GCPConfig
    pdf_processing: PDFProcessingConfig
    training_data: TrainingDataConfig
    fine_tuning: FineTuningConfig
    evaluation: EvaluationConfig
    deployment: DeploymentConfig


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Config: Validated configuration object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValidationError: If the configuration is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = Config(**config_dict)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except ValidationError as e:
        logger.error(f"Invalid configuration: {str(e)}")
        raise


def get_default_config_path() -> Path:
    """
    Get the default configuration path.

    Returns:
        Path: Default configuration path.
    """
    # Look for config in the current directory
    if Path("config/config.yaml").exists():
        return Path("config/config.yaml")
    
    # Look for config in the parent directory
    if Path("../config/config.yaml").exists():
        return Path("../config/config.yaml")
    
    # Look for config in the package directory
    package_dir = Path(__file__).parent.parent.parent
    config_path = package_dir / "config" / "config.yaml"
    if config_path.exists():
        return config_path
    
    raise FileNotFoundError("Default configuration file not found")


def validate_gcp_config(config: GCPConfig) -> bool:
    """
    Validate Google Cloud Platform configuration.

    Args:
        config: GCP configuration to validate.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    # Check if project ID is set
    if config.project_id == "your-project-id":
        logger.warning("GCP project ID is not set. Please update the configuration.")
        return False
    
    # Check if storage bucket is set
    if "bucket_name" in config.storage and config.storage["bucket_name"] == "gemini-pdf-finetuning-data":
        logger.warning("GCP storage bucket name is using the default value. Please update the configuration.")
        return False
    
    return True


def validate_document_ai_config(config: PDFProcessingConfig) -> bool:
    """
    Validate Document AI configuration.

    Args:
        config: PDF processing configuration to validate.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    if config.extraction_method == "document_ai":
        if not config.document_ai:
            logger.warning("Document AI is selected as extraction method, but no Document AI configuration is provided.")
            return False
        
        if "processor_id" not in config.document_ai or config.document_ai["processor_id"] == "your-processor-id":
            logger.warning("Document AI processor ID is not set. Please update the configuration.")
            return False
    
    return True


def validate_config(config: Config) -> bool:
    """
    Validate the complete configuration.

    Args:
        config: Configuration to validate.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    valid = True
    
    # Validate GCP configuration
    if not validate_gcp_config(config.gcp):
        valid = False
    
    # Validate Document AI configuration
    if not validate_document_ai_config(config.pdf_processing):
        valid = False
    
    # Validate training data splits
    splits_sum = config.training_data.train_split + config.training_data.val_split + config.training_data.test_split
    if abs(splits_sum - 1.0) > 0.001:
        logger.warning(f"Training data splits do not sum to 1.0: {splits_sum}")
        valid = False
    
    return valid
