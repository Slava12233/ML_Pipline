"""
Fine-tuning training module.

This module provides functions for fine-tuning Gemini models.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.fine_tuning.config import FineTuningConfig, load_config
from src.fine_tuning.setup import setup_training_environment

logger = logging.getLogger(__name__)


def load_training_data(
    data_dir: Union[str, Path],
    tokenizer,
    max_length: Optional[int] = None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load training data.

    Args:
        data_dir: Directory containing training data.
        tokenizer: Tokenizer to use for tokenization.
        max_length: Maximum sequence length.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Training and validation datasets.
    """
    logger.info(f"Loading training data from {data_dir}")
    
    data_dir = Path(data_dir)
    
    # Check if data files exist
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training data file not found: {train_file}")
    
    if not val_file.exists():
        logger.warning(f"Validation data file not found: {val_file}")
        logger.warning("Using a portion of training data for validation")
        
        # Load training dataset
        dataset = load_dataset(
            "json",
            data_files=str(train_file),
            split="train",
        )
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    else:
        # Load datasets
        train_dataset = load_dataset(
            "json",
            data_files=str(train_file),
            split="train",
        )
        val_dataset = load_dataset(
            "json",
            data_files=str(val_file),
            split="train",
        )
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    # Set maximum sequence length
    if max_length is None:
        max_length = tokenizer.model_max_length
    
    # Tokenize datasets
    def tokenize_function(examples):
        # Combine input and output text
        texts = [
            f"Input: {input_text}\nOutput: {output_text}"
            for input_text, output_text in zip(examples["input_text"], examples["output_text"])
        ]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Set labels
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Apply tokenization
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input_text", "output_text"],
    )
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input_text", "output_text"],
    )
    
    return train_dataset, val_dataset


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config: FineTuningConfig,
) -> Trainer:
    """
    Create a trainer.

    Args:
        model: Model to train.
        tokenizer: Tokenizer to use.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Fine-tuning configuration.

    Returns:
        Trainer: Trainer object.
    """
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=config.training.fp16,
        max_grad_norm=config.training.max_grad_norm,
        seed=config.seed,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    return trainer


def finetune_model(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Union[FineTuningConfig, Dict, str, Path]] = None,
) -> str:
    """
    Fine-tune a Gemini model.

    Args:
        data_dir: Directory containing training data.
        output_dir: Directory to save fine-tuned model.
        config: Fine-tuning configuration.

    Returns:
        str: Path to the fine-tuned model.
    """
    # Load configuration
    if config is None:
        # Use default configuration
        config = FineTuningConfig(
            model_name="gemini-pro",
            vertex_ai={"project_id": "your-project-id"},
        )
    elif isinstance(config, dict):
        # Create configuration from dictionary
        config = FineTuningConfig(**config)
    elif isinstance(config, (str, Path)):
        # Load configuration from file
        config = load_config(config)
    
    # Override output directory
    config.output_dir = str(output_dir)
    
    # Set up training environment
    model, tokenizer, env_config = setup_training_environment(config)
    
    # Load training data
    train_dataset, val_dataset = load_training_data(data_dir, tokenizer)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, config)
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    config_path = Path(output_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False, sort_keys=False)
    
    logger.info("Training completed successfully")
    
    return str(output_dir)


def create_training_script(
    output_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Create a training script.

    Args:
        output_path: Path to save the script.
        config_path: Path to the configuration file.

    Returns:
        str: Path to the created script.
    """
    from src.fine_tuning.setup import save_training_template
    
    # Save training script template
    script_path = save_training_template(output_path)
    
    # Create configuration file if not provided
    if config_path is None:
        config_path = Path(output_path).parent / "fine_tuning_config.yaml"
        from src.fine_tuning.config import create_config_template
        create_config_template(config_path)
    
    return script_path


def create_vertex_ai_script(
    output_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Create a Vertex AI training script.

    Args:
        output_path: Path to save the script.
        config_path: Path to the configuration file.

    Returns:
        str: Path to the created script.
    """
    from src.fine_tuning.setup import save_vertex_ai_template
    
    # Save Vertex AI training script template
    script_path = save_vertex_ai_template(output_path)
    
    # Create configuration file if not provided
    if config_path is None:
        config_path = Path(output_path).parent / "fine_tuning_config.yaml"
        from src.fine_tuning.config import create_config_template
        create_config_template(config_path)
    
    return script_path


def create_training_templates(
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """
    Create training templates.

    Args:
        output_dir: Directory to save templates.

    Returns:
        Dict[str, str]: Dictionary mapping template names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration template
    from src.fine_tuning.config import create_config_template
    config_path = create_config_template(output_dir / "fine_tuning_config.yaml")
    
    # Create training script template
    script_path = create_training_script(output_dir / "train.py", config_path)
    
    # Create Vertex AI script template
    vertex_ai_path = create_vertex_ai_script(output_dir / "vertex_ai_train.py", config_path)
    
    return {
        "config": config_path,
        "train_script": script_path,
        "vertex_ai_script": vertex_ai_path,
    }
