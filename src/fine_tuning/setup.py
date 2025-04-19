"""
Fine-tuning setup module.

This module provides utilities for setting up PyTorch with PEFT/LoRA for fine-tuning.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from src.fine_tuning.config import FineTuningConfig

logger = logging.getLogger(__name__)


def setup_pytorch_environment(config: FineTuningConfig) -> Dict:
    """
    Set up PyTorch environment.

    Args:
        config: Fine-tuning configuration.

    Returns:
        Dict: PyTorch environment configuration.
    """
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Set up CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        
        # Log CUDA information
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available. Training will be slow.")
    
    # Set up mixed precision training
    if config.training.fp16:
        if torch.cuda.is_available():
            logger.info("Using mixed precision training")
        else:
            logger.warning("Mixed precision training requested but CUDA not available. Disabling.")
            config.training.fp16 = False
    
    # Return environment configuration
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "fp16": config.training.fp16,
        "seed": config.seed,
    }


def load_model_and_tokenizer(
    config: FineTuningConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer.

    Args:
        config: Fine-tuning configuration.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Model and tokenizer.
    """
    logger.info(f"Loading model {config.model_name}")
    
    # Use the model name directly - we're using gpt2 instead of Gemini models
    # since Gemini models are not available on Hugging Face
    hf_model_name = config.model_name
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        cache_dir=config.cache_dir,
        use_fast=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        cache_dir=config.cache_dir,
        torch_dtype=torch.float16 if config.training.fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    logger.info(f"Loaded model with {model.num_parameters():,} parameters")
    
    return model, tokenizer


def setup_peft_model(
    model: PreTrainedModel,
    config: FineTuningConfig,
) -> PreTrainedModel:
    """
    Set up PEFT model with LoRA.

    Args:
        model: Base model.
        config: Fine-tuning configuration.

    Returns:
        PreTrainedModel: PEFT model.
    """
    logger.info("Setting up PEFT model with LoRA")
    
    # Prepare model for k-bit training if using mixed precision
    if config.training.fp16:
        model = prepare_model_for_kbit_training(model)
    
    # Create LoRA configuration
    # For GPT-2, the attention modules are named 'c_attn' and 'c_proj'
    # instead of 'q_proj' and 'v_proj'
    target_modules = ["c_attn", "c_proj"] if config.model_name == "gpt2" else config.lora.target_modules
    
    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=target_modules,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Get PEFT model
    peft_model = get_peft_model(model, peft_config)
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)")
    
    return peft_model


def setup_training_environment(
    config: FineTuningConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict]:
    """
    Set up training environment.

    Args:
        config: Fine-tuning configuration.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer, Dict]: Model, tokenizer, and environment configuration.
    """
    # Set up PyTorch environment
    env_config = setup_pytorch_environment(config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Set up PEFT model if using PEFT
    if config.method == "peft":
        model = setup_peft_model(model, config)
    
    return model, tokenizer, env_config


def save_training_template(output_path: Union[str, Path]) -> str:
    """
    Save a training script template.

    Args:
        output_path: Path to save the template.

    Returns:
        str: Path to the saved template.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Training script template
    template = """#!/usr/bin/env python3
\"\"\"
Gemini fine-tuning script.

This script fine-tunes a Gemini model using PEFT/LoRA.
\"\"\"

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.fine_tuning.config import load_config
from src.fine_tuning.setup import setup_training_environment


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    \"\"\"Parse command-line arguments.\"\"\"
    parser = argparse.ArgumentParser(description="Fine-tune a Gemini model")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/fine_tuning_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save fine-tuned model",
    )
    return parser.parse_args()


def load_training_data(data_dir, tokenizer):
    \"\"\"Load training data.\"\"\"
    logger.info(f"Loading training data from {data_dir}")
    
    # Load datasets
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_dir, "train.jsonl"),
        split="train",
    )
    val_dataset = load_dataset(
        "json",
        data_files=os.path.join(data_dir, "val.jsonl"),
        split="train",
    )
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    # Tokenize datasets
    def tokenize_function(examples):
        # Combine input and output text
        texts = [
            f"Input: {input_text}\\nOutput: {output_text}"
            for input_text, output_text in zip(examples["input_text"], examples["output_text"])
        ]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
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


def main():
    \"\"\"Main function.\"\"\"
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Override output directory
    config.output_dir = args.output_dir
    
    # Set up training environment
    model, tokenizer, env_config = setup_training_environment(config)
    
    # Load training data
    train_dataset, val_dataset = load_training_data(args.data_dir, tokenizer)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
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
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
"""
    
    # Save template
    with open(output_path, "w") as f:
        f.write(template)
    
    logger.info(f"Saved training script template to {output_path}")
    
    return str(output_path)


def save_vertex_ai_template(output_path: Union[str, Path]) -> str:
    """
    Save a Vertex AI training script template.

    Args:
        output_path: Path to save the template.

    Returns:
        str: Path to the saved template.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Vertex AI training script template
    template = """#!/usr/bin/env python3
\"\"\"
Vertex AI training script for Gemini fine-tuning.

This script sets up a Vertex AI custom training job for fine-tuning a Gemini model.
\"\"\"

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from google.cloud import aiplatform

from src.fine_tuning.config import load_config


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    \"\"\"Parse command-line arguments.\"\"\"
    parser = argparse.ArgumentParser(description="Set up Vertex AI training job")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/fine_tuning_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="GCS directory containing training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="GCS directory to save fine-tuned model",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="gemini-fine-tuning",
        help="Name of the training job",
    )
    return parser.parse_args()


def main():
    \"\"\"Main function.\"\"\"
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Initialize Vertex AI
    aiplatform.init(
        project=config.vertex_ai.project_id,
        location=config.vertex_ai.region,
    )
    
    # Create custom training job
    job = aiplatform.CustomTrainingJob(
        display_name=args.job_name,
        script_path="scripts/train.py",
        container_uri=config.vertex_ai.container_uri,
        requirements=["peft", "transformers", "datasets"],
    )
    
    # Start training job
    model = job.run(
        args=[
            f"--config_path={args.config_path}",
            f"--data_dir={args.data_dir}",
            f"--output_dir={args.output_dir}",
        ],
        replica_count=1,
        machine_type=config.vertex_ai.machine_type,
        accelerator_type=config.vertex_ai.accelerator_type,
        accelerator_count=config.vertex_ai.accelerator_count,
        service_account=config.vertex_ai.service_account,
    )
    
    logger.info(f"Training job started: {model.display_name}")
    logger.info(f"Model will be saved to {args.output_dir}")


if __name__ == "__main__":
    main()
"""
    
    # Save template
    with open(output_path, "w") as f:
        f.write(template)
    
    logger.info(f"Saved Vertex AI training script template to {output_path}")
    
    return str(output_path)
