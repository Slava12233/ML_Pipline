"""
Hyperparameter optimization module.

This module provides functions for hyperparameter optimization for fine-tuning.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import optuna
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.fine_tuning.config import FineTuningConfig, load_config
from src.fine_tuning.train import finetune_model, load_training_data

logger = logging.getLogger(__name__)


def define_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define hyperparameter search space.

    Args:
        trial: Optuna trial.

    Returns:
        Dict[str, Any]: Hyperparameter dictionary.
    """
    # Learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    
    # Batch size
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    
    # LoRA rank
    lora_r = trial.suggest_int("lora_r", 4, 32, step=4)
    
    # LoRA alpha
    lora_alpha = trial.suggest_int("lora_alpha", 8, 64, step=8)
    
    # LoRA dropout
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.5, step=0.1)
    
    # Weight decay
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2, step=0.05)
    
    # Warmup steps
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2, step=0.05)
    
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
    }


def create_config_from_params(
    base_config: FineTuningConfig,
    params: Dict[str, Any],
) -> FineTuningConfig:
    """
    Create a configuration from hyperparameters.

    Args:
        base_config: Base configuration.
        params: Hyperparameters.

    Returns:
        FineTuningConfig: Updated configuration.
    """
    # Create a copy of the base configuration
    config_dict = base_config.dict()
    
    # Update training parameters
    if "learning_rate" in params:
        config_dict["training"]["learning_rate"] = params["learning_rate"]
    
    if "batch_size" in params:
        config_dict["training"]["batch_size"] = params["batch_size"]
    
    if "weight_decay" in params:
        config_dict["training"]["weight_decay"] = params["weight_decay"]
    
    if "warmup_ratio" in params:
        # Convert warmup ratio to steps based on dataset size and batch size
        # This will be updated in the objective function
        config_dict["training"]["warmup_ratio"] = params["warmup_ratio"]
    
    # Update LoRA parameters
    if "lora_r" in params:
        config_dict["lora"]["r"] = params["lora_r"]
    
    if "lora_alpha" in params:
        config_dict["lora"]["alpha"] = params["lora_alpha"]
    
    if "lora_dropout" in params:
        config_dict["lora"]["dropout"] = params["lora_dropout"]
    
    # Create new configuration
    return FineTuningConfig(**config_dict)


def objective(
    trial: optuna.Trial,
    data_dir: Union[str, Path],
    base_config: FineTuningConfig,
    tmp_dir: Union[str, Path],
    eval_fn: Optional[Callable[[str, str], float]] = None,
) -> float:
    """
    Objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial.
        data_dir: Directory containing training data.
        base_config: Base configuration.
        tmp_dir: Temporary directory for outputs.
        eval_fn: Evaluation function.

    Returns:
        float: Evaluation metric (lower is better).
    """
    # Define hyperparameters
    params = define_search_space(trial)
    
    # Create configuration
    config = create_config_from_params(base_config, params)
    
    # Create output directory
    output_dir = Path(tmp_dir) / f"trial_{trial.number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update output directory
    config.output_dir = str(output_dir)
    
    # Update warmup steps based on dataset size and batch size
    try:
        # Import here to avoid circular imports
        from src.fine_tuning.setup import setup_training_environment
        
        # Set up training environment
        model, tokenizer, _ = setup_training_environment(config)
        
        # Load training data
        train_dataset, _ = load_training_data(data_dir, tokenizer)
        
        # Calculate number of training steps
        num_examples = len(train_dataset)
        batch_size = config.training.batch_size
        grad_accum_steps = config.training.gradient_accumulation_steps
        epochs = config.training.epochs
        
        num_steps = (num_examples // (batch_size * grad_accum_steps)) * epochs
        
        # Update warmup steps
        if "warmup_ratio" in params:
            warmup_steps = int(num_steps * params["warmup_ratio"])
            config.training.warmup_steps = warmup_steps
    
    except Exception as e:
        logger.error(f"Error calculating warmup steps: {str(e)}")
        # Use default warmup steps
        config.training.warmup_steps = 100
    
    try:
        # Fine-tune model
        model_path = finetune_model(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config,
        )
        
        # Evaluate model
        if eval_fn is not None:
            # Use provided evaluation function
            metric = eval_fn(model_path, data_dir)
        else:
            # Use validation loss as metric
            metrics_file = Path(model_path) / "trainer_state.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    trainer_state = json.load(f)
                
                # Get best validation loss
                if "best_metric" in trainer_state:
                    metric = trainer_state["best_metric"]
                else:
                    # Get last validation loss
                    log_history = trainer_state.get("log_history", [])
                    val_losses = [
                        entry["eval_loss"]
                        for entry in log_history
                        if "eval_loss" in entry
                    ]
                    
                    if val_losses:
                        metric = min(val_losses)
                    else:
                        # No validation loss found
                        metric = float("inf")
            else:
                # No metrics file found
                metric = float("inf")
        
        # Report intermediate values
        trial.report(metric, step=config.training.epochs)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return metric
    
    except optuna.TrialPruned:
        # Re-raise pruning exception
        raise
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        return float("inf")


def optimize_hyperparameters(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Union[FineTuningConfig, Dict, str, Path]] = None,
    n_trials: int = 20,
    timeout: Optional[int] = None,
    eval_fn: Optional[Callable[[str, str], float]] = None,
) -> Tuple[Dict[str, Any], FineTuningConfig]:
    """
    Optimize hyperparameters for fine-tuning.

    Args:
        data_dir: Directory containing training data.
        output_dir: Directory to save results.
        config: Fine-tuning configuration.
        n_trials: Number of trials.
        timeout: Timeout in seconds.
        eval_fn: Evaluation function.

    Returns:
        Tuple[Dict[str, Any], FineTuningConfig]: Best parameters and configuration.
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
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for trials
    tmp_dir = output_dir / "trials"
    tmp_dir.mkdir(exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(),
        pruner=MedianPruner(),
    )
    
    # Optimize
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
    study.optimize(
        lambda trial: objective(trial, data_dir, config, tmp_dir, eval_fn),
        n_trials=n_trials,
        timeout=timeout,
    )
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    
    # Create best configuration
    best_config = create_config_from_params(config, best_params)
    
    # Save best parameters
    params_path = output_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    # Save best configuration
    config_path = output_dir / "best_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(best_config.dict(), f, default_flow_style=False, sort_keys=False)
    
    # Save study
    study_path = output_dir / "study.pkl"
    optuna.save_study(study, str(study_path))
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image(str(output_dir / "optimization_history.png"))
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        fig.write_image(str(output_dir / "param_importances.png"))
        
        # Plot parallel coordinate
        from optuna.visualization import plot_parallel_coordinate
        fig = plot_parallel_coordinate(study)
        fig.write_image(str(output_dir / "parallel_coordinate.png"))
        
    except ImportError:
        logger.warning("Visualization packages not installed. Skipping visualization.")
    
    logger.info(f"Hyperparameter optimization completed. Best parameters saved to {params_path}")
    
    return best_params, best_config


def finetune_with_best_params(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    hpo_dir: Union[str, Path],
) -> str:
    """
    Fine-tune model with best hyperparameters.

    Args:
        data_dir: Directory containing training data.
        output_dir: Directory to save fine-tuned model.
        hpo_dir: Directory containing hyperparameter optimization results.

    Returns:
        str: Path to the fine-tuned model.
    """
    # Load best configuration
    config_path = Path(hpo_dir) / "best_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Best configuration not found: {config_path}")
    
    config = load_config(config_path)
    
    # Override output directory
    config.output_dir = str(output_dir)
    
    # Fine-tune model
    logger.info(f"Fine-tuning model with best hyperparameters from {hpo_dir}")
    model_path = finetune_model(
        data_dir=data_dir,
        output_dir=output_dir,
        config=config,
    )
    
    return model_path
