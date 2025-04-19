#!/usr/bin/env python3
"""
Training progress checker script.

This script monitors a model directory to check if training is complete
by looking for expected model files. It is useful for monitoring long-running
training jobs.
"""

import os
import time
import sys
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config

def check_model_directory(model_dir, expected_files=None):
    """
    Check if the model directory contains expected model files.
    
    Args:
        model_dir: Path to the model directory
        expected_files: List of expected files (default: common model files)
        
    Returns:
        bool: True if expected files are found, False otherwise
    """
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        print(f"Model directory {model_dir} does not exist yet.")
        return False
    
    # Default expected files for various model types
    if expected_files is None:
        expected_files = [
            # Core model files
            "config.json", 
            "pytorch_model.bin",
            
            # LoRA/PEFT files
            "adapter_config.json",
            "adapter_model.bin",
            
            # Tokenizer files
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            
            # Optimization files
            "optimizer.pt",
            "scheduler.pt",
            
            # Training logs
            "training_args.bin",
            "trainer_state.json"
        ]
    
    found_files = []
    
    for file in model_dir.glob("*"):
        print(f"Found file: {file.name}")
        if file.name in expected_files:
            found_files.append(file.name)
    
    if not found_files:
        print("No model files found yet.")
        return False
    
    print(f"Found {len(found_files)} model files: {', '.join(found_files)}")
    
    # Consider training complete if we find at least config.json and a model file
    core_files = ["config.json"]
    model_files = ["pytorch_model.bin", "adapter_model.bin"]
    
    has_core = any(f in found_files for f in core_files)
    has_model = any(f in found_files for f in model_files)
    
    if has_core and has_model:
        print("Found essential model files. Training appears to be complete!")
        return True
    else:
        print("Some essential model files are still missing.")
        return False

def main():
    """Run the model directory checker."""
    parser = argparse.ArgumentParser(description="Check model directory for training completion")
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="data/model",
        help="Path to model directory"
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=5,
        help="Maximum number of checks"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=10,
        help="Interval between checks in seconds"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Environment (local, vertex, production)"
    )
    
    args = parser.parse_args()
    
    # Try to load model directory from config if not specified
    if args.model_dir == "data/model" and args.config_path:
        try:
            config = get_config(args.config_path, env=args.env)
            model_dir = Path(config.fine_tuning.output_dir) if hasattr(config.fine_tuning, "output_dir") else Path(args.model_dir)
        except Exception:
            model_dir = Path(args.model_dir)
    else:
        model_dir = Path(args.model_dir)
    
    max_checks = args.max_checks
    check_interval = args.check_interval
    
    print(f"Checking model directory {model_dir} every {check_interval} seconds...")
    
    for i in range(max_checks):
        print(f"\nCheck {i+1}/{max_checks}:")
        if check_model_directory(model_dir):
            print("Model training appears to be completed!")
            break
        
        if i < max_checks - 1:
            print(f"Waiting {check_interval} seconds before next check...")
            time.sleep(check_interval)
    
    print("\nModel directory check complete.")

if __name__ == "__main__":
    main() 