#!/usr/bin/env python3
"""
Unified model evaluation script.

This script provides a command-line interface for evaluating models on PDF content.
It replaces the functionality of evaluate_pdf.py and evaluate_pdf_peft.py.
"""

import os
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.model_evaluator import evaluate_model_on_pdf
from src.utils.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main():
    """Run model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model(s) on PDF content")
    
    # PDF input options
    parser.add_argument(
        "--pdf-path", 
        type=str, 
        default="data/pdfs/Slava labovkin- Protfolio+cv.pdf", 
        help="Path to PDF file"
    )
    
    # Model options
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="gpt2", 
        help="Base model name"
    )
    parser.add_argument(
        "--use-standard", 
        action="store_true", 
        default=True, 
        help="Use standard model"
    )
    parser.add_argument(
        "--use-peft", 
        action="store_true", 
        help="Use PEFT-adapted model"
    )
    parser.add_argument(
        "--peft-config", 
        type=str, 
        help="Path to PEFT configuration file (JSON or YAML)"
    )
    
    # Generation options
    parser.add_argument(
        "--num-questions", 
        type=int, 
        default=3, 
        help="Number of questions to generate"
    )
    parser.add_argument(
        "--max-new-tokens", 
        type=int, 
        default=100, 
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Temperature for sampling"
    )
    
    # Output options
    parser.add_argument(
        "--output-path", 
        type=str, 
        help="Path to save results"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/evaluation", 
        help="Directory to save results"
    )
    
    # Config options
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
    
    # Load configuration
    config = get_config(args.config_path, env=args.env)
    
    # Determine output path if not provided
    output_path = args.output_path
    if not output_path and args.output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate filename based on model and PDF
        pdf_name = Path(args.pdf_path).stem
        model_type = []
        if args.use_standard:
            model_type.append("standard")
        if args.use_peft:
            model_type.append("peft")
        
        model_suffix = "-".join(model_type)
        output_path = Path(args.output_dir) / f"evaluation_{pdf_name}_{args.model_name}_{model_suffix}.yaml"
    
    # Load PEFT configuration if provided
    peft_config = None
    if args.peft_config:
        import yaml
        import json
        
        peft_config_path = Path(args.peft_config)
        if peft_config_path.exists():
            try:
                if peft_config_path.suffix.lower() in [".yaml", ".yml"]:
                    with open(peft_config_path, "r") as f:
                        peft_config = yaml.safe_load(f)
                elif peft_config_path.suffix.lower() == ".json":
                    with open(peft_config_path, "r") as f:
                        peft_config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading PEFT configuration: {str(e)}")
    
    # Use config from config file if no explicit PEFT config provided
    if not peft_config and hasattr(config, "fine_tuning") and hasattr(config.fine_tuning, "peft"):
        peft_config = config.fine_tuning.peft.model_dump()
    
    # Ensure at least one model type is enabled
    if not args.use_standard and not args.use_peft:
        logger.warning("No model type specified. Enabling standard model by default.")
        args.use_standard = True
    
    # Run evaluation
    evaluate_model_on_pdf(
        pdf_path=args.pdf_path,
        model_name=args.model_name,
        use_standard=args.use_standard,
        use_peft=args.use_peft,
        num_questions=args.num_questions,
        peft_config=peft_config,
        output_path=output_path
    )

if __name__ == "__main__":
    main() 