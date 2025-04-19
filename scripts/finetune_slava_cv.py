#!/usr/bin/env python3
"""
Fine-tuning script for Slava's CV PDF.

This script runs the full pipeline to fine-tune a model on Slava's CV and portfolio.
"""

import os
import logging
import sys
from pathlib import Path
import time
import argparse

# Add the parent directory to the path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our utilities
from src.utils.config import get_config
from src.pdf_processing import process_directory
from src.evaluation.model_evaluator import evaluate_model_on_pdf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_pdf_processing(pdf_dir, output_dir, config_path):
    """Process PDF files."""
    logger.info(f"Processing PDFs from {pdf_dir}")
    
    # Load configuration
    config = get_config(config_path)
    
    # Process PDFs
    result = process_directory(
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        method=config.pdf_processing.extraction_method,
        config=config.pdf_processing.model_dump(),
        extract_meta=True
    )
    
    logger.info(f"PDF processing complete. Output in {output_dir}")
    return result


def prepare_training_data(text_dir, output_dir, config_path):
    """Prepare training data from extracted text."""
    logger.info(f"Preparing training data from {text_dir}")
    
    import subprocess
    cmd = [
        sys.executable, "-m", "src.main", "prepare-data",
        str(text_dir),  # Convert Path to string
        str(output_dir),  # Convert Path to string
        "--config-path", str(config_path)  # Convert Path to string
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    logger.info(f"Training data preparation complete. Output in {output_dir}")


def run_fine_tuning(data_dir, model_dir, config_path):
    """Run fine-tuning."""
    logger.info(f"Running fine-tuning with data from {data_dir}")
    
    import subprocess
    cmd = [
        sys.executable, "-m", "src.main", "finetune",
        str(data_dir),  # Convert Path to string
        str(model_dir),  # Convert Path to string
        "--config-path", str(config_path)  # Convert Path to string
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    logger.info(f"Fine-tuning complete. Model saved to {model_dir}")


def evaluate_model(model_dir, test_data, eval_dir, config_path):
    """Evaluate fine-tuned model."""
    logger.info(f"Evaluating model from {model_dir}")
    
    import subprocess
    cmd = [
        sys.executable, "-m", "src.main", "evaluate",
        str(model_dir),  # Convert Path to string
        str(test_data),  # Convert Path to string
        str(eval_dir),   # Convert Path to string
        "--config-path", str(config_path)  # Convert Path to string
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    logger.info(f"Evaluation complete. Results in {eval_dir}")


def test_model_interactively(model_dir, pdf_path=None):
    """Test the model interactively."""
    logger.info(f"Testing model from {model_dir} interactively")
    
    try:
        # Try to load the model and evaluate on the PDF
        if pdf_path:
            evaluate_model_on_pdf(
                pdf_path=pdf_path,
                model_name="gpt2",  # Base model
                use_standard=True,  # Use standard model
                use_peft=True,      # Use PEFT-adapted model
                num_questions=3,
                output_path=Path(model_dir) / "evaluation_results.yaml"
            )
        else:
            # Interactive mode with user questions
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Run interactive session
            print("\n" + "=" * 50)
            print("Interactive testing session")
            print("Type 'quit' to exit")
            print("=" * 50 + "\n")
            
            while True:
                # Get user input
                user_input = input("Enter question: ")
                if user_input.lower() == "quit":
                    break
                
                # Prepare input
                prompt = f"Question: {user_input}\nAnswer:"
                
                # Tokenize input
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                
                # Generate response
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=100,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7
                    )
                
                # Decode output
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract just the answer part
                answer_parts = response.split("Answer:")
                answer = answer_parts[-1].strip() if len(answer_parts) > 1 else response.strip()
                
                # Print response
                print("\nResponse:")
                print(answer)
                print("\n" + "-" * 50)
    
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        logger.info("Falling back to evaluate_pdf_peft.py for testing")
        
        # Fall back to the script in our new location
        import subprocess
        subprocess.run([sys.executable, "scripts/evaluate_model.py"], check=True)


def main():
    """Run the full pipeline."""
    parser = argparse.ArgumentParser(description="Fine-tune a model on Slava's CV")
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--env",
        default="local",
        help="Environment to use (local, vertex, production)"
    )
    parser.add_argument(
        "--skip-processing", 
        action="store_true",
        help="Skip PDF processing step"
    )
    parser.add_argument(
        "--skip-preparation", 
        action="store_true",
        help="Skip training data preparation step"
    )
    parser.add_argument(
        "--skip-training", 
        action="store_true",
        help="Skip fine-tuning step"
    )
    parser.add_argument(
        "--skip-evaluation", 
        action="store_true",
        help="Skip evaluation step"
    )
    parser.add_argument(
        "--skip-testing", 
        action="store_true",
        help="Skip interactive testing step"
    )
    
    args = parser.parse_args()
    
    # Set up directories
    base_dir = Path(__file__).parent.parent
    pdf_dir = base_dir / "data" / "pdfs"
    text_dir = base_dir / "data" / "extracted_text"
    data_dir = base_dir / "data" / "training_data"
    model_dir = base_dir / "data" / "model"
    eval_dir = base_dir / "data" / "evaluation"
    
    # Create directories if they don't exist
    for directory in [text_dir, data_dir, model_dir, eval_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Run pipeline steps
    start_time = time.time()
    
    if not args.skip_processing:
        run_pdf_processing(pdf_dir, text_dir, args.config)
    
    if not args.skip_preparation:
        prepare_training_data(text_dir, data_dir, args.config)
    
    if not args.skip_training:
        run_fine_tuning(data_dir, model_dir, args.config)
    
    if not args.skip_evaluation:
        test_data = data_dir / "test.jsonl"
        if test_data.exists():
            evaluate_model(model_dir, test_data, eval_dir, args.config)
        else:
            logger.warning(f"Test data not found at {test_data}. Skipping evaluation.")
    
    if not args.skip_testing:
        # Use the first PDF in the directory for testing
        pdf_file = next(pdf_dir.glob("*.pdf"), None)
        test_model_interactively(model_dir, pdf_file)
    
    total_time = time.time() - start_time
    logger.info(f"Pipeline completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    main() 