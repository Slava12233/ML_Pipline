#!/usr/bin/env python3
"""
Unified entry point for fine-tuning pipeline.

This script provides a consistent command-line interface to access all 
functionality in the fine-tuning pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from typing import Optional, List
from enum import Enum

from src.utils.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="fine-tuning",
    help="Fine-tuning pipeline for PDF-based model training",
    add_completion=False,
)

# Environment enum
class Environment(str, Enum):
    LOCAL = "local"
    VERTEX = "vertex"
    PRODUCTION = "production"


# PDF processing command group
pdf_app = typer.Typer(help="PDF processing commands")
app.add_typer(pdf_app, name="pdf")

@pdf_app.command("process")
def process_pdfs(
    pdf_dir: str = typer.Argument("data/pdfs", help="Directory containing PDF files"),
    output_dir: str = typer.Argument("data/extracted_text", help="Directory to save extracted text"),
    config_path: str = typer.Option("config/config.yaml", "--config", "-c", help="Path to configuration file"),
    env: Optional[Environment] = typer.Option(None, "--env", "-e", help="Environment (local, vertex, production)"),
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Extraction method (pypdf2 or document_ai)"),
    extract_meta: bool = typer.Option(True, "--extract-meta/--no-extract-meta", help="Extract and save metadata"),
):
    """Extract text and metadata from PDF files."""
    from src.pdf_processing import process_directory
    
    # Load configuration
    config = get_config(config_path, env=env.value if env else None)
    
    # Use method from command line or config
    extraction_method = method or config.pdf_processing.extraction_method
    
    # Process PDFs
    logger.info(f"Processing PDFs from {pdf_dir} to {output_dir} using {extraction_method}")
    result = process_directory(
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        method=extraction_method,
        config=config.pdf_processing.model_dump(),
        extract_meta=extract_meta
    )
    
    logger.info(f"Processed {len(result['text_files'])} PDF files")
    if extract_meta:
        logger.info(f"Extracted metadata for {len(result['metadata_files'])} PDF files")


# Data preparation command group
data_app = typer.Typer(help="Data preparation commands")
app.add_typer(data_app, name="data")

@data_app.command("prepare")
def prepare_data(
    text_dir: str = typer.Argument("data/extracted_text", help="Directory containing extracted text"),
    output_dir: str = typer.Argument("data/training_data", help="Directory to save training data"),
    config_path: str = typer.Option("config/config.yaml", "--config", "-c", help="Path to configuration file"),
    env: Optional[Environment] = typer.Option(None, "--env", "-e", help="Environment (local, vertex, production)"),
    export_to_gcs: bool = typer.Option(False, "--export-gcs/--no-export-gcs", help="Export training data to Google Cloud Storage"),
    generate_quality_report: bool = typer.Option(True, "--quality-report/--no-quality-report", help="Generate quality report"),
):
    """Prepare training data from extracted text."""
    # Import here to avoid circular imports
    import subprocess
    
    cmd = [
        sys.executable, "-m", "src.main", "prepare-data",
        text_dir,
        output_dir,
        "--config-path", config_path
    ]
    
    if env:
        os.environ["PIPELINE_ENV"] = env.value
    
    if export_to_gcs:
        cmd.append("--export-to-gcs")
    
    if not generate_quality_report:
        cmd.append("--no-generate-quality-report")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# Training command group
train_app = typer.Typer(help="Training commands")
app.add_typer(train_app, name="train")

@train_app.command("finetune")
def finetune(
    data_dir: str = typer.Argument("data/training_data", help="Directory containing training data"),
    output_dir: str = typer.Argument("data/model", help="Directory to save fine-tuned model"),
    config_path: str = typer.Option("config/config.yaml", "--config", "-c", help="Path to configuration file"),
    env: Optional[Environment] = typer.Option(None, "--env", "-e", help="Environment (local, vertex, production)"),
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Fine-tuning method (peft or full)"),
    model_name: Optional[str] = typer.Option(None, "--model", help="Model name"),
    use_best_params: bool = typer.Option(False, "--use-best-params", help="Use best hyperparameters from optimization"),
    hpo_dir: Optional[str] = typer.Option(None, "--hpo-dir", help="Directory containing hyperparameter optimization results"),
):
    """Fine-tune model on prepared data."""
    # Import here to avoid circular imports
    import subprocess
    
    cmd = [
        sys.executable, "-m", "src.main", "finetune",
        data_dir,
        output_dir,
        "--config-path", config_path
    ]
    
    if env:
        os.environ["PIPELINE_ENV"] = env.value
    
    if method:
        cmd.extend(["--method", method])
    
    if model_name:
        cmd.extend(["--model-name", model_name])
    
    if use_best_params:
        cmd.append("--use-best-params")
        if hpo_dir:
            cmd.extend(["--hpo-dir", hpo_dir])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# Evaluation command group
eval_app = typer.Typer(help="Evaluation commands")
app.add_typer(eval_app, name="eval")

@eval_app.command("model")
def evaluate_model(
    model_dir: str = typer.Argument("data/model", help="Directory containing fine-tuned model"),
    test_data: str = typer.Argument("data/training_data/test.jsonl", help="Path to test data"),
    output_dir: str = typer.Argument("data/evaluation", help="Directory to save evaluation results"),
    config_path: str = typer.Option("config/config.yaml", "--config", "-c", help="Path to configuration file"),
    env: Optional[Environment] = typer.Option(None, "--env", "-e", help="Environment (local, vertex, production)"),
):
    """Evaluate fine-tuned model on test data."""
    # Import here to avoid circular imports
    import subprocess
    
    cmd = [
        sys.executable, "-m", "src.main", "evaluate",
        model_dir,
        test_data,
        output_dir,
        "--config-path", config_path
    ]
    
    if env:
        os.environ["PIPELINE_ENV"] = env.value
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

@eval_app.command("pdf")
def evaluate_on_pdf(
    pdf_path: str = typer.Argument("data/pdfs/Slava labovkin- Protfolio+cv.pdf", help="Path to PDF file"),
    model_name: str = typer.Option("gpt2", "--model", "-m", help="Base model name"),
    model_dir: Optional[str] = typer.Option("data/model", "--model-dir", help="Path to fine-tuned model"),
    use_standard: bool = typer.Option(True, "--use-standard/--no-use-standard", help="Use standard model"),
    use_peft: bool = typer.Option(False, "--use-peft/--no-use-peft", help="Use PEFT-adapted model"),
    num_questions: int = typer.Option(3, "--num-questions", "-n", help="Number of questions to generate"),
    output_dir: str = typer.Option("data/evaluation", "--output-dir", "-o", help="Directory to save evaluation results"),
    config_path: str = typer.Option("config/config.yaml", "--config", "-c", help="Path to configuration file"),
    env: Optional[Environment] = typer.Option(None, "--env", "-e", help="Environment (local, vertex, production)"),
):
    """Evaluate model on PDF content."""
    from src.evaluation.model_evaluator import evaluate_model_on_pdf
    from pathlib import Path
    
    # Generate output path
    pdf_name = Path(pdf_path).stem
    model_type = []
    if use_standard:
        model_type.append("standard")
    if use_peft:
        model_type.append("peft")
    
    model_suffix = "-".join(model_type)
    output_path = Path(output_dir) / f"evaluation_{pdf_name}_{model_name}_{model_suffix}.yaml"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    config = get_config(config_path, env=env.value if env else None)
    
    # Get PEFT config from the configuration if using PEFT
    peft_config = None
    if use_peft and hasattr(config, "fine_tuning") and hasattr(config.fine_tuning, "peft"):
        peft_config = config.fine_tuning.peft.model_dump()
    
    # Run evaluation
    evaluate_model_on_pdf(
        pdf_path=pdf_path,
        model_name=model_name,
        use_standard=use_standard,
        use_peft=use_peft,
        num_questions=num_questions,
        peft_config=peft_config,
        output_path=output_path
    )


# Utility commands
@app.command("check-training")
def check_training(
    model_dir: str = typer.Option("data/model", "--model-dir", "-m", help="Path to model directory"),
    max_checks: int = typer.Option(5, "--max-checks", "-n", help="Maximum number of checks"),
    check_interval: int = typer.Option(10, "--check-interval", "-i", help="Interval between checks in seconds"),
    config_path: str = typer.Option("config/config.yaml", "--config", "-c", help="Path to configuration file"),
    env: Optional[Environment] = typer.Option(None, "--env", "-e", help="Environment (local, vertex, production)"),
):
    """Check model directory for training completion."""
    # Import the moved check_training script
    from scripts.check_training import check_model_directory
    
    # Try to load model directory from config if not specified
    if model_dir == "data/model" and config_path:
        try:
            config = get_config(config_path, env=env.value if env else None)
            model_dir = config.fine_tuning.output_dir if hasattr(config.fine_tuning, "output_dir") else model_dir
        except Exception:
            pass
    
    print(f"Checking model directory {model_dir} every {check_interval} seconds...")
    
    for i in range(max_checks):
        print(f"\nCheck {i+1}/{max_checks}:")
        if check_model_directory(model_dir):
            print("Model training appears to be completed!")
            break
        
        if i < max_checks - 1:
            print(f"Waiting {check_interval} seconds before next check...")
            import time
            time.sleep(check_interval)
    
    print("\nModel directory check complete.")


# Pipeline command - run the full pipeline
@app.command("pipeline")
def run_pipeline(
    pdf_dir: str = typer.Option("data/pdfs", "--pdf-dir", help="Directory containing PDF files"),
    output_dir: str = typer.Option("data/", "--output-dir", "-o", help="Base directory for all outputs"),
    config_path: str = typer.Option("config/config.yaml", "--config", "-c", help="Path to configuration file"),
    env: Optional[Environment] = typer.Option(None, "--env", "-e", help="Environment (local, vertex, production)"),
    steps: List[str] = typer.Option(
        ["process", "prepare", "finetune", "evaluate", "test"],
        "--steps",
        "-s",
        help="Pipeline steps to run"
    ),
    extract_metadata: bool = typer.Option(True, "--extract-meta/--no-extract-meta", help="Extract and save metadata from PDFs"),
    generate_quality_report: bool = typer.Option(True, "--quality-report/--no-quality-report", help="Generate quality report"),
    export_to_gcs: bool = typer.Option(False, "--export-gcs/--no-export-gcs", help="Export data to Google Cloud Storage"),
    model_name: Optional[str] = typer.Option(None, "--model", help="Model name for fine-tuning"),
):
    """Run the complete pipeline or specified steps."""
    # Import here to avoid circular imports
    import subprocess
    from pathlib import Path
    
    # Set up directories
    output_dir = Path(output_dir)
    pdf_dir = Path(pdf_dir)
    text_dir = output_dir / "extracted_text"
    data_dir = output_dir / "training_data"
    model_dir = output_dir / "model"
    eval_dir = output_dir / "evaluation"
    
    # Create directories if they don't exist
    for directory in [text_dir, data_dir, model_dir, eval_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Process PDFs
    if "process" in steps:
        logger.info("Running PDF processing step")
        process_pdfs(
            pdf_dir=str(pdf_dir),
            output_dir=str(text_dir),
            config_path=config_path,
            env=env,
            extract_meta=extract_metadata
        )
    
    # Prepare training data
    if "prepare" in steps:
        logger.info("Running data preparation step")
        prepare_data(
            text_dir=str(text_dir),
            output_dir=str(data_dir),
            config_path=config_path,
            env=env,
            export_to_gcs=export_to_gcs,
            generate_quality_report=generate_quality_report
        )
    
    # Fine-tune model
    if "finetune" in steps:
        logger.info("Running fine-tuning step")
        finetune(
            data_dir=str(data_dir),
            output_dir=str(model_dir),
            config_path=config_path,
            env=env,
            model_name=model_name
        )
    
    # Evaluate model on test data
    if "evaluate" in steps:
        logger.info("Running evaluation step")
        test_data = data_dir / "test.jsonl"
        if test_data.exists():
            evaluate_model(
                model_dir=str(model_dir),
                test_data=str(test_data),
                output_dir=str(eval_dir),
                config_path=config_path,
                env=env
            )
        else:
            logger.warning(f"Test data not found at {test_data}. Skipping evaluation.")
    
    # Test model on PDF
    if "test" in steps:
        logger.info("Running PDF evaluation step")
        # Use the first PDF in the directory for testing
        pdf_file = next(pdf_dir.glob("*.pdf"), None)
        if pdf_file:
            evaluate_on_pdf(
                pdf_path=str(pdf_file),
                model_name="gpt2",
                model_dir=str(model_dir),
                use_standard=True,
                use_peft=True,
                num_questions=3,
                output_dir=str(eval_dir),
                config_path=config_path,
                env=env
            )
        else:
            logger.warning(f"No PDF files found in {pdf_dir}. Skipping testing.")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    app() 