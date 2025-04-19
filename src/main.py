#!/usr/bin/env python3
"""
Gemini PDF Fine-tuning Pipeline - Main Entry Point.

This script provides a command-line interface for running the Gemini PDF fine-tuning pipeline.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="gemini-pdf-finetuning",
    help="Pipeline for fine-tuning Gemini models on PDF documentation",
    add_completion=False,
)


class Config(BaseModel):
    """Configuration model for the pipeline."""

    project: dict
    gcp: dict
    pdf_processing: dict
    training_data: dict
    fine_tuning: dict
    evaluation: dict
    deployment: dict


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)


@app.command()
def process_pdfs(
    pdf_dir: str = typer.Argument(..., help="Directory containing PDF files"),
    output_dir: str = typer.Argument(..., help="Directory to save extracted text"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
    method: str = typer.Option(
        None, help="Extraction method (pypdf2 or document_ai). Overrides config."
    ),
    extract_metadata: bool = typer.Option(
        True, help="Extract and save metadata from PDFs"
    ),
):
    """Extract text and metadata from PDF files."""
    config = load_config(config_path)
    logger.info(f"Processing PDFs from {pdf_dir} to {output_dir}")
    logger.info(f"Using method: {method or config.pdf_processing['extraction_method']}")
    logger.info(f"Extracting metadata: {extract_metadata}")

    # Import here to avoid circular imports
    from pdf_processing import extract

    result = extract.process_directory(
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        method=method or config.pdf_processing["extraction_method"],
        config=config.pdf_processing,
        extract_meta=extract_metadata,
    )
    
    logger.info(f"Processed {len(result['text_files'])} PDF files")
    if extract_metadata:
        logger.info(f"Extracted metadata for {len(result['metadata_files'])} PDF files")


@app.command()
def prepare_data(
    text_dir: str = typer.Argument(..., help="Directory containing extracted text"),
    output_dir: str = typer.Argument(..., help="Directory to save training data"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
    export_to_gcs: bool = typer.Option(
        False, help="Export training data to Google Cloud Storage"
    ),
    generate_quality_report: bool = typer.Option(
        True, help="Generate quality report for training data"
    ),
):
    """Prepare training data from extracted text."""
    config = load_config(config_path)
    logger.info(f"Preparing training data from {text_dir} to {output_dir}")

    # Import here to avoid circular imports
    from data_preparation import generate

    result = generate.create_training_data(
        text_dir=text_dir,
        output_dir=output_dir,
        config=config.training_data,
    )
    
    logger.info(f"Created {result['train']} training examples, {result['val']} validation examples, and {result['test']} test examples")
    
    # Generate quality report if requested
    if generate_quality_report:
        from data_preparation import quality
        
        try:
            logger.info("Generating quality report for training data")
            
            # Create quality report directory
            quality_dir = Path(output_dir) / "quality_report"
            
            # Generate quality report
            report = quality.generate_quality_report(
                data_dir=output_dir,
                output_dir=quality_dir,
                include_plots=True,
            )
            
            # Generate HTML report
            html_path = quality.generate_html_report(
                report=report,
                output_dir=quality_dir,
            )
            
            logger.info(f"Quality report generated at {html_path}")
            
        except Exception as e:
            logger.error(f"Error generating quality report: {str(e)}")
    
    # Export to GCS if requested
    if export_to_gcs:
        from data_preparation import export
        
        try:
            logger.info("Exporting training data to Google Cloud Storage")
            
            # Get GCP config
            project_id = config.gcp["project_id"]
            bucket_name = config.gcp["storage"]["bucket_name"]
            
            # Check if project ID and bucket name are set
            if project_id == "your-project-id" or bucket_name == "gemini-pdf-finetuning-data":
                logger.error("GCP project ID or bucket name not set in config. Please update the configuration.")
                return
            
            # Export training data
            gcs_uris = export.export_training_data_to_gcs(
                data_dir=output_dir,
                project_id=project_id,
                bucket_name=bucket_name,
                gcs_dir=config.gcp["storage"]["training_data_folder"],
            )
            
            logger.info(f"Exported training data to GCS: {gcs_uris}")
            
            # Create manifest file
            manifest_path = Path(output_dir) / "gcs_manifest.json"
            export.create_manifest_file(gcs_uris, manifest_path)
            logger.info(f"Created manifest file at {manifest_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to GCS: {str(e)}")


@app.command()
def finetune(
    data_dir: str = typer.Argument(..., help="Directory containing training data"),
    output_dir: str = typer.Argument(..., help="Directory to save fine-tuned model"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
    method: str = typer.Option(
        None, help="Fine-tuning method (peft or full). Overrides config."
    ),
    model_name: str = typer.Option(
        None, help="Model name (gemini-pro, gemini-1.5-pro, etc.). Overrides config."
    ),
    use_best_params: bool = typer.Option(
        False, help="Use best hyperparameters from optimization"
    ),
    hpo_dir: str = typer.Option(
        None, help="Directory containing hyperparameter optimization results"
    ),
):
    """Fine-tune Gemini model on prepared data."""
    config = load_config(config_path)
    logger.info(f"Fine-tuning model with data from {data_dir}")
    logger.info(f"Model will be saved to {output_dir}")
    
    # Override configuration if specified
    if method:
        config.fine_tuning["method"] = method
        logger.info(f"Using fine-tuning method: {method}")
    
    if model_name:
        config.fine_tuning["model_name"] = model_name
        logger.info(f"Using model: {model_name}")

    # Import here to avoid circular imports
    from fine_tuning import train

    # Check if using best hyperparameters
    if use_best_params:
        if hpo_dir is None:
            logger.error("HPO directory not specified. Please provide --hpo-dir.")
            return
        
        # Import here to avoid circular imports
        from fine_tuning import hyperparameter
        
        # Fine-tune with best hyperparameters
        model_path = hyperparameter.finetune_with_best_params(
            data_dir=data_dir,
            output_dir=output_dir,
            hpo_dir=hpo_dir,
        )
    else:
        # Fine-tune model with specified configuration
        model_path = train.finetune_model(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config.fine_tuning,
        )
    
    logger.info(f"Fine-tuned model saved to {model_path}")


@app.command()
def evaluate(
    model_dir: str = typer.Argument(..., help="Directory containing fine-tuned model"),
    test_data: str = typer.Argument(..., help="Path to test data"),
    output_dir: str = typer.Argument(..., help="Directory to save evaluation results"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
):
    """Evaluate fine-tuned model on test data."""
    config = load_config(config_path)
    logger.info(f"Evaluating model from {model_dir} on {test_data}")
    logger.info(f"Results will be saved to {output_dir}")

    # Import here to avoid circular imports
    from evaluation import metrics

    metrics.evaluate_model(
        model_dir=model_dir,
        test_data=test_data,
        output_dir=output_dir,
        config=config.evaluation,
    )


@app.command()
def deploy(
    model_dir: str = typer.Argument(..., help="Directory containing fine-tuned model"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
):
    """Deploy fine-tuned model to Vertex AI Endpoints."""
    config = load_config(config_path)
    logger.info(f"Deploying model from {model_dir}")

    # Import here to avoid circular imports
    from deployment import vertex

    endpoint = vertex.deploy_model(
        model_dir=model_dir,
        config=config.deployment,
    )
    logger.info(f"Model deployed to endpoint: {endpoint}")


@app.command()
def generate_report(
    data_dir: str = typer.Argument(..., help="Directory containing training data"),
    output_dir: str = typer.Argument(..., help="Directory to save quality report"),
    include_plots: bool = typer.Option(
        True, help="Include plots in the report"
    ),
    generate_html: bool = typer.Option(
        True, help="Generate HTML report"
    ),
):
    """Generate quality report for training data."""
    logger.info(f"Generating quality report for {data_dir}")
    
    # Import here to avoid circular imports
    from data_preparation import quality
    
    try:
        # Generate quality report
        report = quality.generate_quality_report(
            data_dir=data_dir,
            output_dir=output_dir,
            include_plots=include_plots,
        )
        
        logger.info(f"Quality report generated at {report['report_path']}")
        
        # Generate HTML report if requested
        if generate_html:
            html_path = quality.generate_html_report(
                report=report,
                output_dir=output_dir,
            )
            
            logger.info(f"HTML report generated at {html_path}")
            
    except Exception as e:
        logger.error(f"Error generating quality report: {str(e)}")


@app.command()
def export_to_gcs(
    base_dir: str = typer.Argument(..., help="Base directory containing all data"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
    gcs_base_dir: str = typer.Option(
        "", help="Base directory path within the GCS bucket"
    ),
):
    """Export all data to Google Cloud Storage."""
    config = load_config(config_path)
    
    # Get GCP config
    project_id = config.gcp["project_id"]
    bucket_name = config.gcp["storage"]["bucket_name"]
    
    # Check if project ID and bucket name are set
    if project_id == "your-project-id" or bucket_name == "gemini-pdf-finetuning-data":
        logger.error("GCP project ID or bucket name not set in config. Please update the configuration.")
        return
    
    logger.info(f"Exporting data from {base_dir} to gs://{bucket_name}/{gcs_base_dir}")
    
    # Import here to avoid circular imports
    from data_preparation import export
    
    try:
        result = export.export_all_to_gcs(
            base_dir=base_dir,
            project_id=project_id,
            bucket_name=bucket_name,
            gcs_base_dir=gcs_base_dir,
        )
        
        logger.info(f"Exported data to GCS: {result}")
        
    except Exception as e:
        logger.error(f"Error exporting to GCS: {str(e)}")


@app.command()
def optimize_hyperparameters(
    data_dir: str = typer.Argument(..., help="Directory containing training data"),
    output_dir: str = typer.Argument(..., help="Directory to save optimization results"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
    n_trials: int = typer.Option(
        20, help="Number of optimization trials"
    ),
    timeout: int = typer.Option(
        None, help="Timeout in seconds"
    ),
):
    """Optimize hyperparameters for fine-tuning."""
    logger.info(f"Optimizing hyperparameters for {data_dir}")
    logger.info(f"Results will be saved to {output_dir}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Import here to avoid circular imports
    from fine_tuning import hyperparameter
    
    # Optimize hyperparameters
    best_params, best_config = hyperparameter.optimize_hyperparameters(
        data_dir=data_dir,
        output_dir=output_dir,
        config=config.fine_tuning,
        n_trials=n_trials,
        timeout=timeout,
    )
    
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best configuration saved to {output_dir}/best_config.yaml")


@app.command()
def create_training_templates(
    output_dir: str = typer.Argument(..., help="Directory to save templates"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
):
    """Create training templates for fine-tuning."""
    logger.info(f"Creating training templates in {output_dir}")
    
    # Import here to avoid circular imports
    from fine_tuning import train
    
    # Create templates
    templates = train.create_training_templates(output_dir)
    
    logger.info(f"Created training templates:")
    logger.info(f"  Configuration: {templates['config']}")
    logger.info(f"  Training script: {templates['train_script']}")
    logger.info(f"  Vertex AI script: {templates['vertex_ai_script']}")


@app.command()
def create_pipeline(
    pipeline_name: str = typer.Argument(..., help="Name of the pipeline"),
    output_dir: str = typer.Argument(..., help="Directory to save pipeline definition"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
    steps: List[str] = typer.Option(
        ["process", "prepare", "finetune", "evaluate", "deploy"],
        help="Pipeline steps to include",
    ),
    extract_metadata: bool = typer.Option(
        True, help="Extract and save metadata from PDFs"
    ),
    generate_quality_report: bool = typer.Option(
        True, help="Generate quality report for training data"
    ),
    export_to_gcs: bool = typer.Option(
        False, help="Export data to Google Cloud Storage"
    ),
    optimize_hyperparams: bool = typer.Option(
        False, help="Optimize hyperparameters before fine-tuning"
    ),
    n_trials: int = typer.Option(
        20, help="Number of hyperparameter optimization trials"
    ),
):
    """Create a Vertex AI Pipeline for fine-tuning."""
    logger.info(f"Creating pipeline {pipeline_name}")
    
    # Import here to avoid circular imports
    from fine_tuning import pipeline
    
    # Create pipeline
    pipeline_path = pipeline.create_pipeline(
        pipeline_name=pipeline_name,
        config_path=config_path,
        output_dir=output_dir,
        steps=steps,
        extract_metadata=extract_metadata,
        generate_quality_report=generate_quality_report,
        export_to_gcs=export_to_gcs,
        optimize_hyperparams=optimize_hyperparams,
        n_trials=n_trials,
    )
    
    logger.info(f"Pipeline created at {pipeline_path}")


@app.command()
def run_vertex_pipeline(
    pipeline_path: str = typer.Argument(..., help="Path to pipeline definition"),
    pdf_dir: str = typer.Argument(..., help="Directory containing PDF files"),
    output_dir: str = typer.Argument(..., help="Base directory for all outputs"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
    service_account: str = typer.Option(
        None, help="Service account to use"
    ),
    enable_caching: bool = typer.Option(
        True, help="Enable caching"
    ),
):
    """Run a Vertex AI Pipeline."""
    logger.info(f"Running pipeline {pipeline_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Get GCP config
    project_id = config.gcp["project_id"]
    region = config.gcp["region"]
    
    # Check if project ID is set
    if project_id == "your-project-id":
        logger.error("GCP project ID not set in config. Please update the configuration.")
        return
    
    # Import here to avoid circular imports
    from fine_tuning import pipeline
    
    # Run pipeline
    job_name = pipeline.run_pipeline(
        pipeline_path=pipeline_path,
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        config_path=config_path,
        project_id=project_id,
        region=region,
        service_account=service_account,
        enable_caching=enable_caching,
    )
    
    logger.info(f"Pipeline job submitted: {job_name}")


@app.command()
def create_pipeline_trigger(
    pipeline_path: str = typer.Argument(..., help="Path to pipeline definition"),
    trigger_name: str = typer.Argument(..., help="Name of the trigger"),
    schedule: str = typer.Argument(..., help="Cron schedule"),
    pdf_dir: str = typer.Argument(..., help="Directory containing PDF files"),
    output_dir: str = typer.Argument(..., help="Base directory for all outputs"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
    service_account: str = typer.Option(
        None, help="Service account to use"
    ),
):
    """Create a pipeline trigger."""
    logger.info(f"Creating pipeline trigger {trigger_name}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Get GCP config
    project_id = config.gcp["project_id"]
    region = config.gcp["region"]
    
    # Check if project ID is set
    if project_id == "your-project-id":
        logger.error("GCP project ID not set in config. Please update the configuration.")
        return
    
    # Import here to avoid circular imports
    from fine_tuning import pipeline
    
    # Create trigger
    trigger_name = pipeline.create_pipeline_trigger(
        pipeline_path=pipeline_path,
        trigger_name=trigger_name,
        schedule=schedule,
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        config_path=config_path,
        project_id=project_id,
        region=region,
        service_account=service_account,
    )
    
    logger.info(f"Pipeline trigger created: {trigger_name}")


@app.command()
def run_pipeline(
    pdf_dir: str = typer.Argument(..., help="Directory containing PDF files"),
    output_dir: str = typer.Argument(..., help="Base directory for all outputs"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to configuration file"
    ),
    steps: List[str] = typer.Option(
        ["process", "prepare", "finetune", "evaluate", "deploy"],
        help="Pipeline steps to run",
    ),
    extract_metadata: bool = typer.Option(
        True, help="Extract and save metadata from PDFs"
    ),
    generate_quality_report: bool = typer.Option(
        True, help="Generate quality report for training data"
    ),
    export_to_gcs: bool = typer.Option(
        False, help="Export data to Google Cloud Storage"
    ),
    model_name: str = typer.Option(
        None, help="Model name for fine-tuning. Overrides config."
    ),
    optimize_hyperparams: bool = typer.Option(
        False, help="Optimize hyperparameters before fine-tuning"
    ),
    n_trials: int = typer.Option(
        20, help="Number of hyperparameter optimization trials"
    ),
    use_vertex_pipeline: bool = typer.Option(
        False, help="Use Vertex AI Pipeline"
    ),
):
    """Run the complete pipeline or specified steps.
    
    If use_vertex_pipeline is True, the pipeline will be run on Vertex AI.
    Otherwise, it will be run locally.
    """
    config = load_config(config_path)
    logger.info(f"Running pipeline with PDFs from {pdf_dir}")
    logger.info(f"Output will be saved to {output_dir}")
    logger.info(f"Steps to run: {steps}")
    logger.info(f"Extracting metadata: {extract_metadata}")

    if use_vertex_pipeline:
        # Create pipeline
        pipeline_name = f"gemini-pdf-finetuning-{Path(output_dir).stem}"
        pipeline_output_dir = os.path.join(output_dir, "pipeline")
        
        # Import here to avoid circular imports
        from fine_tuning import pipeline
        
        # Create pipeline
        pipeline_path = pipeline.create_pipeline(
            pipeline_name=pipeline_name,
            config_path=config_path,
            output_dir=pipeline_output_dir,
            steps=steps,
            extract_metadata=extract_metadata,
            generate_quality_report=generate_quality_report,
            export_to_gcs=export_to_gcs,
            optimize_hyperparams=optimize_hyperparams,
            n_trials=n_trials,
        )
        
        # Load configuration
        config = load_config(config_path)
        
        # Get GCP config
        project_id = config.gcp["project_id"]
        region = config.gcp["region"]
        
        # Check if project ID is set
        if project_id == "your-project-id":
            logger.error("GCP project ID not set in config. Please update the configuration.")
            return
        
        # Run pipeline
        job_name = pipeline.run_pipeline(
            pipeline_path=pipeline_path,
            pdf_dir=pdf_dir,
            output_dir=output_dir,
            config_path=config_path,
            project_id=project_id,
            region=region,
            enable_caching=True,
        )
        
        logger.info(f"Pipeline job submitted: {job_name}")
    else:
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        text_dir = os.path.join(output_dir, "extracted_text")
        data_dir = os.path.join(output_dir, "training_data")
        model_dir = os.path.join(output_dir, "model")
        eval_dir = os.path.join(output_dir, "evaluation")

        # Run pipeline steps
        if "process" in steps:
            process_pdfs(pdf_dir, text_dir, config_path, extract_metadata=extract_metadata)

        if "prepare" in steps:
            prepare_data(
                text_dir, 
                data_dir, 
                config_path, 
                export_to_gcs=export_to_gcs,
                generate_quality_report=generate_quality_report,
            )

        if "finetune" in steps:
            if optimize_hyperparams:
                # Create hyperparameter optimization directory
                hpo_dir = os.path.join(output_dir, "hyperparameter_optimization")
                
                # Optimize hyperparameters
                optimize_hyperparameters(
                    data_dir=data_dir,
                    output_dir=hpo_dir,
                    config_path=config_path,
                    n_trials=n_trials,
                )
                
                # Fine-tune with best hyperparameters
                finetune(
                    data_dir=data_dir,
                    output_dir=model_dir,
                    config_path=config_path,
                    model_name=model_name,
                    use_best_params=True,
                    hpo_dir=hpo_dir,
                )
            else:
                # Fine-tune with specified configuration
                finetune(
                    data_dir=data_dir,
                    output_dir=model_dir,
                    config_path=config_path,
                    model_name=model_name,
                )

        if "evaluate" in steps:
            test_data = os.path.join(data_dir, "test.jsonl")
            evaluate(model_dir, test_data, eval_dir, config_path)

        if "deploy" in steps:
            deploy(model_dir, config_path)

        # Export all data to GCS if requested
        if export_to_gcs and "export" in steps:
            export_to_gcs(output_dir, config_path)
        
        logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    app()
