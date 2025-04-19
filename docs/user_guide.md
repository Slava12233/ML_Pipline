# Gemini PDF Fine-tuning Pipeline - User Guide

## Introduction

The Gemini PDF Fine-tuning Pipeline is a tool for fine-tuning Gemini models on PDF documentation. This guide will help you get started with the pipeline and show you how to use its various features.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Monitoring and Evaluation](#monitoring-and-evaluation)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Installation

### Prerequisites

Before installing the pipeline, make sure you have the following:

- Python 3.9 or higher
- pip (Python package installer)
- Google Cloud SDK (for GCP integration)
- A Google Cloud project with Vertex AI and Document AI APIs enabled (for cloud features)

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/your-organization/gemini-pdf-finetuning.git
cd gemini-pdf-finetuning
```

2. Install the package and its dependencies:

```bash
pip install -e .
```

3. Set up Google Cloud authentication:

```bash
gcloud auth application-default login
```

## Configuration

The pipeline is configured using a YAML file. A default configuration file is provided at `config/config.yaml`. You should modify this file to match your requirements.

### Configuration File Structure

The configuration file has the following sections:

1. **Project Configuration**: Basic project information.
2. **GCP Configuration**: Google Cloud Platform settings.
3. **PDF Processing Configuration**: Settings for PDF extraction.
4. **Training Data Configuration**: Settings for training data generation.
5. **Fine-tuning Configuration**: Settings for model fine-tuning.
6. **Evaluation Configuration**: Settings for model evaluation.
7. **Deployment Configuration**: Settings for model deployment.

### Example Configuration

```yaml
project:
  name: gemini-pdf-finetuning
  version: 0.1.0

gcp:
  project_id: your-project-id  # Replace with your GCP project ID
  region: us-central1
  storage:
    bucket_name: gemini-pdf-finetuning-data  # Replace with your GCS bucket name
    training_data_folder: training_data

pdf_processing:
  extraction_method: pypdf2  # Options: pypdf2, document_ai
  chunk_size: 1000
  chunk_overlap: 200

training_data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  min_quality_score: 0.7

fine_tuning:
  model_name: gemini-pro  # Options: gemini-pro, gemini-1.5-pro, gemini-1.5-flash
  method: peft  # Options: peft, full
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
  training:
    batch_size: 8
    learning_rate: 1e-5
    epochs: 3
  vertex_ai:
    project_id: your-project-id  # Replace with your GCP project ID
    region: us-central1

evaluation:
  metrics:
    - accuracy
    - f1
    - rouge

deployment:
  endpoint_name: gemini-pdf-finetuned
  min_replicas: 1
  max_replicas: 5
```

### Important Configuration Parameters

- **gcp.project_id**: Your Google Cloud project ID.
- **gcp.storage.bucket_name**: The name of your Google Cloud Storage bucket.
- **pdf_processing.extraction_method**: The method to use for PDF extraction.
- **fine_tuning.model_name**: The Gemini model to fine-tune.
- **fine_tuning.method**: The fine-tuning method to use.

## Basic Usage

The pipeline provides a command-line interface (CLI) for running the pipeline and its individual components. Here are some basic usage examples:

### Running the Complete Pipeline

To run the complete pipeline:

```bash
python -m src.main run_pipeline pdf_dir output_dir
```

This will:
1. Process PDFs from `pdf_dir`
2. Prepare training data
3. Fine-tune a Gemini model
4. Evaluate the fine-tuned model
5. Deploy the model to Vertex AI Endpoints

### Running Individual Steps

You can also run individual steps of the pipeline:

#### 1. Process PDFs

```bash
python -m src.main process_pdfs pdf_dir output_dir/extracted_text
```

#### 2. Prepare Training Data

```bash
python -m src.main prepare_data output_dir/extracted_text output_dir/training_data
```

#### 3. Fine-tune Model

```bash
python -m src.main finetune output_dir/training_data output_dir/model
```

#### 4. Evaluate Model

```bash
python -m src.main evaluate output_dir/model output_dir/training_data/test.jsonl output_dir/evaluation
```

#### 5. Deploy Model

```bash
python -m src.main deploy output_dir/model
```

## Advanced Usage

### Hyperparameter Optimization

To optimize hyperparameters for fine-tuning:

```bash
python -m src.main optimize_hyperparameters output_dir/training_data output_dir/hyperparameter_optimization
```

Then, to fine-tune with the best hyperparameters:

```bash
python -m src.main finetune output_dir/training_data output_dir/model --use_best_params --hpo_dir output_dir/hyperparameter_optimization
```

### Vertex AI Pipeline

To create and run a Vertex AI Pipeline:

```bash
# Create pipeline
python -m src.main create_pipeline gemini-pipeline output_dir/pipeline

# Run pipeline
python -m src.main run_vertex_pipeline output_dir/pipeline/gemini-pipeline.json pdf_dir output_dir
```

### Scheduled Pipeline Execution

To create a scheduled pipeline trigger:

```bash
python -m src.main create_pipeline_trigger output_dir/pipeline/gemini-pipeline.json daily-trigger "0 0 * * *" pdf_dir output_dir
```

This will run the pipeline every day at midnight.

### Data Quality Reporting

To generate a quality report for training data:

```bash
python -m src.main generate_report output_dir/training_data output_dir/quality_report
```

### Exporting Data to GCS

To export data to Google Cloud Storage:

```bash
python -m src.main export_to_gcs output_dir
```

## Monitoring and Evaluation

### Monitoring Training

During fine-tuning, you can monitor the training progress in the logs. The logs will show:

- Training loss
- Validation loss
- Learning rate
- Training time

### Evaluating Model Performance

After fine-tuning, you can evaluate the model's performance using the evaluation command:

```bash
python -m src.main evaluate output_dir/model output_dir/training_data/test.jsonl output_dir/evaluation
```

This will generate an evaluation report with metrics like:

- Accuracy
- F1 score
- ROUGE score

### Monitoring Deployed Models

Once deployed, you can monitor the model's performance using Google Cloud Console:

1. Go to the [Vertex AI section](https://console.cloud.google.com/vertex-ai) of Google Cloud Console.
2. Navigate to the "Endpoints" page.
3. Click on your endpoint to view its details.
4. Check the "Monitoring" tab to see metrics like:
   - Request rate
   - Latency
   - Error rate

## Troubleshooting

### Common Issues

#### PDF Extraction Failures

If PDF extraction fails:

1. Check if the PDF is corrupted or password-protected.
2. Try a different extraction method:

```bash
python -m src.main process_pdfs pdf_dir output_dir/extracted_text --method document_ai
```

#### Training Data Quality Issues

If training data quality is poor:

1. Generate a quality report:

```bash
python -m src.main generate_report output_dir/training_data output_dir/quality_report
```

2. Adjust quality thresholds in the configuration file.

#### Fine-tuning Failures

If fine-tuning fails:

1. Check if you have enough GPU memory.
2. Try reducing the batch size in the configuration file.
3. Check the logs for specific error messages.

#### Deployment Failures

If deployment fails:

1. Check if your service account has the necessary permissions.
2. Check if the model was saved correctly.
3. Check the logs for specific error messages.

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs for error messages.
2. Consult the [technical documentation](technical_documentation.md).
3. Contact the project maintainers.

## FAQ

### General Questions

#### Q: What is fine-tuning?

A: Fine-tuning is the process of adapting a pre-trained model to a specific task or domain by training it on a smaller, task-specific dataset.

#### Q: What is PEFT/LoRA?

A: PEFT (Parameter-Efficient Fine-Tuning) is a technique for fine-tuning large language models with fewer resources. LoRA (Low-Rank Adaptation) is a specific PEFT method that adds trainable low-rank matrices to the model's weights.

#### Q: How much data do I need for fine-tuning?

A: The amount of data needed depends on the complexity of the task and the desired performance. Generally, a few hundred to a few thousand examples can be sufficient for good results.

### Usage Questions

#### Q: Can I use this pipeline with models other than Gemini?

A: The pipeline is designed specifically for Gemini models, but it could be adapted for other models with some modifications.

#### Q: Can I run the pipeline without Google Cloud?

A: Yes, you can run most of the pipeline locally, but some features like Document AI extraction and Vertex AI deployment require Google Cloud.

#### Q: How long does fine-tuning take?

A: Fine-tuning time depends on the model size, dataset size, and hardware. With a GPU, it can take from a few hours to a few days.

### Technical Questions

#### Q: What is the difference between PEFT and full fine-tuning?

A: PEFT fine-tunes only a small subset of the model's parameters, which is more efficient but may be less effective for some tasks. Full fine-tuning updates all parameters, which can be more effective but requires more resources.

#### Q: How do I choose hyperparameters?

A: You can use the hyperparameter optimization feature to automatically find good hyperparameters:

```bash
python -m src.main optimize_hyperparameters output_dir/training_data output_dir/hyperparameter_optimization
```

#### Q: Can I use my own PDF extraction method?

A: Yes, you can implement your own extraction method by modifying the `pdf_processing/extract.py` file.
