# Fine-tuning Pipeline for PDF-Based Training

This repository contains a modular pipeline for fine-tuning language models on PDF content.

## Architecture Overview

```
                   ┌─────────────┐
                   │    PDFs     │
                   └──────┬──────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ PDF Processing  │
                 └────────┬────────┘
                          │
                          ▼
               ┌───────────────────┐
               │ Data Preparation  │
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │    Fine-tuning    │◄────┐
               └─────────┬─────────┘     │
                         │               │
                         ▼               │
               ┌───────────────────┐     │
               │    Evaluation     │─────┘
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │    Deployment     │
               └───────────────────┘
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fine-tuning-pipeline.git
cd fine-tuning-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
pip install -r requirements.txt
```

## Quick Start

Run the complete pipeline with one command:

```bash
python scripts/run.py pipeline --pdf-dir data/pdfs --env local
```

Or run individual steps:

```bash
# Process PDFs
python scripts/run.py pdf process data/pdfs data/extracted_text

# Prepare training data
python scripts/run.py data prepare data/extracted_text data/training_data

# Fine-tune the model
python scripts/run.py train finetune data/training_data data/model

# Evaluate the model
python scripts/run.py eval model data/model data/training_data/test.jsonl data/evaluation

# Test the model on a PDF
python scripts/run.py eval pdf data/pdfs/example.pdf --use-peft
```

## Configuration

The pipeline uses a hierarchical configuration system with environment-specific overrides:

- `config/config.yaml` - Base configuration
- `config/local_config.yaml` - Local development overrides
- `config/vertex_config.yaml` - Vertex AI environment overrides
- `config/production_config.yaml` - Production environment overrides

Specify the environment using the `--env` parameter:

```bash
python scripts/run.py pipeline --env production
```

## Pipeline Components

### 1. PDF Processing

The PDF processing component extracts text and metadata from PDF files using methods like PyPDF2 or Google Document AI.

```bash
python scripts/run.py pdf process data/pdfs data/extracted_text --method pypdf2
```

### 2. Data Preparation

The data preparation component generates training examples from the extracted text.

```bash
python scripts/run.py data prepare data/extracted_text data/training_data --quality-report
```

### 3. Fine-tuning

The fine-tuning component trains models using techniques like PEFT/LoRA for parameter-efficient adaptation.

```bash
python scripts/run.py train finetune data/training_data data/model --method peft
```

### 4. Evaluation

The evaluation component assesses model performance using metrics like ROUGE, BLEU, and BERTScore.

```bash
python scripts/run.py eval model data/model data/training_data/test.jsonl data/evaluation
```

### 5. Deployment

The deployment component handles model packaging and serving.

## Development

- Check the `OPTIMIZATIONS.md` file for details on recent code optimizations
- Task descriptions are available in the `docs/` directory
- See `PLANNING.md` for architectural decisions

## License

[MIT License]
