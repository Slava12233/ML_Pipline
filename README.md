# Gemini PDF Fine-tuning Pipeline

A streamlined pipeline for fine-tuning Google's Gemini models on PDF documentation.

## Project Overview

This project implements an end-to-end pipeline for fine-tuning Gemini models on PDF documentation. The pipeline extracts text from PDFs, processes it into training examples, fine-tunes Gemini models using Parameter-Efficient Fine-Tuning (PEFT) with LoRA, evaluates the models, and deploys them for inference.

## Key Features

- PDF text extraction with PyPDF2 and Document AI
- Semantic chunking and training data generation
- Fine-tuning Gemini models with PEFT/LoRA on Vertex AI
- Automated evaluation and deployment
- Comprehensive monitoring and logging

## Project Structure

```
gemini-pdf-finetuning/
├── docs/                    # Documentation
├── src/                     # Source code
│   ├── pdf_processing/      # PDF extraction and processing
│   ├── data_preparation/    # Training data generation
│   ├── fine_tuning/         # Model fine-tuning
│   ├── evaluation/          # Model evaluation
│   ├── deployment/          # Model deployment
│   └── utils/               # Shared utilities
├── tests/                   # Test suite
├── config/                  # Configuration files
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/                 # Utility scripts
├── PLANNING.md              # Project planning document
├── TASK.md                  # Task breakdown
└── README.md                # This file
```

## Getting Started

1. Review the [PLANNING.md](PLANNING.md) document for project architecture and technical decisions
2. Check the [TASK.md](TASK.md) file for the detailed task breakdown and timeline
3. Set up your GCP project and development environment
4. Follow the implementation tasks in order

## Prerequisites

- Python 3.10+
- Google Cloud Platform account with Vertex AI access
- PyTorch 2.0+
- Access to Gemini models via Vertex AI

## Documentation

For detailed documentation on each component, refer to the `docs/` directory.

## Project Timeline

- **Phase 1**: Infrastructure & PDF Processing (Weeks 1-2)
- **Phase 2**: Training Data Preparation (Weeks 3-4)
- **Phase 3**: Fine-tuning Implementation (Weeks 5-6)
- **Phase 4**: Evaluation & Deployment (Weeks 7-8)

## License

[Specify your license here]

## Contact

[Your contact information]
