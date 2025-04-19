# TRAINING_TASK.md - Gemini Model Fine-tuning Plan

## Overview

This document outlines the specific steps for implementing an efficient fine-tuning pipeline for Gemini models on Vertex AI. We'll focus on Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) to optimize both performance and cost.

## Target Timeline
- **Training Script Improvements**: 2 days
- **Hyperparameter Optimization**: 2 days
- **Vertex AI Integration**: 2 days
- **Evaluation Framework**: 1 day

## Prerequisites

- Training data in `data/training_data` directory (JSONL format)
- GCP project with Vertex AI API enabled
- Service account with appropriate permissions
- GPU quotas on Vertex AI

## Training Tasks

### 1. Enhance Training Scripts

#### Task 1.1: Improve PEFT Setup
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/fine_tuning/setup.py`
- **Function:** `setup_peft_model()`
- **Task:** Optimize PEFT configuration for Gemini models

```python
def setup_peft_model(model, config):
    """Set up PEFT model with LoRA for fine-tuning."""
    from peft import LoraConfig, TaskType, get_peft_model
    
    # Create LoRA configuration
    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Get PEFT model
    peft_model = get_peft_model(model, peft_config)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")
    
    return peft_model
```

#### Task 1.2: Update Training Loop
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/fine_tuning/train.py`
- **Function:** `finetune_model()`
- **Task:** Enhance training loop with checkpoint saving and gradient accumulation

```python
# Example command to run training
python -m src.main finetune data/training_data data/model --config-path config/modified_config.yaml
```

### 2. Implement Hyperparameter Optimization

#### Task 2.1: Create Optimization Framework
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/fine_tuning/hyperparameter.py`
- **Function:** `optimize_hyperparameters()`
- **Task:** Implement Optuna-based hyperparameter optimization

```python
def optimize_hyperparameters(data_dir, output_dir, config, n_trials=20, timeout=None):
    """Optimize hyperparameters for fine-tuning."""
    import optuna
    
    # Define optimization objective
    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
        
        # Create configuration
        trial_config = create_config_from_params(config, {
            "learning_rate": lr,
            "batch_size": batch_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "weight_decay": weight_decay,
        })
        
        # Fine-tune with trial configuration
        model_dir = os.path.join(output_dir, f"trial_{trial.number}")
        finetune_model(data_dir, model_dir, trial_config)
        
        # Evaluate and return metric
        eval_metric = evaluate_model(model_dir, os.path.join(data_dir, "val.jsonl"))
        
        return eval_metric
    
    # Create study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters
    best_params = study.best_params
    
    # Fine-tune with best parameters
    best_config = create_config_from_params(config, best_params)
    
    return best_params, best_config
```

#### Task 2.2: Implement Best Parameters Training
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/fine_tuning/hyperparameter.py`
- **Function:** `finetune_with_best_params()`
- **Task:** Create function to train with optimized parameters

### 3. Vertex AI Integration

#### Task 3.1: Create Vertex AI Training Job
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/fine_tuning/vertex.py`
- **Function:** `create_training_job()`
- **Task:** Set up Vertex AI custom training job for fine-tuning

```python
def create_training_job(data_dir, output_dir, config):
    """Create Vertex AI custom training job."""
    from google.cloud import aiplatform
    
    # Initialize Vertex AI
    aiplatform.init(
        project=config.gcp.project_id,
        location=config.gcp.region,
    )
    
    # Create custom training job
    job = aiplatform.CustomTrainingJob(
        display_name=f"{config.project.name}-training",
        script_path="scripts/vertex_train.py",
        container_uri=config.vertex_ai.container_uri,
        requirements=["peft", "transformers", "torch", "datasets"],
    )
    
    # Start training job
    model = job.run(
        args=[
            f"--data_dir={data_dir}",
            f"--output_dir={output_dir}",
            f"--config_path={config.config_path}",
        ],
        replica_count=1,
        machine_type=config.vertex_ai.machine_type,
        accelerator_type=config.vertex_ai.accelerator_type,
        accelerator_count=config.vertex_ai.accelerator_count,
    )
    
    return model
```

#### Task 3.2: Create Training Script for Vertex AI
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `scripts/vertex_train.py`
- **Task:** Create a self-contained training script for Vertex AI

### 4. Evaluation Framework

#### Task 4.1: Implement Automated Evaluation
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/evaluation/metrics.py`
- **Function:** `evaluate_model()`
- **Task:** Create comprehensive evaluation metrics for fine-tuned models

```python
def evaluate_model(model_dir, test_data, output_dir=None):
    """Evaluate fine-tuned model on test data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import json
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load test data
    with open(test_data, "r") as f:
        test_examples = [json.loads(line) for line in f]
    
    # Generate predictions
    predictions = []
    metrics = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bleu": 0.0,
    }
    
    for example in test_examples:
        input_text = example["input_text"]
        reference = example["output_text"]
        
        # Generate prediction
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Calculate metrics
        # [Metrics calculation code here]
        
        predictions.append({
            "input": input_text,
            "output": reference,
            "prediction": prediction,
        })
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2)
        
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics
```

#### Task 4.2: Create Comparison Framework
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/evaluation/comparison.py`
- **Function:** `compare_models()`
- **Task:** Create framework to compare baseline and fine-tuned models

## Technical Implementation Details

### Fine-tuning Configuration

Key hyperparameters to experiment with:

1. **LoRA Configuration**:
   - `r`: Rank of the low-rank matrices (8, 16, 32)
   - `alpha`: Scaling factor (16, 32, 64)
   - `target_modules`: Which layers to fine-tune (attention layers)

2. **Training Parameters**:
   - Learning rate (1e-5 to 5e-5)
   - Batch size (4, 8, 16)
   - Training epochs (2-5)
   - Weight decay (0.01 to 0.1)

Example configuration:

```yaml
fine_tuning:
  model: gemini-pro  # Will be implemented with gpt2 for local testing
  method: peft
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["c_attn", "c_proj"]  # For gpt2
    bias: "none"
  training:
    batch_size: 8
    learning_rate: 1.0e-5
    epochs: 3
    warmup_steps: 100
    weight_decay: 0.01
    gradient_accumulation_steps: 4
    max_grad_norm: 1.0
    fp16: true
```

### Vertex AI Integration

To run training on Vertex AI, we need to:

1. **Package the training code**:
   - Create a self-contained training script
   - Ensure all dependencies are specified

2. **Configure compute resources**:
   - Machine type: n1-standard-8
   - Accelerator: NVIDIA_TESLA_T4 or NVIDIA_TESLA_V100
   - Accelerator count: 1-2 depending on dataset size

3. **Data management**:
   - Upload training data to GCS
   - Configure output location in GCS

Example Vertex AI script:

```python
def run_vertex_pipeline(
    pipeline_path,
    pdf_dir,
    output_dir,
    config_path,
    project_id,
    region,
    service_account=None,
    enable_caching=True,
):
    """Run a Vertex AI Pipeline."""
    from google.cloud import aiplatform
    
    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=region,
    )
    
    # Import pipeline
    from kfp.v2 import compiler
    from kfp.v2.google.client import AIPlatformClient
    
    # Create pipeline job
    job_name = f"gemini-finetuning-{int(time.time())}"
    
    # Create client
    api_client = AIPlatformClient(
        project_id=project_id,
        region=region,
    )
    
    # Run pipeline
    job = api_client.create_run_from_job_spec(
        job_spec_path=pipeline_path,
        parameter_values={
            "pdf_dir": pdf_dir,
            "output_dir": output_dir,
            "config_path": config_path,
        },
        service_account=service_account,
        enable_caching=enable_caching,
    )
    
    return job_name
```

## Monitoring Training

During training, we need to track:

1. **Loss curves**: Training and validation loss
2. **Resource utilization**: CPU, memory, and GPU usage
3. **Training speed**: Examples per second, time per epoch
4. **Checkpoints**: Save intermediate models

Example monitoring setup:

```python
def setup_training_monitoring(job_name, project_id):
    """Set up monitoring for a training job."""
    from google.cloud import monitoring_v3
    
    # Create client
    client = monitoring_v3.MetricServiceClient()
    
    # Create metric descriptors
    # [Implementation details]
    
    return client
```

## Success Criteria

1. Successfully fine-tune Gemini model with LoRA
2. Evaluation metrics show improvement over baseline
3. Training completes within budget constraints
4. Model successfully exported and ready for deployment

## Notes for Development Team

- Start with small experiments to validate the pipeline
- GPU memory is the primary constraint - optimize batch size accordingly
- Log all hyperparameters and results for reproducibility
- Set up early stopping to avoid overfitting
- Consider cost tradeoffs when selecting GPU types 