"""
Tests for evaluation modules.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.evaluation.metrics import (
    load_test_data,
    calculate_text_metrics,
    calculate_classification_metrics,
    calculate_qa_metrics,
    calculate_metrics,
    generate_evaluation_report,
)
from src.evaluation.human import (
    HumanEvaluationTask,
    HumanEvaluation,
    create_evaluation_criteria,
    create_evaluation_task_from_example,
    analyze_evaluations,
)
from src.evaluation.compare import (
    compare_model_metrics,
    calculate_improvement,
    compare_model_outputs,
)
from src.evaluation.report import (
    generate_evaluation_summary,
    generate_markdown_report,
)


@pytest.fixture
def test_data():
    """Create test data."""
    return [
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "prediction": "Paris is the capital of France.",
        },
        {
            "input": "Who wrote Romeo and Juliet?",
            "output": "William Shakespeare wrote Romeo and Juliet.",
            "prediction": "Romeo and Juliet was written by William Shakespeare.",
        },
        {
            "input": "What is the largest planet in our solar system?",
            "output": "Jupiter is the largest planet in our solar system.",
            "prediction": "The largest planet in our solar system is Jupiter.",
        },
    ]


@pytest.fixture
def classification_data():
    """Create classification test data."""
    return [
        {
            "input": "I love this product!",
            "output": "positive",
            "prediction": "positive",
            "label": "positive",
        },
        {
            "input": "This is terrible.",
            "output": "negative",
            "prediction": "negative",
            "label": "negative",
        },
        {
            "input": "It's okay, nothing special.",
            "output": "neutral",
            "prediction": "positive",
            "label": "neutral",
        },
    ]


@pytest.fixture
def qa_data():
    """Create QA test data."""
    return [
        {
            "input": "What is the capital of France?",
            "output": "Paris",
            "prediction": "Paris",
        },
        {
            "input": "Who wrote Romeo and Juliet?",
            "output": "William Shakespeare",
            "prediction": "Shakespeare",
        },
        {
            "input": "What is the largest planet in our solar system?",
            "output": "Jupiter",
            "prediction": "Jupiter is the largest planet",
        },
    ]


@pytest.fixture
def model_predictions():
    """Create model predictions."""
    return {
        "baseline": [
            {
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris.",
                "prediction": "I think it's Paris.",
            },
            {
                "input": "Who wrote Romeo and Juliet?",
                "output": "William Shakespeare wrote Romeo and Juliet.",
                "prediction": "Shakespeare wrote that play.",
            },
        ],
        "fine-tuned": [
            {
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris.",
                "prediction": "The capital of France is Paris.",
            },
            {
                "input": "Who wrote Romeo and Juliet?",
                "output": "William Shakespeare wrote Romeo and Juliet.",
                "prediction": "William Shakespeare wrote Romeo and Juliet.",
            },
        ],
    }


@pytest.fixture
def human_evaluation_tasks():
    """Create human evaluation tasks."""
    return [
        HumanEvaluationTask(
            task_id="task1",
            input_text="What is the capital of France?",
            reference_output="The capital of France is Paris.",
            model_outputs={
                "baseline": "I think it's Paris.",
                "fine-tuned": "The capital of France is Paris.",
            },
            evaluation_criteria=[
                create_evaluation_criteria(
                    criteria_type="likert",
                    name="accuracy",
                    description="How accurate is the response?",
                    min_value=1,
                    max_value=5,
                ),
                create_evaluation_criteria(
                    criteria_type="binary",
                    name="preference",
                    description="Which model do you prefer?",
                    options=["baseline", "fine-tuned"],
                ),
            ],
        ),
        HumanEvaluationTask(
            task_id="task2",
            input_text="Who wrote Romeo and Juliet?",
            reference_output="William Shakespeare wrote Romeo and Juliet.",
            model_outputs={
                "baseline": "Shakespeare wrote that play.",
                "fine-tuned": "William Shakespeare wrote Romeo and Juliet.",
            },
            evaluation_criteria=[
                create_evaluation_criteria(
                    criteria_type="likert",
                    name="accuracy",
                    description="How accurate is the response?",
                    min_value=1,
                    max_value=5,
                ),
                create_evaluation_criteria(
                    criteria_type="binary",
                    name="preference",
                    description="Which model do you prefer?",
                    options=["baseline", "fine-tuned"],
                ),
            ],
        ),
    ]


@pytest.fixture
def human_evaluations():
    """Create human evaluations."""
    return [
        HumanEvaluation(
            evaluation_id="eval1",
            task_id="task1",
            evaluator_id="evaluator1",
            ratings={
                "baseline": {
                    "accuracy": 3,
                },
                "fine-tuned": {
                    "accuracy": 5,
                },
                "preference": "fine-tuned",
            },
            comments="The fine-tuned model is more accurate.",
        ),
        HumanEvaluation(
            evaluation_id="eval2",
            task_id="task2",
            evaluator_id="evaluator1",
            ratings={
                "baseline": {
                    "accuracy": 4,
                },
                "fine-tuned": {
                    "accuracy": 5,
                },
                "preference": "fine-tuned",
            },
            comments="Both are good, but fine-tuned is better.",
        ),
    ]


def test_load_test_data():
    """Test loading test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data file
        test_data_path = Path(temp_dir) / "test_data.jsonl"
        with open(test_data_path, "w") as f:
            f.write('{"input": "test input", "output": "test output"}\n')
            f.write('{"input": "another input", "output": "another output"}\n')
        
        # Load test data
        test_data = load_test_data(test_data_path)
        
        # Check test data
        assert len(test_data) == 2
        assert test_data[0]["input"] == "test input"
        assert test_data[0]["output"] == "test output"
        assert test_data[1]["input"] == "another input"
        assert test_data[1]["output"] == "another output"


def test_calculate_text_metrics(test_data):
    """Test calculating text metrics."""
    # Calculate text metrics
    metrics = calculate_text_metrics(test_data, metrics=["rouge", "bleu"])
    
    # Check metrics
    assert "rouge1" in metrics
    assert "rouge2" in metrics
    assert "rougeL" in metrics
    assert "bleu" in metrics
    
    # Check metric values
    assert 0 <= metrics["rouge1"] <= 1
    assert 0 <= metrics["rouge2"] <= 1
    assert 0 <= metrics["rougeL"] <= 1
    assert 0 <= metrics["bleu"] <= 1


def test_calculate_classification_metrics(classification_data):
    """Test calculating classification metrics."""
    # Calculate classification metrics
    metrics = calculate_classification_metrics(classification_data)
    
    # Check metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    
    # Check metric values
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    
    # Check accuracy value (2/3 correct)
    assert metrics["accuracy"] == pytest.approx(2/3)


def test_calculate_qa_metrics(qa_data):
    """Test calculating QA metrics."""
    # Calculate QA metrics
    metrics = calculate_qa_metrics(qa_data)
    
    # Check metrics
    assert "exact_match" in metrics
    assert "contains_answer" in metrics
    
    # Check metric values
    assert 0 <= metrics["exact_match"] <= 1
    assert 0 <= metrics["contains_answer"] <= 1
    
    # Check exact match value (1/3 exact matches)
    assert metrics["exact_match"] == pytest.approx(1/3)
    
    # Check contains answer value (3/3 contain answer)
    assert metrics["contains_answer"] == pytest.approx(1.0)


def test_calculate_metrics(test_data, classification_data, qa_data):
    """Test calculating metrics for different task types."""
    # Calculate general metrics
    general_metrics = calculate_metrics(test_data, metrics=["rouge", "bleu"], task_type="general")
    
    # Check general metrics
    assert "rouge1" in general_metrics
    assert "rouge2" in general_metrics
    assert "rougeL" in general_metrics
    assert "bleu" in general_metrics
    
    # Calculate classification metrics
    classification_metrics = calculate_metrics(classification_data, metrics=["rouge", "bleu"], task_type="classification")
    
    # Check classification metrics
    assert "rouge1" in classification_metrics
    assert "rouge2" in classification_metrics
    assert "rougeL" in classification_metrics
    assert "bleu" in classification_metrics
    assert "accuracy" in classification_metrics
    assert "precision" in classification_metrics
    assert "recall" in classification_metrics
    assert "f1" in classification_metrics
    
    # Calculate QA metrics
    qa_metrics = calculate_metrics(qa_data, metrics=["rouge", "bleu"], task_type="qa")
    
    # Check QA metrics
    assert "rouge1" in qa_metrics
    assert "rouge2" in qa_metrics
    assert "rougeL" in qa_metrics
    assert "bleu" in qa_metrics
    assert "exact_match" in qa_metrics
    assert "contains_answer" in qa_metrics


def test_generate_evaluation_report(test_data):
    """Test generating evaluation report."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Calculate metrics
        metrics = calculate_metrics(test_data, metrics=["rouge", "bleu"], task_type="general")
        
        # Generate evaluation report
        report_dir = generate_evaluation_report(metrics, test_data, temp_dir)
        
        # Check report directory
        report_dir = Path(report_dir)
        assert report_dir.exists()
        
        # Check report files
        assert (report_dir / "metrics.json").exists()
        assert (report_dir / "predictions.jsonl").exists()
        assert (report_dir / "metrics_chart.png").exists()


def test_create_evaluation_criteria():
    """Test creating evaluation criteria."""
    # Create likert criteria
    likert_criteria = create_evaluation_criteria(
        criteria_type="likert",
        name="accuracy",
        description="How accurate is the response?",
        min_value=1,
        max_value=5,
    )
    
    # Check likert criteria
    assert likert_criteria["type"] == "likert"
    assert likert_criteria["name"] == "accuracy"
    assert likert_criteria["description"] == "How accurate is the response?"
    assert likert_criteria["min_value"] == 1
    assert likert_criteria["max_value"] == 5
    assert likert_criteria["required"] is True
    
    # Create binary criteria
    binary_criteria = create_evaluation_criteria(
        criteria_type="binary",
        name="preference",
        description="Which model do you prefer?",
        options=["baseline", "fine-tuned"],
    )
    
    # Check binary criteria
    assert binary_criteria["type"] == "binary"
    assert binary_criteria["name"] == "preference"
    assert binary_criteria["description"] == "Which model do you prefer?"
    assert binary_criteria["options"] == ["baseline", "fine-tuned"]
    assert binary_criteria["required"] is True


def test_create_evaluation_task_from_example():
    """Test creating evaluation task from example."""
    # Create example
    example = {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris.",
    }
    
    # Create model outputs
    model_outputs = {
        "baseline": "I think it's Paris.",
        "fine-tuned": "The capital of France is Paris.",
    }
    
    # Create evaluation task
    task = create_evaluation_task_from_example(example, model_outputs)
    
    # Check task
    assert task.input_text == "What is the capital of France?"
    assert task.reference_output == "The capital of France is Paris."
    assert task.model_outputs == model_outputs
    assert len(task.evaluation_criteria) == 4
    assert task.evaluation_criteria[0]["name"] == "relevance"
    assert task.evaluation_criteria[1]["name"] == "accuracy"
    assert task.evaluation_criteria[2]["name"] == "fluency"
    assert task.evaluation_criteria[3]["name"] == "preference"


def test_analyze_evaluations(human_evaluations, human_evaluation_tasks):
    """Test analyzing evaluations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Analyze evaluations
        summary = analyze_evaluations(human_evaluations, human_evaluation_tasks, temp_dir)
        
        # Check summary
        assert "mean" in summary
        assert "median" in summary
        assert "std" in summary
        assert "overall" in summary
        
        # Check mean ratings
        assert "baseline" in summary["mean"].index
        assert "fine-tuned" in summary["mean"].index
        assert "accuracy" in summary["mean"].columns
        
        # Check overall scores
        assert "baseline" in summary["overall"].index
        assert "fine-tuned" in summary["overall"].index
        
        # Check preferences
        assert "preferences" in summary
        assert "fine-tuned" in summary["preferences"].index
        assert summary["preferences"]["fine-tuned"] == 2


def test_compare_model_metrics(model_predictions):
    """Test comparing model metrics."""
    # Compare model metrics
    metrics = compare_model_metrics(model_predictions, metrics=["rouge", "bleu"])
    
    # Check metrics
    assert "baseline" in metrics
    assert "fine-tuned" in metrics
    assert "rouge1" in metrics["baseline"]
    assert "rouge2" in metrics["baseline"]
    assert "rougeL" in metrics["baseline"]
    assert "bleu" in metrics["baseline"]
    assert "rouge1" in metrics["fine-tuned"]
    assert "rouge2" in metrics["fine-tuned"]
    assert "rougeL" in metrics["fine-tuned"]
    assert "bleu" in metrics["fine-tuned"]


def test_calculate_improvement():
    """Test calculating improvement."""
    # Create baseline and fine-tuned metrics
    baseline_metrics = {
        "rouge1": 0.5,
        "rouge2": 0.3,
        "rougeL": 0.4,
        "bleu": 0.2,
    }
    
    finetuned_metrics = {
        "rouge1": 0.7,
        "rouge2": 0.5,
        "rougeL": 0.6,
        "bleu": 0.4,
    }
    
    # Calculate improvement
    improvement = calculate_improvement(baseline_metrics, finetuned_metrics)
    
    # Check improvement
    assert "absolute" in improvement
    assert "relative" in improvement
    assert "rouge1" in improvement["absolute"]
    assert "rouge2" in improvement["absolute"]
    assert "rougeL" in improvement["absolute"]
    assert "bleu" in improvement["absolute"]
    assert "rouge1" in improvement["relative"]
    assert "rouge2" in improvement["relative"]
    assert "rougeL" in improvement["relative"]
    assert "bleu" in improvement["relative"]
    
    # Check absolute improvement values
    assert improvement["absolute"]["rouge1"] == 0.2
    assert improvement["absolute"]["rouge2"] == 0.2
    assert improvement["absolute"]["rougeL"] == 0.2
    assert improvement["absolute"]["bleu"] == 0.2
    
    # Check relative improvement values
    assert improvement["relative"]["rouge1"] == 40.0
    assert improvement["relative"]["rouge2"] == pytest.approx(66.67, rel=1e-2)
    assert improvement["relative"]["rougeL"] == 50.0
    assert improvement["relative"]["bleu"] == 100.0


def test_compare_model_outputs():
    """Test comparing model outputs."""
    # Create baseline and fine-tuned outputs
    baseline_outputs = [
        "I think the capital of France is Paris.",
        "Shakespeare wrote Romeo and Juliet.",
    ]
    
    finetuned_outputs = [
        "The capital of France is Paris.",
        "William Shakespeare wrote Romeo and Juliet.",
    ]
    
    reference_outputs = [
        "The capital of France is Paris.",
        "William Shakespeare wrote Romeo and Juliet.",
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Compare model outputs
        stats = compare_model_outputs(
            baseline_outputs,
            finetuned_outputs,
            reference_outputs,
            temp_dir,
        )
        
        # Check stats
        assert "baseline" in stats
        assert "fine-tuned" in stats
        assert "exact_match" in stats
        assert "exact_match_percentage" in stats
        
        # Check baseline stats
        assert "mean_length" in stats["baseline"]
        assert "median_length" in stats["baseline"]
        assert "min_length" in stats["baseline"]
        assert "max_length" in stats["baseline"]
        assert "reference_match" in stats["baseline"]
        assert "reference_match_percentage" in stats["baseline"]
        
        # Check fine-tuned stats
        assert "mean_length" in stats["fine-tuned"]
        assert "median_length" in stats["fine-tuned"]
        assert "min_length" in stats["fine-tuned"]
        assert "max_length" in stats["fine-tuned"]
        assert "reference_match" in stats["fine-tuned"]
        assert "reference_match_percentage" in stats["fine-tuned"]
        
        # Check exact match
        assert stats["exact_match"] == 0
        assert stats["exact_match_percentage"] == 0.0
        
        # Check reference match
        assert stats["baseline"]["reference_match"] == 0
        assert stats["baseline"]["reference_match_percentage"] == 0.0
        assert stats["fine-tuned"]["reference_match"] == 2
        assert stats["fine-tuned"]["reference_match_percentage"] == 100.0


def test_generate_evaluation_summary():
    """Test generating evaluation summary."""
    # Create results
    results = {
        "metrics": {
            "rouge1": 0.7,
            "rouge2": 0.5,
            "rougeL": 0.6,
            "bleu": 0.4,
        },
        "predictions": [
            {
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris.",
                "prediction": "The capital of France is Paris.",
            },
            {
                "input": "Who wrote Romeo and Juliet?",
                "output": "William Shakespeare wrote Romeo and Juliet.",
                "prediction": "William Shakespeare wrote Romeo and Juliet.",
            },
        ],
        "comparison": {
            "metrics": [
                {
                    "Metric": "rouge1",
                    "baseline": 0.5,
                    "fine-tuned": 0.7,
                    "Absolute Improvement": 0.2,
                    "Relative Improvement (%)": 40.0,
                },
                {
                    "Metric": "rouge2",
                    "baseline": 0.3,
                    "fine-tuned": 0.5,
                    "Absolute Improvement": 0.2,
                    "Relative Improvement (%)": 66.67,
                },
            ],
        },
        "human_evaluation": {
            "summary": {
                "overall": {
                    "baseline": 3.5,
                    "fine-tuned": 5.0,
                },
            },
        },
    }
    
    # Generate summary
    summary = generate_evaluation_summary(results)
    
    # Check summary
    assert "metrics" in summary
    assert "prediction_stats" in summary
    assert "improvement" in summary
    assert "human_evaluation" in summary
    
    # Check metrics
    assert "rouge1" in summary["metrics"]
    assert "rouge2" in summary["metrics"]
    assert "rougeL" in summary["metrics"]
    assert "bleu" in summary["metrics"]
    
    # Check prediction stats
    assert "count" in summary["prediction_stats"]
    assert "input_length" in summary["prediction_stats"]
    assert "output_length" in summary["prediction_stats"]
    assert "prediction_length" in summary["prediction_stats"]
    
    # Check improvement
    assert "rouge1" in summary["improvement"]
    assert "rouge2" in summary["improvement"]
    
    # Check human evaluation
    assert "overall" in summary["human_evaluation"]


def test_generate_markdown_report():
    """Test generating Markdown report."""
    # Create results
    results = {
        "metrics": {
            "rouge1": 0.7,
            "rouge2": 0.5,
            "rougeL": 0.6,
            "bleu": 0.4,
        },
        "predictions": [
            {
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris.",
                "prediction": "The capital of France is Paris.",
            },
            {
                "input": "Who wrote Romeo and Juliet?",
                "output": "William Shakespeare wrote Romeo and Juliet.",
                "prediction": "William Shakespeare wrote Romeo and Juliet.",
            },
        ],
        "comparison": {
            "metrics": [
                {
                    "Metric": "rouge1",
                    "baseline": 0.5,
                    "fine-tuned": 0.7,
                    "Absolute Improvement": 0.2,
                    "Relative Improvement (%)": 40.0,
                },
                {
                    "Metric": "rouge2",
                    "baseline": 0.3,
                    "fine-tuned": 0.5,
                    "Absolute Improvement": 0.2,
                    "Relative Improvement (%)": 66.67,
                },
            ],
        },
        "human_evaluation": {
            "summary": {
                "mean": {
                    "baseline": {
                        "accuracy": 3.5,
                    },
                    "fine-tuned": {
                        "accuracy": 5.0,
                    },
                },
                "preferences": {
                    "baseline": 0,
                    "fine-tuned": 2,
                },
                "overall": {
                    "baseline": 3.5,
                    "fine-tuned": 5.0,
                },
            },
        },
    }
    
    # Generate summary
    summary = generate_evaluation_summary(results)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate Markdown report
        report_path = generate_markdown_report(results, summary, temp_dir)
        
        # Check report path
        report_path = Path(report_path)
        assert report_path.exists()
        
        # Check report content
        with open(report_path, "r") as f:
            content = f.read()
            
            # Check sections
            assert "# Evaluation Report" in content
            assert "## Executive Summary" in content
            assert "## Automated Metrics" in content
            assert "## Example Predictions" in content
            assert "## Human Evaluation" in content
