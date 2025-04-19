"""
Tests for the data quality reporting module.
"""

import json
import os
from pathlib import Path
import tempfile

import pytest
import pandas as pd
import numpy as np

from src.data_preparation import quality


@pytest.fixture
def sample_jsonl_content():
    """Sample JSONL content for testing."""
    return [
        {"input_text": "What is machine learning?", "output_text": "Machine learning is a branch of artificial intelligence that focuses on developing systems that can learn from data."},
        {"input_text": "Explain neural networks", "output_text": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains."},
        {"input_text": "What is deep learning?", "output_text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers."},
    ]


@pytest.fixture
def sample_jsonl_file(sample_jsonl_content, tmp_path):
    """Create a sample JSONL file for testing."""
    file_path = tmp_path / "sample.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in sample_jsonl_content:
            f.write(json.dumps(item) + "\n")
    return file_path


@pytest.fixture
def sample_data_dir(sample_jsonl_content, tmp_path):
    """Create a sample data directory with train, val, and test files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create train, val, and test files
    splits = {
        "train": sample_jsonl_content[:2],
        "val": sample_jsonl_content[2:3],
        "test": sample_jsonl_content[1:2],
    }
    
    for split, content in splits.items():
        file_path = data_dir / f"{split}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for item in content:
                f.write(json.dumps(item) + "\n")
    
    return data_dir


def test_calculate_basic_stats():
    """Test calculating basic statistics."""
    # Test with empty list
    stats = quality.calculate_basic_stats([])
    assert stats["count"] == 0
    assert stats["min"] == 0
    assert stats["max"] == 0
    assert stats["mean"] == 0
    assert stats["median"] == 0
    assert stats["std"] == 0
    
    # Test with values
    values = [1, 2, 3, 4, 5]
    stats = quality.calculate_basic_stats(values)
    assert stats["count"] == 5
    assert stats["min"] == 1
    assert stats["max"] == 5
    assert stats["mean"] == 3
    assert stats["median"] == 3
    assert stats["std"] > 0


def test_analyze_jsonl_file(sample_jsonl_file):
    """Test analyzing a JSONL file."""
    # Analyze file
    stats = quality.analyze_jsonl_file(sample_jsonl_file)
    
    # Check that stats were calculated
    assert "input_length" in stats
    assert "output_length" in stats
    
    # Check input length stats
    assert stats["input_length"]["count"] == 3
    assert stats["input_length"]["min"] > 0
    assert stats["input_length"]["max"] > 0
    assert stats["input_length"]["mean"] > 0
    assert stats["input_length"]["median"] > 0
    assert stats["input_length"]["std"] >= 0
    
    # Check output length stats
    assert stats["output_length"]["count"] == 3
    assert stats["output_length"]["min"] > 0
    assert stats["output_length"]["max"] > 0
    assert stats["output_length"]["mean"] > 0
    assert stats["output_length"]["median"] > 0
    assert stats["output_length"]["std"] >= 0


def test_analyze_training_data(sample_data_dir):
    """Test analyzing training data."""
    # Analyze data
    stats = quality.analyze_training_data(sample_data_dir)
    
    # Check that stats were calculated for each split
    assert "train" in stats
    assert "val" in stats
    assert "test" in stats
    
    # Check train stats
    assert "input_length" in stats["train"]
    assert "output_length" in stats["train"]
    assert stats["train"]["input_length"]["count"] == 2
    
    # Check val stats
    assert "input_length" in stats["val"]
    assert "output_length" in stats["val"]
    assert stats["val"]["input_length"]["count"] == 1
    
    # Check test stats
    assert "input_length" in stats["test"]
    assert "output_length" in stats["test"]
    assert stats["test"]["input_length"]["count"] == 1


def test_calculate_quality_metrics(sample_data_dir):
    """Test calculating quality metrics."""
    # Analyze data
    stats = quality.analyze_training_data(sample_data_dir)
    
    # Calculate metrics
    metrics = quality.calculate_quality_metrics(stats)
    
    # Check that metrics were calculated
    assert "input_output_ratio" in metrics
    assert "split_balance" in metrics
    assert "length_variability" in metrics
    assert "overall_quality" in metrics
    
    # Check that metrics are within expected ranges
    assert metrics["input_output_ratio"] > 0
    assert 0 <= metrics["split_balance"] <= 1
    assert metrics["length_variability"] >= 0
    assert 0 <= metrics["overall_quality"] <= 1


def test_generate_quality_report(sample_data_dir, tmp_path):
    """Test generating a quality report."""
    # Generate report
    report = quality.generate_quality_report(
        data_dir=sample_data_dir,
        output_dir=tmp_path,
        include_plots=True,
    )
    
    # Check that report was generated
    assert "report" in report
    assert "report_path" in report
    
    # Check that report file exists
    report_path = Path(report["report_path"])
    assert report_path.exists()
    
    # Check that report contains expected fields
    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
    
    assert "stats" in report_data
    assert "metrics" in report_data
    
    # Check that plots were generated
    if "plots" in report_data:
        plots_dir = tmp_path / "plots"
        assert plots_dir.exists()
        
        # Check that at least one plot was generated
        assert len(list(plots_dir.glob("*.png"))) > 0


def test_generate_html_report(sample_data_dir, tmp_path):
    """Test generating an HTML report."""
    # Generate quality report
    report = quality.generate_quality_report(
        data_dir=sample_data_dir,
        output_dir=tmp_path,
        include_plots=True,
    )
    
    # Generate HTML report
    html_path = quality.generate_html_report(
        report=report,
        output_dir=tmp_path,
    )
    
    # Check that HTML report was generated
    assert Path(html_path).exists()
    
    # Check that HTML report contains expected content
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    assert "Training Data Quality Report" in html_content
    assert "Quality Metrics" in html_content
    assert "Statistics by Split" in html_content


def test_get_rating():
    """Test getting a rating class."""
    # Test good rating
    assert quality.get_rating(0.9, 0.5, 0.8) == "good"
    
    # Test medium rating
    assert quality.get_rating(0.6, 0.5, 0.8) == "medium"
    
    # Test poor rating
    assert quality.get_rating(0.3, 0.5, 0.8) == "poor"


def test_get_rating_text():
    """Test getting a rating text."""
    # Test good rating
    assert quality.get_rating_text(0.9, 0.5, 0.8) == "Good"
    
    # Test medium rating
    assert quality.get_rating_text(0.6, 0.5, 0.8) == "Medium"
    
    # Test poor rating
    assert quality.get_rating_text(0.3, 0.5, 0.8) == "Poor"


def test_analyze_jsonl_file_with_invalid_path():
    """Test analyzing a JSONL file with an invalid path."""
    with pytest.raises(FileNotFoundError):
        quality.analyze_jsonl_file("nonexistent.jsonl")


def test_analyze_training_data_with_invalid_path():
    """Test analyzing training data with an invalid path."""
    with pytest.raises(FileNotFoundError):
        quality.analyze_training_data("nonexistent_dir")


def test_plot_length_distributions(sample_data_dir, tmp_path):
    """Test plotting length distributions."""
    # Analyze data
    stats = quality.analyze_training_data(sample_data_dir)
    
    # Plot distributions
    plot_paths = quality.plot_length_distributions(stats, tmp_path)
    
    # Check that plots were generated
    assert "mean_lengths" in plot_paths
    assert "split_stats" in plot_paths
    
    # Check that plot files exist
    assert Path(plot_paths["mean_lengths"]).exists()
    assert Path(plot_paths["split_stats"]).exists()
