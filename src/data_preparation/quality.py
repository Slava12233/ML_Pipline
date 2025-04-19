"""
Data quality reporting module.

This module provides functions for generating quality reports for training data.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_basic_stats(values: List[int]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.

    Args:
        values: List of values.

    Returns:
        Dict[str, float]: Dictionary with statistics.
    """
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
        }
    
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
    }


def analyze_jsonl_file(file_path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Analyze a JSONL file with training examples.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary with statistics.
    """
    logger.info(f"Analyzing JSONL file: {file_path}")
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    
    # Initialize counters
    input_lengths = []
    output_lengths = []
    
    # Read JSONL file
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                example = json.loads(line)
                
                # Check if example has required fields
                if "input_text" in example and "output_text" in example:
                    input_lengths.append(len(example["input_text"]))
                    output_lengths.append(len(example["output_text"]))
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in file {file_path}")
                continue
    
    # Calculate statistics
    stats = {
        "input_length": calculate_basic_stats(input_lengths),
        "output_length": calculate_basic_stats(output_lengths),
    }
    
    return stats


def analyze_training_data(data_dir: Union[str, Path]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Analyze training data in a directory.

    Args:
        data_dir: Path to the directory containing training data.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Dictionary with statistics for each split.
    """
    logger.info(f"Analyzing training data in {data_dir}")
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")
    
    # Initialize results
    results = {}
    
    # Analyze each split
    for split in ["train", "val", "test"]:
        file_path = data_dir / f"{split}.jsonl"
        if file_path.exists():
            results[split] = analyze_jsonl_file(file_path)
    
    return results


def plot_length_distributions(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """
    Plot length distributions for training data.

    Args:
        stats: Statistics for each split.
        output_dir: Directory to save the plots.

    Returns:
        Dict[str, str]: Dictionary mapping plot types to file paths.
    """
    logger.info(f"Plotting length distributions to {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a DataFrame for easier plotting
    data = []
    for split, split_stats in stats.items():
        for field, field_stats in split_stats.items():
            data.append({
                "split": split,
                "field": field,
                "count": field_stats["count"],
                "min": field_stats["min"],
                "max": field_stats["max"],
                "mean": field_stats["mean"],
                "median": field_stats["median"],
                "std": field_stats["std"],
            })
    
    df = pd.DataFrame(data)
    
    # Plot mean lengths
    plt.figure(figsize=(10, 6))
    
    # Create a grouped bar chart
    bar_width = 0.35
    index = np.arange(len(df["split"].unique()))
    
    # Filter for input and output lengths
    input_df = df[df["field"] == "input_length"]
    output_df = df[df["field"] == "output_length"]
    
    # Make sure we have both input and output data
    if not input_df.empty and not output_df.empty:
        plt.bar(index, input_df["mean"], bar_width, label="Input Length")
        plt.bar(index + bar_width, output_df["mean"], bar_width, label="Output Length")
        
        plt.xlabel("Split")
        plt.ylabel("Mean Length (characters)")
        plt.title("Mean Input and Output Lengths by Split")
        # Fixed: Make sure the number of ticks matches the number of locations
        plt.xticks(index + bar_width / 2, list(input_df["split"]))
        plt.legend()
        
        # Save the plot
        mean_lengths_path = output_dir / "mean_lengths.png"
        plt.savefig(mean_lengths_path)
    else:
        mean_lengths_path = None
        
    plt.close()
    
    # Plot statistics for each split
    split_stats_paths = {}
    for split in df["split"].unique():
        plt.figure(figsize=(10, 6))
        
        # Filter for the current split
        split_df = df[df["split"] == split]
        
        # Get unique fields for this split
        unique_fields = split_df["field"].unique()
        
        if len(unique_fields) > 0:
            # Create a grouped bar chart for statistical measures
            field_positions = np.arange(len(unique_fields))
            
            # Get min, mean, and max values for all fields
            all_mins = []
            all_means = []
            all_maxes = []
            all_field_names = []
            
            for field in unique_fields:
                field_data = split_df[split_df["field"] == field]
                if not field_data.empty:
                    all_mins.append(field_data["min"].values[0])
                    all_means.append(field_data["mean"].values[0])
                    all_maxes.append(field_data["max"].values[0])
                    all_field_names.append(field)
            
            # Plot min, mean, and max if we have data
            if all_field_names:
                bar_width = 0.25
                plt.bar(field_positions, all_mins, bar_width, label="Min")
                plt.bar(field_positions + bar_width, all_means, bar_width, label="Mean")
                plt.bar(field_positions + 2 * bar_width, all_maxes, bar_width, label="Max")
                
                plt.xlabel("Field")
                plt.ylabel("Length (characters)")
                plt.title(f"Length Statistics for {split.capitalize()} Split")
                # Fixed: Make sure the number of ticks matches the number of locations
                plt.xticks(field_positions + bar_width, all_field_names)
                plt.legend()
                
                # Save the plot
                stats_path = output_dir / f"{split}_stats.png"
                plt.savefig(stats_path)
                split_stats_paths[split] = str(stats_path)
        
        plt.close()
    
    result = {"split_stats": str(output_dir)}
    if mean_lengths_path:
        result["mean_lengths"] = str(mean_lengths_path)
    
    return result


def calculate_quality_metrics(
    stats: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, float]:
    """
    Calculate quality metrics for training data.

    Args:
        stats: Statistics for each split.

    Returns:
        Dict[str, float]: Dictionary with quality metrics.
    """
    logger.info("Calculating quality metrics")
    
    # Initialize metrics
    metrics = {
        "input_output_ratio": 0.0,
        "split_balance": 0.0,
        "length_variability": 0.0,
        "overall_quality": 0.0,
    }
    
    # Calculate input/output ratio
    input_lengths = []
    output_lengths = []
    
    for split, split_stats in stats.items():
        if "input_length" in split_stats and "output_length" in split_stats:
            input_lengths.append(split_stats["input_length"]["mean"])
            output_lengths.append(split_stats["output_length"]["mean"])
    
    if input_lengths and output_lengths:
        avg_input_length = statistics.mean(input_lengths)
        avg_output_length = statistics.mean(output_lengths)
        
        if avg_input_length > 0:
            metrics["input_output_ratio"] = avg_output_length / avg_input_length
    
    # Calculate split balance
    split_counts = []
    
    for split, split_stats in stats.items():
        if "input_length" in split_stats:
            split_counts.append(split_stats["input_length"]["count"])
    
    if split_counts:
        total_count = sum(split_counts)
        if total_count > 0:
            # Calculate entropy of split distribution
            split_probs = [count / total_count for count in split_counts]
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in split_probs)
            max_entropy = np.log2(len(split_counts))
            
            if max_entropy > 0:
                metrics["split_balance"] = entropy / max_entropy
    
    # Calculate length variability
    length_stds = []
    
    for split, split_stats in stats.items():
        if "input_length" in split_stats and "output_length" in split_stats:
            length_stds.append(split_stats["input_length"]["std"])
            length_stds.append(split_stats["output_length"]["std"])
    
    if length_stds:
        metrics["length_variability"] = statistics.mean(length_stds) / 100  # Normalize
    
    # Calculate overall quality
    metrics["overall_quality"] = (
        0.3 * min(metrics["input_output_ratio"], 1.0) +
        0.3 * metrics["split_balance"] +
        0.4 * min(metrics["length_variability"], 1.0)
    )
    
    return metrics


def generate_quality_report(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    include_plots: bool = True,
) -> Dict[str, Union[Dict, str]]:
    """
    Generate a quality report for training data.

    Args:
        data_dir: Path to the directory containing training data.
        output_dir: Directory to save the report.
        include_plots: Whether to include plots in the report.

    Returns:
        Dict[str, Union[Dict, str]]: Dictionary with the report.
    """
    logger.info(f"Generating quality report for {data_dir}")
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze training data
    stats = analyze_training_data(data_dir)
    
    # Calculate quality metrics
    metrics = calculate_quality_metrics(stats)
    
    # Create report
    report = {
        "stats": stats,
        "metrics": metrics,
    }
    
    # Generate plots
    if include_plots:
        plots_dir = output_dir / "plots"
        plot_paths = plot_length_distributions(stats, plots_dir)
        report["plots"] = plot_paths
    
    # Save report
    report_path = output_dir / "quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Quality report saved to {report_path}")
    
    return {
        "report": report,
        "report_path": str(report_path),
    }


def generate_html_report(
    report: Dict[str, Union[Dict, str]],
    output_dir: Union[str, Path],
) -> str:
    """
    Generate an HTML report from a quality report.

    Args:
        report: Quality report.
        output_dir: Directory to save the HTML report.

    Returns:
        str: Path to the HTML report.
    """
    logger.info(f"Generating HTML report in {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data from report
    stats = report["report"]["stats"]
    metrics = report["report"]["metrics"]
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Data Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric {{ font-weight: bold; }}
            .good {{ color: green; }}
            .medium {{ color: orange; }}
            .poor {{ color: red; }}
            .plot {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Training Data Quality Report</h1>
        
        <h2>Quality Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Rating</th>
            </tr>
            <tr>
                <td>Input/Output Ratio</td>
                <td>{metrics["input_output_ratio"]:.2f}</td>
                <td class="{get_rating(metrics["input_output_ratio"], 0.5, 2.0)}">
                    {get_rating_text(metrics["input_output_ratio"], 0.5, 2.0)}
                </td>
            </tr>
            <tr>
                <td>Split Balance</td>
                <td>{metrics["split_balance"]:.2f}</td>
                <td class="{get_rating(metrics["split_balance"], 0.7, 0.9)}">
                    {get_rating_text(metrics["split_balance"], 0.7, 0.9)}
                </td>
            </tr>
            <tr>
                <td>Length Variability</td>
                <td>{metrics["length_variability"]:.2f}</td>
                <td class="{get_rating(metrics["length_variability"], 0.3, 0.7)}">
                    {get_rating_text(metrics["length_variability"], 0.3, 0.7)}
                </td>
            </tr>
            <tr>
                <td class="metric">Overall Quality</td>
                <td class="metric">{metrics["overall_quality"]:.2f}</td>
                <td class="metric {get_rating(metrics["overall_quality"], 0.5, 0.8)}">
                    {get_rating_text(metrics["overall_quality"], 0.5, 0.8)}
                </td>
            </tr>
        </table>
        
        <h2>Statistics by Split</h2>
    """
    
    # Add statistics for each split
    for split, split_stats in stats.items():
        html_content += f"""
        <h3>{split.capitalize()} Split</h3>
        <table>
            <tr>
                <th>Field</th>
                <th>Count</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std</th>
            </tr>
        """
        
        for field, field_stats in split_stats.items():
            html_content += f"""
            <tr>
                <td>{field.replace("_", " ").title()}</td>
                <td>{field_stats["count"]}</td>
                <td>{field_stats["min"]:.0f}</td>
                <td>{field_stats["max"]:.0f}</td>
                <td>{field_stats["mean"]:.2f}</td>
                <td>{field_stats["median"]:.2f}</td>
                <td>{field_stats["std"]:.2f}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
    
    # Add plots if available
    if "plots" in report["report"]:
        html_content += """
        <h2>Plots</h2>
        """
        
        for plot_name, plot_path in report["report"]["plots"].items():
            if isinstance(plot_path, str) and plot_path.endswith(".png"):
                # Convert absolute path to relative path
                rel_path = os.path.relpath(plot_path, output_dir)
                html_content += f"""
                <div class="plot">
                    <h3>{plot_name.replace("_", " ").title()}</h3>
                    <img src="{rel_path}" alt="{plot_name}" />
                </div>
                """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML report
    html_path = output_dir / "quality_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {html_path}")
    
    return str(html_path)


def get_rating(value: float, medium_threshold: float, good_threshold: float) -> str:
    """
    Get a rating class based on a value.

    Args:
        value: Value to rate.
        medium_threshold: Threshold for medium rating.
        good_threshold: Threshold for good rating.

    Returns:
        str: Rating class.
    """
    if value >= good_threshold:
        return "good"
    elif value >= medium_threshold:
        return "medium"
    else:
        return "poor"


def get_rating_text(value: float, medium_threshold: float, good_threshold: float) -> str:
    """
    Get a rating text based on a value.

    Args:
        value: Value to rate.
        medium_threshold: Threshold for medium rating.
        good_threshold: Threshold for good rating.

    Returns:
        str: Rating text.
    """
    if value >= good_threshold:
        return "Good"
    elif value >= medium_threshold:
        return "Medium"
    else:
        return "Poor"
